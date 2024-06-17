import os
from datetime import datetime

import requests
import streamlit as st
from apify_client import ApifyClient
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self,
        container: st.delta_generator.DeltaGenerator,
        initial_text: str = "",
    ):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


class ArticleScraper:
    def __init__(self, article_url: str):
        load_dotenv()

        self.url = article_url.strip()
        self.article_namespace = extract_namespace(self.url)
        self.gemini_key = os.getenv("GOOGLE_API_KEY")
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index = "study-buddy-test"
        self.apify_token = os.environ.get("APIFY_API_TOKEN")
        self.apify_actor = "apify/website-content-crawler"

        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

        self.apify_client = ApifyClient(self.apify_token)
        self.pinecone_client = Pinecone(api_key=self.pinecone_key)
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            google_api_key=self.gemini_key,
            model="models/text-embedding-004",
            task_type="retrieval_document",
            title="article scrapping",
        )

    def check_loaded_documents(self):
        actor = self.apify_actor.replace("/", "~")
        actor_url = (
            f"https://api.apify.com/v2/acts/{actor}/runs/last/dataset/items"
        )
        querystring = {"token": self.apify_token}
        response = requests.get(actor_url, params=querystring)
        if response.status_code == 200:
            crawled_data = response.json()
            crawled_data_url = crawled_data[0]["url"]
            if crawled_data_url == self.url:
                return crawled_data
        return None

    def load_documents(self):
        st.write("Checking documents on apify...")
        check_last_run = self.check_loaded_documents()
        if isinstance(check_last_run, dict):
            loader_list = []
            for data in check_last_run:
                doc = Document(
                    page_content=data.get("text", ""),
                    metadata={"source": data.get("url")},
                )
                loader_list.append(doc)
            return loader_list

        st.write("Loading article...")
        actor_run_info = self.apify_client.actor(self.apify_actor).call(
            run_input={"startUrls": [{"url": self.url}]}
        )
        loader = ApifyDatasetLoader(
            dataset_id=actor_run_info["defaultDatasetId"],
            dataset_mapping_function=lambda item: Document(
                page_content=item["text"] or "",
                metadata={"source": item["url"]},
            ),
        )
        return loader.load()

    def generate_chunks(self):
        st.write("Loading documents...")
        documents = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=10
        )
        st.write("Splitting documents...")
        chunks = text_splitter.split_documents(documents)
        return chunks

    def start_vectordb(self):
        self.vectordb = PineconeVectorStore(
            pinecone_api_key=self.pinecone_key,
            embedding=self.embedding_function,
            index_name=self.pinecone_index,
            namespace=self.article_namespace,
        )

    def check_pinecone_db(self):
        if self.pinecone_index in self.pinecone_client.list_indexes().names():
            self.start_vectordb()
            index_stats = self.pinecone_client.Index(
                name=self.pinecone_index
            ).describe_index_stats()
            namespaces = index_stats.get("namespaces")
            if isinstance(namespaces, dict) is False:
                return 0
            article_vector_count = namespaces.get(self.article_namespace, 0)
            if isinstance(article_vector_count, int):
                return 0
            return article_vector_count.get("vector_count")

        self.pinecone_client.create_index(
            name=self.pinecone_index,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        self.start_vectordb()
        return 0

    def save_to_pinecone(self):
        st.write("Checking Pinecone for existing namespace...")
        print(datetime.now(), "Checking Pinecone for existing namespace...")
        check_vector_db = self.check_pinecone_db()
        if check_vector_db > 0:
            st.write(f"Data already loaded into {check_vector_db} vectors!")
            print(datetime.now(), "Data already loaded!")
        else:
            st.write("Generating chunks...")
            print(datetime.now(), "Generating chunks")
            chunks = self.generate_chunks()
            st.write("Saving to Pinecone...")
            print(datetime.now(), "Saving to Pinecone")
            self.vectordb.from_documents(
                documents=chunks,
                embedding=self.embedding_function,
                index_name=self.pinecone_index,
                namespace=self.article_namespace,
            )

    def generate_context(self, question):
        self.start_vectordb()
        results = self.vectordb.similarity_search(question, k=4)
        if len(results) == 0:
            print("Unable to find matching results.")
            return None
        return results


def click_button():
    st.session_state.clicked = True


def extract_namespace(url_path):
    if url_path is None or url_path == "":
        return ""
    domain = str(url_path).split(".")[1]
    title = str(url_path).split("/")[-1] if "/" in str(url_path) else "no-title"
    return f"{domain}-{title}"


CONTEXTUALIZE_QUESTION_SYSTEM_PROMPT = """
Given the chat history and the latest user question, which might reference
context in the chat history, formulate a standalone question which can be
undestood without the chat history. Do not answer the question, just reformulate
it if needed, otherwise return it as is.
"""

QA_SYSTEM_PROMPT = """
You are a personalized assistant for question answering tasks for students. Use
the following retrieved context to answer the question. If you don't know the
answer, say that you don't know. Keep the answer concise.

Context:
<<<
{context}
>>>

Question:
<<<
{question}
>>>
"""


st.set_page_config(
    page_title="Study Buddy",
    page_icon="🦾",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.title("Article Scraper")
st.write("Enter an article and start chatting with it using Google Gemini!")
model = st.radio(
    "Select model:",
    [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    index=0,
)
temperature = st.slider(
    label="Select model's temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.1,
)
llm_model = ChatGoogleGenerativeAI(
    model=model,
    temperature=temperature,
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

url = st.text_input("Enter the URL of the article you want to study:")
if "clicked" not in st.session_state:
    st.session_state.clicked = False

if "scraped" not in st.session_state:
    st.session_state.scraped = False

st.button("Scrape", on_click=click_button)

if st.session_state.clicked:
    scraper = ArticleScraper(article_url=url)
    with st.status("Scraping article...") as status:
        scraper = ArticleScraper(article_url=url)
        scraper.save_to_pinecone()
        status.update(label="Article scraped!", state="complete")
    st.session_state.scraped = True
    st.session_state.clicked = False

st.divider()

if st.session_state.scraped:
    output_parser = StrOutputParser()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("Clear message history"):
        st.session_state.messages = []
    if len(st.session_state.messages) == 0:
        st.session_state.messages.append(
            {"type": "assistant", "content": "How can i help you today?"}
        )

    for msg in st.session_state.messages:
        st.chat_message(msg["type"]).write(msg["content"])

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)
        st.session_state.messages.append(
            {"type": "user", "content": user_query}
        )

        scraper = ArticleScraper(article_url=url)

        results = scraper.generate_context(user_query)
        context = "\n---\n".join([doc.page_content for doc in results])
        sources = [doc.metadata.get("source", None) for doc in results]

        qa_template = ChatPromptTemplate.from_template(QA_SYSTEM_PROMPT)
        prompt = qa_template.format(context=context, question=user_query)
        print(prompt)
        chain = qa_template | llm_model | output_parser

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = chain.invoke(
                input={"context": context, "question": user_query},
                config={"configurable": {"session_id": "test_1"}},
            )

            st.write(response)
            st.session_state.messages.append(
                {"type": "assistant", "content": response}
            )
