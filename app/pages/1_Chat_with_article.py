import os
from datetime import datetime

import requests
import streamlit as st
from apify_client import ApifyClient
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_community.document_loaders import ApifyDatasetLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
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

    def extract_namespace(self):
        domain = self.url.split(".")[1]
        title = self.url.split("/")[-1] if "/" in self.url else "no-title"
        return f"{domain}-{title}"

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
        documents = self.load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def start_vectordb(self):
        self.vectordb = PineconeVectorStore(
            pinecone_api_key=self.pinecone_key,
            embedding=self.embedding_function,
            index_name=self.pinecone_index,
            namespace=self.extract_namespace(),
        )

    def check_pinecone_db(self):
        if self.pinecone_index in self.pinecone_client.list_indexes().names():
            self.start_vectordb()
            index_stats = self.pinecone_client.Index(
                name=self.pinecone_index
            ).describe_index_stats()
            return index_stats.get("total_vector_count", 0)

        self.pinecone_client.create_index(
            name=self.pinecone_index,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        self.start_vectordb()
        return 0

    def save_to_pinecone(self):
        check_vector_db = self.check_pinecone_db()
        if check_vector_db > 0:
            print(datetime.now(), "All done, data already loaded!")
        else:
            chunks = self.generate_chunks()
            self.vectordb.from_documents(
                documents=chunks,
                embedding=self.embedding_function,
                index_name=self.pinecone_index,
                namespace=self.extract_namespace(),
            )


def click_button():
    st.session_state.clicked = True


def extract_namespace(url):
    domain = url.split(".")[1]
    title = url.split("/")[-1] if "/" in url else "no-title"
    return f"{domain}-{title}"

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = StreamlitChatMessageHistory(key="langchain_messages")
    return store[session_id]

@st.cache_resource(ttl="1h")
def get_retriever():
    embedding_function = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ["GOOGLE_API_KEY"],
        model="models/text-embedding-004",
        task_type="retrieval_document",
        title="article scrapping",
    )
    vectordb = PineconeVectorStore(
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
        embedding=embedding_function,
        index_name="study-buddy-test",
        namespace=extract_namespace(url),
    )
    return vectordb.as_retriever(search_type="mmr")

st.set_page_config(
    page_title="Study Buddy", 
    page_icon="🦾",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.title("Article Scraper")
st.write(
    "Enter an article and start chatting with it using Google Gemini!"
)
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
    step=0.1
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
    with st.spinner(text="Scraping article..."):
        scraper = ArticleScraper(article_url=url)
        scraper.save_to_pinecone()
    st.session_state.scraped = True
    st.session_state.clicked = False
    st.divider()

if st.session_state.scraped:
    retriever = get_retriever()
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=msgs,
        return_messages=True,
    )

    if st.sidebar.button("Clear message history"):
        msgs.clear()
    if len(msgs.messages) == 0:
        msgs.add_ai_message(f"Ask me anything about {url}!")

    contextualize_question_system_prompt = """
    Given the chat history and the latest user question, which might reference 
    context in the chat history, formulate a standalone question which can be 
    undestood without the chat history. Do not answer the question, just reformulate 
    it if needed, otherwise return it as is.
    """
    qa_system_prompt = """
    You are a personalized assistant for question answering tasks for students. Use 
    the following retrieved context to answer the question. If you don't know the 
    answer, say that you don't know. Keep the answer concise.
    
    Context: 
    <<<
    {context}
    >>>
    """
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_question_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm_model, retriever, contextualize_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="answer",
    )

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = conversational_rag_chain.invoke(
                input={"input": user_query},
                config={
                    "configurable": {"session_id": "test_1"}
                }
            )
            st.write(response["answer"])

