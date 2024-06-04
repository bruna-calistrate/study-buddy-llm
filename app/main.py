import os
from datetime import datetime

import requests
import streamlit as st
from apify_client import ApifyClient
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.conversational_retrieval.base import (
    ConversationalRetrievalChain,
)
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_community.document_loaders import ApifyDatasetLoader
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
        print(
            datetime.now(), f"Extracting data from '{self.url}'. Please wait..."
        )
        check_last_run = self.check_loaded_documents()
        if isinstance(check_last_run, dict):
            print(datetime.now(), "Loading documents. Please wait...")
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
        print(datetime.now(), "Loading documents. Please wait...")
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

        print(
            datetime.now(),
            f"Split {len(documents)} documents into {len(chunks)} chunks.",
        )
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
            print(datetime.now(), "All done!")

    @st.cache_resource(ttl="1h")
    def get_retriever(self):
        return self.vectordb.as_retriever(search_type="mmr")

    def create_conversation_chain(self, gemini_model):
        messages = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=messages,
            return_messages=True,
        )
        llm_model = ChatGoogleGenerativeAI(
            model=f"model/{gemini_model}",
            temperature=0.3,
            google_api_key=self.gemini_key,
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm_model,
            retriever=self.get_retriever(),
            memory=memory,
            verbose=False,
        )
        return messages, qa_chain


st.title("Study Buddy - Article Scrapping")
st.caption("This app allows you to scrape a website using Google Gemini.")
model = st.radio(
    "Select model:",
    [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    index=0,
)
url = st.text_input("Enter the URL of the article you want to study")

if st.button("Scrape"):
    scraper = ArticleScraper(article_url=url)
    scraper.save_to_pinecone()

    msgs, qa = scraper.create_conversation_chain(gemini_model=model)

    if st.sidebar.button("Clear message history") or len(msgs.messages) == 0:
        msgs.clear()
        msgs.add_ai_message(f"Ask me anything about {url}!")

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = qa.run(user_query, callbacks=[stream_handler])
