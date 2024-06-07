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

st.set_page_config(page_title="Study Buddy", page_icon="🦾")
st.title("Study Buddy - Article Scrapping")
st.caption(
    "This app allows you to scrape a website and chat with it using Google Gemini."
)
model = st.radio(
    "Select model:",
    [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    index=0,
)
llm_model = ChatGoogleGenerativeAI(
    model=model,
    temperature=0.3,
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

url = st.text_input("Enter the URL of the article you want to study")
if "clicked" not in st.session_state:
    st.session_state.clicked = False

st.button("Scrape", on_click=click_button)
if st.session_state.clicked:
    scraper = ArticleScraper(article_url=url)
    scraper.save_to_pinecone()
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

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             f"You are an AI chabot having a conversation with a human about the data extracted from {url}",
    #         ),
    #         MessagesPlaceholder(variable_name="history"),
    #         ("human", "{question}"),
    #     ]
    # )

    # conversation_chain = prompt | llm_model
    # chain_with_history = RunnableWithMessageHistory(
    #     conversation_chain,
    #     lambda session_id: msgs,
    #     input_messages_key="question",
    #     history_messages_key="history",
    # )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=retriever,
        return_source_documents=True,
        memory=memory,
        verbose=False,
    )

    # for msg in msgs.messages:
    #     st.chat_message(msg.type).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[stream_handler])
            # print(response)
        st.chat_message("assistant").write(response)

    # for msg in msgs.messages:
    #     st.chat_message(msg.type).write(msg.content)

    # if prompt := st.chat_input(placeholder="Ask me anything!"):
    #     st.chat_message("human").write(prompt)
    #     config = {"configurable": {"session_id": "unused"}}
    #     response = chain_with_history.invoke({"question": prompt}, config)
    #     st.chat_message("ai").write(response.content)
