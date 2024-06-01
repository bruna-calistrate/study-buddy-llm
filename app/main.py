import os

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
    def __init__(self, article_url):
        load_dotenv()

        self.url = article_url
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.pinecone_key = os.getenv("PINECONE_API_KEY")
        self.apify_token = os.environ.get("APIFY_API_TOKEN")
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            google_api_key=self.gemini_key,
            model="models/text-embedding-004",
            task_type="retrieval_document",
        )
        self.vectordb = PineconeVectorStore(
            pinecone_api_key=self.pinecone_key,
            embedding=self.embedding_function,
            index_name="study-buddy-test",
        )

    def scrape_article(self):
        apify_client = ApifyClient(self.apify_token)
        print(f"Extracting data from '{self.url}'. Please wait...")
        actor_run_info = apify_client.actor(
            "apify/website-content-crawler"
        ).call(run_input={"startUrls": [{"url": self.url}]})
        print("Saving data into the vector database. Please wait...")
        loader = ApifyDatasetLoader(
            dataset_id=actor_run_info["defaultDatasetId"],
            dataset_mapping_function=lambda item: Document(
                page_content=item["text"] or "",
                metadata={"source": item["url"]},
            ),
        )
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        self.vectordb.from_documents(
            documents=docs, 
            embedding=self.embedding_function,
            index_name="study-buddy-test",
        )
        print("All done!")

    @st.cache_resource(ttl="1h")
    def get_retriever(self):
        return self.vectordb.as_retriever(search_type="mmr")

    def create_conversation_chain(self, model):
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=msgs, return_messages=True
        )
        llm_model = ChatGoogleGenerativeAI(
            model=f"model/{model}",
            temperature=0,
            google_api_key=self.gemini_key,
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm_model,
            retriever=self.get_retriever(),
            memory=memory,
            verbose=False,
        )
        return msgs, qa_chain


st.title("Study Buddy - Article Scrapping")
st.caption("This app allows you to scrape a website using Google Gemini.")
model = st.radio(
    "Select model:",
    [
        "gemini-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ],
    index=0,
)
url = st.text_input("Enter the URL of the article you want to study")

if st.button("Scrape"):
    scraper = ArticleScraper(article_url=url)
    scraper.scrape_article()

    msgs, qa_chain = scraper.create_conversation_chain(model=model)

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
            response = qa_chain.run(user_query, callbacks=[stream_handler])
