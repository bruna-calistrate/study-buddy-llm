import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import nest_asyncio
import streamlit as st
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph

nest_asyncio.apply()

load_dotenv()
executor = ThreadPoolExecutor()
gemini_key = os.getenv("GEMINI_API_KEY")


async def run_thread(blocking_func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, blocking_func, *args, **kwargs)


async def scrape_source(llm_model, prompt, source):
    graph_config = {
        "llm": {
            "api_key": gemini_key,
            "model": llm_model,
            "temperature": 0.3,
            "streaming": False,
        }
    }
    smart_scraper_graph = SmartScraperGraph(
        prompt=prompt, source=source, config=graph_config
    )
    return await run_thread(smart_scraper_graph.run)


def run_article_app():
    st.title("Study Buddy - Article Scrapping")
    st.caption("This app allows you to scrape a website using Google Gemini.")

    model = st.radio(
        "Select model:",
        [
            "gemini-pro",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "text-embedding-004",
        ],
        index=0,
    )
    url = st.text_input("Enter the URL of the article you want to study")
    user_prompt = st.text_input("What do you want to know from this article?")

    if st.button("Scrape"):
        async def run_scrape():
            return await scrape_source(
                prompt=user_prompt, llm_model=model, source=url
            )
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(run_scrape())
        st.write("Scraped Result:", result)


run_article_app()
