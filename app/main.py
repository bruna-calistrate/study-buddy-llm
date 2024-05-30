import os

import streamlit as st
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph

load_dotenv()


gemini_key = os.getenv("GEMINI_API_KEY")

st.title("Study Buddy - Article Scrapping")
st.caption("This app allows you yo scrape a website using Google Gemini.")

if gemini_key:
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
    graph_config = {"llm": {"api_key": gemini_key, "model": model}}
    url = st.text_input("Enter the URL of the article you want to study")
    user_prompt = st.text_input("What do you want to know from this article?")

    smart_screaper_graph = SmartScraperGraph(
        prompt=user_prompt, source=url, config=graph_config
    )

    if st.button("Scrape"):
        result = smart_screaper_graph.run()
        st.write(result)
