import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
import asyncio

## Streamlit App
st.set_page_config(page_title="Langchain: Summarize YouTube & Websites")
st.title("Langchain: Summarize YouTube & Websites")
st.subheader("Summarize URL")

## Sidebar Inputs
with st.sidebar:
    groq_api_key = st.text_input("Groq API KEY", value='', type="password")
    language = st.text_input("Select language for summarization")

generic_url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

## **Prompts**
map_prompt = PromptTemplate(
    template="Summarize the following text concisely:\n\n{text}",
    input_variables=['text']
)

combine_prompt = PromptTemplate(
    template="Combine these summaries into a structured final summary:\n\n{text}\n\nTranslate into {language}.",
    input_variables=['text', 'language']
)

## **Fetch YouTube Transcript (Async for Speed)**
async def fetch_youtube_transcript(video_url):
    try:
        video_id = video_url.split("v=")[-1]
        transcript = await asyncio.to_thread(YouTubeTranscriptApi.get_transcript, video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Error fetching transcript: {e}"

if st.button("Summarize"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Enter required fields")

    elif not validators.url(generic_url):
        st.error("Provide a valid URL")

    else:
        try:
            with st.spinner("Processing..."):
                ## Fetch Content
                if "youtube.com" in generic_url:
                    text_data = asyncio.run(fetch_youtube_transcript(generic_url))
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url])
                    loaded_docs = loader.load()
                    text_data = " ".join([doc.page_content for doc in loaded_docs])

                ## **Optimize Chunking for Speed**
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                chunks = text_splitter.split_text(text_data)
                docs = [Document(page_content=chunk) for chunk in chunks]

                ## Use map_reduce for faster summarization
                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt
                )

                output = chain.run({"input_documents": docs, "language": language})

                st.success(output)

        except Exception as e:
            st.exception(f"Exception: {e}")
