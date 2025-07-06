
## Problem Statement: - Just for me
"""
- YouTube summarizer — an extension in any browser where it takes a YouTube URL and summarizes it for you according to your need.
- You can also ask questions if you have any doubts.
- Build a standalone pipeline first — not necessary to add to tools or agent.
"""

# -----------------------------
# GLOBALS & IMPORTS
# -----------------------------
# URL = "https://www.youtube.com/watch?v=5CJA1Hbutqc&t=128s"  # Sample YouTube URL

from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env file

import os
from hashlib import md5

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# INIT MODEL & EMBEDDING
# -----------------------------
# Set up LLM and embedding models
model = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")
embeds = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# -----------------------------
# PROMPT TEMPLATE
# -----------------------------
SYSTEM_TEMPLATE = """
The user is 10 years old but he is super smart and does work like a 25-year-old. 
However, he finds it difficult to understand some terms. 
- The answer should be straight forward. NO bs.
- If the context doesn't contain any relevant information to the question, explain the reason and say it's not included in the video. 


<context>
{context}
</context>

Question: {input}
"""

qna_prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)


# -----------------------------
# LOAD AND SPLIT VIDEO
# -----------------------------
def load_n_split_video(url: str):
    """
    Loads the transcript from a YouTube video and splits it into smaller chunks.
    """
    youtube_doc = YoutubeLoader.from_youtube_url(url ).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(youtube_doc)


# -----------------------------
# GENERATE SUMMARY or ANSWER - UPDATED OLD CODE - MAKE CODE INTO REUSABLE ONES
# -----------------------------
def get_vectorstore(url:str, split_docs):
    video_id = md5(url.encode()).hexdigest()
    db_dir = f"./db/{video_id}"

    # Store it in vector - CHROMA
    if os.path.exists(db_dir):
        vectordb = Chroma(persist_directory=db_dir, embedding_function=embeds)
    else:
        vectordb = Chroma.from_documents(split_docs, embedding=embeds, persist_directory=db_dir)
    return vectordb


def answer_query(url: str, query: str):
    # Load and split transcript
    split_docs = load_n_split_video(url)
    #vector store
    vectordb = get_vectorstore(url, split_docs)
    # Build RAG chain
    retriever = vectordb.as_retriever(k=4)
    document_chain = create_stuff_documents_chain(model, qna_prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    return rag_chain.invoke({"input": query})

# ---------------------------------------
# ALL IN ONE CODE - OLD - WORKED FINE
#----------------------------------------

# def generate_summary(url: str, query: str):
#     """
#     Given a video URL and a query, generate a summary or answer using RAG.
#     Uses persistent vector storage for speed.
#     """
#     # Load and split transcript
#     split_docs = load_n_split_video(url)
#
#     # Create unique hash ID for the video URL
#     video_id = md5(url.encode()).hexdigest()
#     db_dir = f"./db/{video_id}"
#
#     # Store it in vector - CHROMA
#     if os.path.exists(db_dir):
#         vectordb = Chroma(persist_directory=db_dir, embedding_function=embeds)
#     else:
#         vectordb = Chroma.from_documents(split_docs, embedding=embeds, persist_directory=db_dir)
#
#     # Build RAG chain
#     retriever = vectordb.as_retriever(k=4)
#     document_chain = create_stuff_documents_chain(model, qna_prompt)
#     rag_chain = create_retrieval_chain(retriever, document_chain)
#
#     return rag_chain.invoke({"input": query})


# -----------------------------
# RUN TEST CASE
# -----------------------------
# print("Here's the output:\n")
#
# query = "Who is the founder of langchain, answer in 1 line"
# response = generate_summary(URL, query)
#
# print("🤖 FINAL ANSWER:")
# print(response["answer"])
# print("-" * 50)
# print("📄 CONTEXT USED:")
# print("length", len(response))
# print(response)

# -----------------------------
# RUN TEST CASE - ONE TO ONE CHATBOT
# -----------------------------
print("\n🎬 Welcome to the YouTube Chatbot!")
url = input("📺 Enter the YouTube video URL: ")

print("💬 Ask anything about the video. Type 'exit' to stop.\n")

while True:
    query = input("🧒 YOU: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Goodbye!")
        break

    response = answer_query(url, query)

    print("\n🤖 BOT:", response["answer"])
    print("-" * 30)
    print("\n🤖 BOT:", response)



# -----------------------------
# ✅ YOUTUBE SUMMARIZER BACKEND - DONE
# ✅ FURTHER TASKS
# 1. CLEAN THE CODE THAT HAVE FUNCTIONS AND INPUT          ✅ DONE
# 2. MAKE IT INTO THE AGENT OR CHATBOT TYPE                ✅ DONE
# 3. Deploy it using Flask
# -----------------------------
