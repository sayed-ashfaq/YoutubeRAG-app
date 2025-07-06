# 📺 YouTube Summarizer & QnA using GenAI (RAG Pipeline)

A powerful GenAI application that **summarizes YouTube videos** and answers questions based on video content using **LLMs, embeddings, and semantic search**. Built using **LangChain**, **Gemini Pro**, **ChromaDB**, and deployed with **Flask**.

---

## 🚀 Features

- 🔍 **YouTube Transcript Extraction** – Automatically pulls transcripts from any YouTube video.
- 🧠 **RAG Pipeline (Retrieval-Augmented Generation)** – Uses vector search + LLMs to provide accurate, context-aware answers.
- 🧾 **Summarization + Q&A** – Ask any question about the video or request a summary.
- 💾 **Persistent Memory** – Stores vectorized chunks in ChromaDB using unique hashes.
- 🌐 **Flask REST API** – Backend ready for integration into web apps or platforms.

---

## 🧠 How It Works

1. **Load Transcript** from a YouTube video using `YoutubeLoader`.
2. **Split Transcript** into chunks using `RecursiveCharacterTextSplitter`.
3. **Generate Embeddings** using `GoogleGenerativeAIEmbeddings`.
4. **Store & Search** chunks using `Chroma` vector database.
5. **Query** through a `RetrievalQA` or `RetrievalChain` powered by Gemini Pro and LangChain.
6. **Respond** in a friendly tone customized for young learners using prompt templates.

---

## 🧪 Example Usage

```bash
python app.py
