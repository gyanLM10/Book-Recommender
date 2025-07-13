Semantic Book Recommender 📚
A Python project that builds an intelligent book recommendation system using LLMs, vector search, and a Gradio-based web app.

🔍 Features
Text Data Cleaning

Load and explore Kaggle book dataset

Handle missing values, long-tail categories, and data quality

Semantic Search

Split book descriptions

Embed text using HuggingFaceEmbeddings or OpenAIEmbeddings

Store embeddings in Chroma vector database

Zero-Shot Text Classification

Classify books as fiction or non-fiction using LLMs

Sentiment & Emotion Analysis

Assign tone labels (e.g., joy, sadness, suspense) with zero-shot sentiment analysis

Interactive Web UI with Gradio

Users can input natural language queries, filter by category and emotion, and view book covers, titles, descriptions, and scores

📊 Evaluation Metrics
Accuracy: 77.8%

Precision@k: 0.50

Recall@k: 0.80

F1@k: 0.58

🚀 Run the App Locally
To run the app on your local machine, execute the main dashboard Python file using your Python environment.

🐳 Run with Docker (Recommended)
To use Docker:

First, build the Docker image for the book recommender app.

Then, run the Docker container and map the correct port for accessing the web interface.

This will launch the app in an isolated environment, accessible from your browser.