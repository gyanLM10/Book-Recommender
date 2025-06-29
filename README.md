# Semantic Book Recommender 📚

A Python project that builds an intelligent book recommendation system using LLMs, vector search, and a Gradio-based web app.

---

##  Features

1. **Text Data Cleaning**  
   - Load and explore Kaggle book dataset  
   - Handle missing values, long-tail categories, and data quality

2. **Semantic Search**  
   - Split book descriptions  
   - Embed text using `HuggingFaceEmbeddings` or `OpenAIEmbeddings`  
   - Store embeddings in Chroma vector database

3. **Zero-Shot Text Classification**  
   - Classify books as **fiction** or **non-fiction** using LLMs

4. **Sentiment & Emotion Analysis**  
   - Assign tone labels (e.g., joy, sadness, suspense) with zero-shot sentiment analysis

5. **Interactive Web UI with Gradio**  
   - Build a frontend allowing users to:
     - Input natural language queries  
     - Filter by category and emotion  
     - View book covers, titles, descriptions, and scores

---## 🚀 Run the App

To launch the semantic book recommender dashboard, run the following command:

```bash
python gradio-dashboard.py
