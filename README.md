# Semantic Book Recommender 📚

An intelligent book recommendation system utilizing Large Language Models (LLMs), semantic vector search, zero-shot classification, and a sleek Gradio web interface. The application processes book descriptions to identify contextual meaning, thematic elements, and emotional tones, providing much richer recommendations than traditional genre-based or keyword-based filtering.

![App Screenshot](./Screenshot\ 2025-06-29\ at\ 7.21.54\ PM.png)

## 🏗️ Project Architecture

To ensure separation of concerns, the project is organized into logical components:

```
BookRecommender/
├── app/
│   └── app.py                        # The main Gradio dashboard and semantic search logic  
├── data/
│   ├── books_cleaned.csv             # Processed dataset with core book metadata
│   ├── books_with_categories.csv     # Extracted and simplified genre classifications
│   ├── books_with_emotions.csv       # Emotion analysis scores (joy, fear, surprise, etc.)
│   ├── tagged_description.txt        # Output corpus for vector embedding
│   └── .env.py                       # Data config vars
├── notebooks/
│   ├── data_exploration.ipynb        # Data preparation and cleaning sandbox  
│   ├── sentiment-analysis.ipynb      # Emotion classification modeling
│   ├── text-classification.ipynb     # Category inference modeling
│   ├── vector_search.ipynb           # Vector DB similarity exploration
│   └── sample.ipynb                  # Experimental scratchpad
├── .gradio/
│   └── accuracy.ipynb                # Precision/Recall/F1 evaluation tools
├── Dockerfile                        # Container setup for reproducible deployments
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## ✨ Core Features

1. **Semantic Search via Embeddings**: 
   Leverages `langchain` and `HuggingFaceEmbeddings` (specifically `sentence-transformers/all-MiniLM-L6-v2`) to convert book descriptions into high-dimensional vector representations.
2. **Vector Retrieval**: 
   Stores text chunks in a `Chroma` vector database for blazing-fast similarity searches based on semantic meaning rather than exact keyword matches.
3. **Sentiment & Emotion Awareness**: 
   Each book description has been pre-analyzed for emotional tones (e.g., Happy, Surprising, Angry, Suspenseful, Sad). Users can explicitly filter their recommendations to fit a specific mood.
4. **Classification Filtering**: 
   Filters results dynamically based on `simple_categories`.
5. **Interactive UI**: 
   A responsive, glassmorphism-themed Gradio gallery interface where users input natural language prompts like *"A story about forgiveness"* and immediately receive visually rich book covers and summaries.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed and pip installed. We strongly recommend using a virtual environment.

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/BookRecommender.git
cd BookRecommender
pip install -r requirements.txt
```

*(Note: If you run into any LangChain deprecation warnings referring to `langchain-huggingface`, you can safely ignore them as the application maintains backward compatibility with `< 1.0`)*

### 2. Environment Setup

Create a `.env` file in the root of the project. If you intend to swap the embedding model to OpenAI, add your key here. For standard local HuggingFace embedding execution, this file can remain empty:

```bash
touch .env
```

### 3. Running the App Locally

Start the Gradio web server:

```bash
python app/app.py
```

The app will initialize the Chroma database and load the embeddings (this may take 15-30 seconds on the first run). Once ready, you will see a local URL printed in your terminal (typically `http://127.0.0.1:7860`). Open that URL in your web browser.

---

## 🐳 Running with Docker

For a guaranteed isolated and reproducible environment, use Docker:

```bash
# Build the image
docker build -t book-recommender .

# Run the container (mapping internal Gradio port 7860 to external port 7860)
docker run --env-file .env -p 7860:7860 book-recommender
```

Visit `http://localhost:7860` in your browser.

---

## 📊 Evaluation & Accuracy

A benchmark was run internally in `.gradio/accuracy.ipynb` evaluating the engine's ability to recreate a user's collection from a single favorite title seed. 

- **Accuracy**: 77.8%
- **Precision@k**: 0.50
- **Recall@k**: 0.833
- **F1@k**: 0.583