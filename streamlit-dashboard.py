import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Load dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

# Load and embed documents
raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

# Semantic search + filter
def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 50, final_top_k: int = 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

# Recommend and format
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"**{row['title']}** by *{authors_str}*  \n{truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


st.set_page_config(page_title="Semantic Book Recommender", layout="wide")
st.title("📚 Semantic Book Recommender")

query = st.text_input("Enter a description of a book you'd like to find:", "A story about forgiveness")

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

col1, col2 = st.columns(2)
category = col1.selectbox("Choose a category:", categories, index=0)
tone = col2.selectbox("Choose an emotional tone:", tones, index=0)

if st.button("🔍 Find Recommendations"):
    recs = recommend_books(query, category, tone)
    st.subheader("Recommended Books")
    for i in range(0, len(recs), 4):  # 4 thumbnails per row
        cols = st.columns(4)
        for j, (image_url, caption) in enumerate(recs[i:i+4]):
            with cols[j]:
                st.image(image_url, use_column_width="always")
                st.markdown(caption)
