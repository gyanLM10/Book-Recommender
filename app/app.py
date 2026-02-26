import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

from collaborative_filtering import CollaborativeFiltering

import gradio as gr

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")

load_dotenv()

books = pd.read_csv(os.path.join(data_dir, "books_with_emotions.csv"))
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader(os.path.join(data_dir, "tagged_description.txt")).load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding_model)

# Initialize Cross-Encoder for re-ranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

cf_model = CollaborativeFiltering(os.path.join(data_dir, "user_ratings.csv"))

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        user_id: int = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    
    # Extract ISBNs and corresponding descriptions for re-ranking
    books_list = []
    cross_input = []
    for rec in recs:
        isbn = int(rec.page_content.strip('"').split()[0])
        books_list.append(isbn)
        # Pair the query with the document's content
        cross_input.append([query, rec.page_content])
        
    # Re-rank with CrossEncoder
    cross_scores = cross_encoder.predict(cross_input)
    
    # Store scores in a dictionary mapping ISBN -> Score, then pick the top final_top_k
    isbn_scores = {isbn: score for isbn, score in zip(books_list, cross_scores)}
    
    # Sort books by the new cross-encoder score
    ranked_isbns = sorted(isbn_scores.keys(), key=lambda x: isbn_scores[x], reverse=True)
    
    # Keep the top N from the re-ranker BEFORE applying category/tone filters to ensure 
    # we have enough high-quality candidates to filter through
    book_recs = books[books["isbn13"].isin(ranked_isbns)].copy()
    
    # Assign the new cross encoder score so we can sort by it later
    book_recs['cross_score'] = book_recs['isbn13'].map(isbn_scores)
    book_recs = book_recs.sort_values(by='cross_score', ascending=False)
    
    # Hybrid Scoring
    if user_id is not None and str(user_id).strip().isdigit():
        user_id = int(str(user_id).strip())
        cf_scores = cf_model.get_cf_scores(user_id, ranked_isbns)
        max_cf = max(cf_scores.values()) if cf_scores else 1.0
        min_cf = min(cf_scores.values()) if cf_scores else 0.0
        range_cf = (max_cf - min_cf) if (max_cf - min_cf) > 0 else 1.0
        
        book_recs['cf_score'] = book_recs['isbn13'].map(cf_scores).fillna(0)
        book_recs['cf_score'] = (book_recs['cf_score'] - min_cf) / range_cf
        c_min, c_max = book_recs['cross_score'].min(), book_recs['cross_score'].max()
        book_recs['cross_score_norm'] = (book_recs['cross_score'] - c_min) / (c_max - c_min + 1e-9)
        
        book_recs['hybrid_score'] = 0.5 * book_recs['cross_score_norm'] + 0.5 * book_recs['cf_score']
        book_recs = book_recs.sort_values(by='hybrid_score', ascending=False).head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str,
        user_id: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone, user_id=user_id)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        user_id = gr.Textbox(label = "User ID (Optional, 1-1000):",
                                placeholder = "e.g., 42")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown, user_id],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch(share=True)
