import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

class CollaborativeFiltering:
    def __init__(self, ratings_path):
        """
        Initializes the collaborative filtering model using Matrix Factorization (TruncatedSVD).
        """
        print(f"Loading user ratings from {ratings_path}...")
        self.ratings = pd.read_csv(ratings_path)
        
        # Create User-Item Matrix
        print("Creating User-Item matrix...")
        self.user_item_matrix = self.ratings.pivot(index='user_id', columns='isbn13', values='rating').fillna(0)
        self.user_ids = self.user_item_matrix.index.tolist()
        self.item_isbns = self.user_item_matrix.columns.astype(str).tolist()
        
        # Matrix Factorization
        print("Training Matrix Factorization model (TruncatedSVD)...")
        # Ensure n_components is not larger than matrix dimensions
        n_components = min(50, min(self.user_item_matrix.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Fit and transform the user-item matrix
        self.user_factors = self.svd.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd.components_.T
        
        # Reconstruct the matrix to get predicted ratings for all user-item pairs
        print("Calculating predicted ratings...")
        self.predicted_ratings = np.dot(self.user_factors, self.item_factors.T)
        self.predicted_ratings_df = pd.DataFrame(
            self.predicted_ratings, 
            index=self.user_ids, 
            columns=self.item_isbns
        )
        print("Collaborative Filtering model ready!")

    def get_cf_scores(self, user_id, candidate_isbns=None):
        """
        Returns a dictionary of CF scores for the given user and candidate ISBNs.
        If candidate_isbns is None, returns scores for all items.
        """
        if user_id not in self.user_ids:
            # User not found (Cold start problem), return 0 scores
            return {isbn: 0.0 for isbn in (candidate_isbns if candidate_isbns else self.item_isbns)}
            
        user_preds = self.predicted_ratings_df.loc[user_id]
        
        if candidate_isbns:
            # Filter predictions to only the candidate ISBNs
            # If an ISBN is not in the training data, assign 0
            candidate_isbns = [str(isbn) for isbn in candidate_isbns]
            scores = {}
            for isbn in candidate_isbns:
                scores[isbn] = user_preds.get(isbn, 0.0)
            return scores
        else:
            return user_preds.to_dict()
