import pandas as pd
import numpy as np
import random
import os

# Set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")
books_path = os.path.join(data_dir, "books_with_emotions.csv")
output_path = os.path.join(data_dir, "user_ratings.csv")

print("Loading books data...")
books = pd.read_csv(books_path)
books['isbn13'] = books['isbn13'].astype(str)

# We will create 1000 fictitious users
num_users = 1000
all_isbns = books['isbn13'].tolist()

# To make it realistic, group books by category
categories = books['simple_categories'].unique().tolist()
category_books = {}
for cat in categories:
    category_books[cat] = books[books['simple_categories'] == cat]['isbn13'].tolist()

ratings_data = []

print("Simulating interactions for 1000 users...")
np.random.seed(42)
random.seed(42)

for user_id in range(1, num_users + 1):
    # Each user has 1-3 favorite categories
    fav_cats = random.sample(categories, random.randint(1, 3))
    
    # User rates between 10 and 50 books
    num_ratings = random.randint(10, 50)
    
    user_rated = set()
    for _ in range(num_ratings):
        # 80% chance to rate a book from a favorite category, 20% random
        if random.random() < 0.8 and fav_cats:
            cat = random.choice(fav_cats)
            if not category_books[cat]:
                continue
            isbn = random.choice(category_books[cat])
        else:
            isbn = random.choice(all_isbns)
            
        if isbn not in user_rated:
            # Random rating from 1 to 5, biased towards higher ratings for fav categories
            if random.random() < 0.8:
                rating = random.choices([3, 4, 5], weights=[0.2, 0.4, 0.4])[0]
            else:
                rating = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.1, 0.2, 0.3, 0.3])[0]
            
            ratings_data.append({
                'user_id': user_id,
                'isbn13': isbn,
                'rating': rating
            })
            user_rated.add(isbn)

ratings_df = pd.DataFrame(ratings_data)
ratings_df.to_csv(output_path, index=False)
print(f"✅ Generated {len(ratings_df)} realistic ratings and saved to {output_path}")
