#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")

print("Path to dataset files:", path)


# In[2]:


import pandas as pd


# In[3]:


books = pd.read_csv(f"{path}/books.csv")


# In[4]:


books


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[36]:


ax = plt.axes()
sns.heatmap(books.isna().transpose(), cbar=False, ax=ax)

plt.xlabel("Columns")
plt.ylabel("Missing Values")

plt.show()


# In[7]:


import numpy as np


# In[9]:


books["missing_description"] = np.where(books["description"].isna(), 1, 0)
books["age_of_book"] = 2024 - books["published_year"]


# In[12]:


columns_of_interest = ["num_pages", "age_of_book", "missing_description", "average_rating"]

correlation_matrix = books[columns_of_interest].corr(method="spearman")

sns.set_theme(style="white")
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Spearman correlation"})
heatmap.set_title("Correlation heatmap")
plt.show()


# In[39]:


book_missing = books[~(books["description"].isna()) &
      ~(books["num_pages"].isna()) &
      ~(books["average_rating"].isna()) &
      ~(books["published_year"].isna())
]


# In[40]:


book_missing


# In[48]:


book_missing["categories"].value_counts().reset_index().sort_values("count", ascending=False)


# In[49]:


book_missing


# In[50]:


book_missing["words_in_description"] = book_missing["description"].str.split().str.len()


# In[51]:


book_missing


# In[52]:


book_missing.loc[book_missing["words_in_description"].between(1, 4), "description"]


# In[53]:


book_missing.loc[book_missing["words_in_description"].between(5, 14), "description"]


# In[54]:


book_missing_25_words = book_missing[book_missing["words_in_description"] >= 25]


# In[55]:


book_missing_25_words


# In[56]:


book_missing_25_words["title_and_subtitle"] = (
    np.where(book_missing_25_words["subtitle"].isna(), book_missing_25_words["title"],
             book_missing_25_words[["title", "subtitle"]].astype(str).agg(": ".join, axis=1))
)
book_missing_25_words


# In[57]:


book_missing_25_words["tagged_description"] = book_missing_25_words[["isbn13", "description"]].astype(str).agg(" ".join, axis=1)
book_missing_25_words


# In[58]:


(
    book_missing_25_words
    .drop(["subtitle", "missing_description", "age_of_book", "words_in_description"], axis=1)
    .to_csv("books_cleaned.csv", index = False)
)

