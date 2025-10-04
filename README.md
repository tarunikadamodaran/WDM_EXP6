### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 26.09.2025
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```
# --- Step 1: Documents ---
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?",
}

# --- Step 2: Preprocessing ---
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
    return " ".join(tokens)

preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

# --- Step 3: Term Frequencies (TF) ---
terms = sorted(set(" ".join(preprocessed_docs.values()).split()))
tf_data = {term: [] for term in terms}
for doc in preprocessed_docs.values():
    doc_tokens = doc.split()
    for term in terms:
        tf_data[term].append(doc_tokens.count(term))

tf_df = pd.DataFrame(tf_data, index=documents.keys())
print("\nTerm Frequencies (TF)")
display(tf_df)

# --- Step 4: DF & IDF ---
df = (tf_df > 0).sum(axis=0)
idf = np.log((len(documents) / df)) + 1  # smooth IDF
df_idf_df = pd.DataFrame({"Document Frequency (DF)": df, "Inverse Document Frequency (IDF)": idf})
print("\nDocument Frequency (DF) and Inverse Document Frequency (IDF)")
display(df_idf_df)

# --- Step 5: TF-IDF Weights ---
tfidf_matrix = tf_df * idf
tfidf_matrix = tfidf_matrix.div(np.linalg.norm(tfidf_matrix, axis=1), axis=0)
print("\nTF-IDF Weights")
display(tfidf_matrix)

query = input("\nEnter your query: ").strip().lower()
query_tokens = [token for token in query.split() if token in terms]

if not query_tokens:
    print("\nNo valid terms from the query found in documents!")
else:
    query_tf = pd.Series({term: query_tokens.count(term) for term in terms})
    query_tfidf = query_tf * idf
    query_tfidf = query_tfidf / np.linalg.norm(query_tfidf)
    query_df = pd.DataFrame({"Query TF-IDF Weight": query_tfidf[query_tfidf > 0]})
    print("\nQuery TF-IDF Weights")
    display(query_df)

    def cosine_similarity(q, d):
        dot = np.dot(q, d)
        norm_q = np.linalg.norm(q)
        norm_d = np.linalg.norm(d)
        return dot, norm_q, norm_d, (dot / (norm_q * norm_d)) if norm_q and norm_d else 0.0

    results = []
    for doc_id, doc_vec in zip(tfidf_matrix.index, tfidf_matrix.values):
        dot, norm_q, norm_d, cos = cosine_similarity(query_tfidf.values, doc_vec)
        results.append([doc_id, documents[doc_id], dot, norm_q, norm_d, cos])

    cosine_df = pd.DataFrame(results, columns=["Doc ID", "Document", "Dot Product", "Query Magnitude", "Doc Magnitude", "Cosine Similarity"])
    print("\n--- Search Results and Cosine Similarity ---")
    display(cosine_df)

    ranked_df = cosine_df.sort_values(by="Cosine Similarity", ascending=False).reset_index(drop=True)
    ranked_df.index += 1
    ranked_df.index.name = "Rank"
    print("\nRanked Documents")
    display(ranked_df[["Doc ID", "Document", "Cosine Similarity"]])
    print(f"\nThe highest rank cosine score is: {ranked_df.iloc[0]['Cosine Similarity']:.3f} (Document ID: {ranked_df.iloc[0]['Doc ID']})")

```

### Output:
<img width="1360" height="1558" alt="image" src="https://github.com/user-attachments/assets/4126a14a-e15c-4c40-a600-df8dfd351a58" />
<img width="939" height="461" alt="Screenshot 2025-09-26 at 11 31 58 AM" src="https://github.com/user-attachments/assets/54232ddc-5cf0-45bd-b8bf-079e30b3fc59" />

### Result:
Thus Information Retrieval Using Vector Space Model in Python has been implemented successfully.
