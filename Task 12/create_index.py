import pandas as pd
import numpy as np
import faiss
import pickle

from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data.csv")

questions = df['question'].tolist()
answers = df['answer'].tolist()

# Load MiniLM model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(questions)

embeddings = np.array(embeddings).astype('float32')

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_index.bin")

# Save answers
with open("embeddings.pkl", "wb") as f:
    pickle.dump({
        "questions": questions,
        "answers": answers
    }, f)

print("FAISS index created successfully!")