from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

app = Flask(__name__)

# Load MiniLM model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load questions and answers
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
answers = data["answers"]

@app.route("/", methods=["GET", "POST"])
def home():

    response = ""

    if request.method == "POST":

        user_question = request.form["question"]

        # Convert question to embedding
        query_embedding = model.encode([user_question])

        query_embedding = np.array(query_embedding).astype('float32')

        # Similarity search
        k = 1
        distances, indices = index.search(query_embedding, k)

        matched_index = indices[0][0]

        response = answers[matched_index]

    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)