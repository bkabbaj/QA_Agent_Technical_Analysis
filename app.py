import os
import fitz  # PyMuPDF
import openai
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === Config ===
openai.api_key = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL = "all-MiniLM-L6-v2"

# === Load PDF and Chunk Text ===
def extract_text_chunks(pdf_path, chunk_size=500, overlap=50):
    doc = fitz.open(pdf_path)
    full_text = " ".join([page.get_text() for page in doc])
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# === Embed and Index Chunks ===
class VectorStore:
    def __init__(self, chunks):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.embeddings = self.model.encode(chunks)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        self.chunks = chunks

    def search(self, query, top_k=3):
        q_emb = self.model.encode([query])
        D, I = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in I[0]]

# === Query LLM ===
def query_gpt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an assistant answering questions from a book. Use the context below to answer the question concisely.

Context:
{context}

Question: {question}
Answer:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# === Flask App ===
app = Flask(__name__)
chunks = extract_text_chunks("book.pdf")
store = VectorStore(chunks)

@app.route("/whatsapp", methods=["POST"])
def whatsapp():
    incoming_msg = request.values.get("Body", "").strip()
    relevant_chunks = store.search(incoming_msg)
    answer = query_gpt(relevant_chunks, incoming_msg)
    resp = MessagingResponse()
    resp.message(answer)
    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)
