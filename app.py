import os
import fitz  # PyMuPDF
import openai
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

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

client = OpenAI(api_key=openai.api_key)  # Replace with your actual key

def query_gpt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an assistant answering questions from a book. Use the context below to answer the question concisely.

Context:
{context}

Question: {question}
Answer:
    """.strip()

    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
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
    port = int(os.environ.get("PORT", 10000))
    print(f"Listening on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

