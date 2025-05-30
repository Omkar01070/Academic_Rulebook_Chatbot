import os
import PyPDF2  # PyPDF2 for PDF text extraction
import faiss  # Facebook AI Similarity Search ,, vectors mai store
import numpy as np
from sentence_transformers import SentenceTransformer  #
from transformers import AutoTokenizer # words to number
from fastapi import FastAPI
from pydantic import BaseModel #
from groq import Groq
import uvicorn

# Load Groq API key from environment variable (set this before running)
os.environ["GROQ_API_KEY"] = "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" # key to use api
client = Groq(api_key=os.environ["GROQ_API_KEY"])
print('starting')

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # model name

#Extract Text from PDF 
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text.strip() #

# Load the rulebook PDF and preprocess
pdf_text = extract_text_from_pdf("C:\\Users\\ASUS\\Downloads\\Academic Rules and Regulations-JAN 2025.pdf")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") #bert tokenizer for converting to numbers

def token_based_chunking(text, max_tokens=256):     # 256 tokens in one chunk, each token may contain more than one word or split words depends on freq
    """ Split text into chunks based on token count. """
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk) for chunk in chunks]

text_chunks = token_based_chunking(pdf_text, max_tokens=256)
print(f"Generated {len(text_chunks)} chunks.")

#Step 2 Generate Embeddings and Store in FAISS 
print("Generating embeddings and building FAISS index...")

# Convert text chunks to embeddings
embeddings = embedding_model.encode(text_chunks, show_progress_bar=True) # chunks to word embeddings, like smaller``

# Store embeddings in FAISS index
dimension = embeddings.shape[1]        # word embedding,,,,sentences matching with similarities, words with similar meaning must have similar representation
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("Indexing complete. Ready for queries.")

# Step 3: FastAPI Setup for Query Processing
app = FastAPI()

# Define input model
class QueryRequest(BaseModel):
    question: str                   #interface que

# Function to query FAISS and return the most relevant document chunks
def retrieve_relevant_texts(query, top_k=5):
    query_embedding = embedding_model.encode([query])             # query related chunk shodhat ahe
    distances, indices = index.search(np.array(query_embedding), top_k) # min. dist. shodhel,, top 3-5 related baher kadhel
    retrieved_texts = [text_chunks[i] for i in indices[0]]
    return " ".join(retrieved_texts)                            # tyanna combine karel

# Function to generate a smart answer using Groq API
def generate_smart_answer(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[ 
            {"role": "system", "content": "You are an AI assistant providing academic rule-based answers."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Define input model
#class QueryRequest(BaseModel):
#    question: str  # The field expected in the incoming request

# Define API endpoint
@app.post("/query/")
def query_rag_system(query_request: QueryRequest):
    # Retrieve relevant rulebook sections
    context = retrieve_relevant_texts(query_request.question)

    # Generate response using Groq API
    answer = generate_smart_answer(query_request.question, context)

    return {
        "query :": query_request.question,
        #"retrieved_context": context,
        "answer": answer
    }


if __name__ == "__main__":
    print("Starting RAG system server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)