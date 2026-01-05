# ================================
# 1. IMPORTS
# ================================

import os

import pandas as pd

from tqdm import tqdm

from flask import Flask, request, jsonify, render_template_string

from transformers import pipeline

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_community.llms import HuggingFacePipeline

from langchain.chains import RetrievalQA

# ================================
# 2. CONFIGURATION
# ================================

IOC_FILE = "Compromised_IOCs_Cleaned.csv"

TWEET_FILE = "Tweetfeed_cleaned.csv"

VECTOR_DB_DIR = "./rag_db"

TOP_K = 3

# Disable tokenizer parallelism warnings for cleaner output
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ================================
# 3. LOAD DATA
# ================================

print("Loading datasets...")

# Load IOC tweets CSV file
df_iocs = pd.read_csv(IOC_FILE, encoding="latin1")

# Load general tweets CSV file
df_tweets = pd.read_csv(TWEET_FILE, encoding="latin1")

# Convert column names to lowercase for consistency
df_iocs.columns = df_iocs.columns.str.lower()
df_tweets.columns = df_tweets.columns.str.lower()

# Combine tweet text from both datasets into one list
# - concat merges the data
# - drop_duplicates removes repeated tweets
# - dropna removes empty tweets
# - tolist converts to Python list
texts = pd.concat(
    [df_iocs["tweet"], df_tweets["tweet"]],
    ignore_index=True
).drop_duplicates().dropna().tolist()

print(f"Loaded {len(texts)} unique tweets")


# ================================
# 4. EMBEDDINGS & VECTOR STORE
# ================================

print("Initializing embeddings...")

# Initialize sentence-transformer model for generating embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create or load the Chroma vector database
vectordb = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_function
)

# Check if the vector database already contains data
if vectordb._collection.count() == 0:
    print("Embedding tweets into ChromaDB...")

    # Create unique IDs for each tweet
    ids = [str(i) for i in range(len(texts))]

    # Define batch size to reduce memory usage
    batch_size = 100

    # Add tweets to vector database in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        vectordb.add_texts(
            texts=texts[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

    # Save vector database to disk
    vectordb.persist()
    print("Vector database created and saved.")

else:
    # If embeddings already exist, reuse them
    print("Existing vector database loaded.")


# ================================
# 5. RAG PIPELINE
# ================================

print("Loading language model...")

# Load Flan-T5 model for text-to-text generation
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

# Wrap Hugging Face pipeline as a LangChain LLM
llm = HuggingFacePipeline(pipeline=generator)

# Retriever fetches top-K similar tweets from vector database
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

# RetrievalQA chain:
# - Retrieves relevant documents
# - Passes them to the language model
# - Returns answer along with source documents
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

print("RAG pipeline ready.")


# ================================
# 6. FLASK APP
# ================================

# Initialize Flask application
app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """
    Root API endpoint.
    Used to check server status and available APIs.
    """
    return jsonify({
        "status": "running",
        "message": "RAG Cyber Threat Intelligence API",
        "endpoints": {
            "POST /chat": {
                "input": {"query": "string"},
                "output": {"answer": "string", "sources": "list"}
            },
            "GET /ui": "Browser-based interface"
        }
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat API endpoint.
    Accepts user query and returns RAG-based response.
    """

    # Get JSON data from request
    data = request.get_json()

    # Validate input
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    # Extract user query
    user_query = data["query"]

    # Prompt tuning to keep answers short and relevant
    prompt = (
        "Answer in 2–3 sentences based on the retrieved sources: "
        + user_query
    )

    # Run the Retrieval-Augmented Generation pipeline
    result = qa_chain({"query": prompt})

    # Return response as JSON
    return jsonify({
        "query": user_query,
        "answer": result["result"],
        "sources": [
            doc.page_content for doc in result["source_documents"]
        ]
    })


@app.route("/ui", methods=["GET"])
def ui():
    """
    UI endpoint.
    Serves a simple HTML frontend for user interaction.
    """

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Cyber Threat Intelligence</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            textarea { width: 100%; height: 80px; }
            button { padding: 10px; margin-top: 10px; }
            .answer { margin-top: 20px; padding: 10px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <h2>RAG Cyber Threat Intelligence Chat</h2>

        <textarea id="query" placeholder="Enter your question here..."></textarea><br>
        <button onclick="sendQuery()">Ask</button>

        <div class="answer" id="result"></div>

        <script>
            // Sends user query to backend chat API
            function sendQuery() {
                const query = document.getElementById("query").value;

                fetch("/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ query: query })
                })
                .then(response => response.json())
                .then(data => {
                    let output = "<b>Answer:</b> " + data.answer + "<br><br>";
                    output += "<b>Sources:</b><ul>";
                    data.sources.forEach(src => {
                        output += "<li>" + src + "</li>";
                    });
                    output += "</ul>";
                    document.getElementById("result").innerHTML = output;
                });
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


# ================================
# 7. RUN SERVER
# ================================

# Start Flask server when this file is executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
