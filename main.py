from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# -------------------------
# 1️⃣ Connect to local ES
# -------------------------
ES_URL = "http://localhost:9200"
INDEX_NAME = "notes_ai_playground"

es = Elasticsearch(ES_URL)

try:
    info = es.info()
    print("✅ Connected! Cluster info:")
    print(info)
except Exception as e:
    print("❌ Connection failed:", e)
    exit()

# -------------------------
# 2️⃣ Delete and recreate index with dense_vector
# -------------------------
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)

mapping = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "content": {"type": "text"},
            "tags": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": 384}  # MiniLM vector size
        }
    }
}

es.indices.create(index=INDEX_NAME, body=mapping)
print("✅ Index created with dense_vector field")

# -------------------------
# 3️⃣ Generate notes
# -------------------------
notes = []
topics = ["AI study", "Workout", "Grocery", "Recipes", "Travel", "Books"]

for i in range(30):
    topic = topics[i % len(topics)]
    content = f"This is a note about {topic} number {i+1}. It contains important tips and details."
    notes.append({
        "title": f"{topic} note {i+1}",
        "content": content,
        "tags": [topic]
    })

# -------------------------
# 4️⃣ Generate embeddings
# -------------------------
print("🔹 Generating embeddings for notes...")
model = SentenceTransformer('all-MiniLM-L6-v2')

for note in notes:
    note['embedding'] = model.encode(note['content']).tolist()

# -------------------------
# 5️⃣ Index all notes
# -------------------------
for note in notes:
    es.index(index=INDEX_NAME, document=note)

es.indices.refresh(index=INDEX_NAME)
print("✅ All notes indexed with embeddings")

# -------------------------
# 6️⃣ BM25 keyword search example
# -------------------------
keyword_query = {
    "query": {
        "match": {
            "content": "AI study"
        }
    }
}

res = es.search(index=INDEX_NAME, body=keyword_query)
print("\n🔹 BM25 Search results for 'AI study':")
for hit in res["hits"]["hits"]:
    print(hit["_source"]["title"])

# -------------------------
# 7️⃣ Semantic vector search via script_score
# -------------------------
query_text = "I want to learn about AI and Elasticsearch"
query_vector = model.encode(query_text).tolist()

vector_query = {
    "size": 3,
    "query": {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
}

res = es.search(index=INDEX_NAME, body=vector_query)
print("\n🔹 Semantic Search results for:", query_text)
for hit in res["hits"]["hits"]:
    print(hit["_source"]["title"])


# -------------------------
# 8️⃣ Hybrid search (BM25 + vector similarity)
# -------------------------
query_text = "I want to learn about AI and Elasticsearch"
query_vector = model.encode(query_text).tolist()

hybrid_query = {
    "size": 3,
    "query": {
        "script_score": {
            "query": {
                "match": {
                    "content": query_text  # BM25 component
                }
            },
            "script": {
                # Combine BM25 (_score) with cosine similarity
                "source": "_score + (cosineSimilarity(params.query_vector, 'embedding') + 1.0)",
                "params": {"query_vector": query_vector}
            }
        }
    }
}

res = es.search(index=INDEX_NAME, body=hybrid_query)
print("\n🔹 Hybrid Search results for:", query_text)
for hit in res["hits"]["hits"]:
    print(f"{hit['_source']['title']} (score: {hit['_score']:.4f})")
