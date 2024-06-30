from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant


client = QdrantClient(url="http://localhost:6333")


client.recreate_collection(
    collection_name="recipe_collection_1",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    timeout=30
)

def get_qdrant_retriever():    
    
    qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=os.getenv("QDRANT_URL"),
    collection_name="recipe_collection_1",
)

    qdrant.similarity_search_with_score(query=query, k=1)








query = "Mac and Cheese recipe"
found_docs = ...