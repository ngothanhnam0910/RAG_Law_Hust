from fastapi import FastAPI
from pydantic import BaseModel
import torch
from pymilvus import connections, Collection
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

app = FastAPI()

class QueryRequest(BaseModel):
    query_text: str
    top_k: int = 20  # Default value is 20

class QueryManager:
    def __init__(self, collection_name, embedding_model_name, host='localhost', port='19530'):
        self.collection_name = collection_name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.embedding_model_name = embedding_model_name
        self.embedding_model = self._initialize_embedding_model(self.embedding_model_name)
        self.host = host
        self.port = port
        self.connect_to_server()
        self.collection = self._get_collection()

    def _initialize_embedding_model(self, model_name):
        print(f"Initializing embedding model on device: {self.device}")
        return BGEM3EmbeddingFunction(
            model_name=model_name,
            use_fp16=False,
            device=self.device)

    def connect_to_server(self):
        print(f"Connecting to Milvus server at {self.collection_name}:{self.collection_name}")
        connections.connect(alias="default", host='localhost', port='19530')

    def _get_collection(self):
        collection = Collection(self.collection_name)
        collection.load()  # Load toàn bộ fields vào bộ nhớ
        return collection

    def get_embedding(self, query_text):
        print(f"Generating embedding for the query: {query_text}")
        return self.embedding_model([query_text])["dense"][0]

    def search(self, query_text, top_k=20):
        query_embedding = self.get_embedding(query_text)
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}  # Parameter for HNSW search, adjust as needed
        }

        print("Searching for the top nearest documents...")
        search_results = self.collection.search(
            data=[query_embedding],  # query_embedding must be a list
            anns_field="dense_vector",  # Field where embeddings are stored
            param=search_params,
            limit=top_k,
            output_fields=["law_id", "title", "chunk"],  # Chỉ định các field mà bạn muốn lấy ra
            expr=None,  # No filter expression, modify if needed
            consistency_level="Strong"
        )
        
        return search_results


# Initialize QueryManager for the collection 'Law_VN'
collection_name = "Hust_Law"
embedding_model_name = "/point/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_fine_tune2/checkpoint-1000"
query_manager = QueryManager(collection_name=collection_name, 
                                embedding_model_name=embedding_model_name)

# define retriver
@app.post("/retriever")
async def search_query(request: QueryRequest):
    # Execute the search
    search_results = query_manager.search(query_text=request.query_text, top_k=request.top_k)

    # Format the results into a list of dictionaries
    results = []
    for result in search_results[0]:  # search_results[0] chứa kết quả tìm kiếm
        results.append({
            "law_id": result.entity.get("law_id"),
            "title": result.entity.get("title"),
            "chunk": result.entity.get("chunk"),  # Show only the first 200 chars of the chunk
            "distance": result.distance
        })

    return {"results": results}

# To run FastAPI, use: uvicorn app:app --host 0.0.0.0 --port 5001 --reload
