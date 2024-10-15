from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pymilvus.model.reranker import BGERerankFunction
import torch

app = FastAPI()

class SearchResult(BaseModel):
    law_id: str
    title: str
    chunk: str
    distance: float

class RerankRequest(BaseModel):
    query_text: str
    search_results: List[SearchResult]  # Danh sách kết quả từ retriever

class Reranker:
    def __init__(self, reranker_model_name):
        self.reranker_model_name = reranker_model_name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.reranker_model = self._initialize_reranker_model(self.reranker_model_name)

    def _initialize_reranker_model(self, model_name):
        print(f"Initializing reranker model on device: {self.device}")
        return BGERerankFunction(
            model_name=model_name,
            use_fp16=False,
            device="cuda:0")

    def rerank(self, query_text: str, search_results: List[SearchResult], top_k_rerank=10):
        print("Reranking search results...")

        # Lấy các văn bản từ các kết quả tìm kiếm
        texts = [result.chunk for result in search_results]

        # Gọi reranker model và lấy kết quả
        reranked_results = self.reranker_model(query_text, texts)

        # Sắp xếp lại các kết quả dựa trên điểm rerank
        sorted_results = sorted(reranked_results, key=lambda x: x.score, reverse=True)

        # Trả về top_k_rerank kết quả sau khi rerank
        return sorted_results[:top_k_rerank]


# Khởi tạo đối tượng reranker
reranker_model_name = "/point/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_reranking/checkpoint-5000_old"
reranker = Reranker(reranker_model_name= reranker_model_name)

@app.post("/rerank")
async def rerank_query(request: RerankRequest):
    # Gọi hàm rerank để sắp xếp lại kết quả
    sorted_results = reranker.rerank(
        query_text=request.query_text, 
        search_results=request.search_results,
        top_k_rerank=5  # Bạn có thể thay đổi giá trị này nếu cần
    )

    results = []
    for i, result in enumerate(sorted_results, start=1):
        results.append({
            "chunk":result.text,
        })

    return {"results": results}


