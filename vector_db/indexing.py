import numpy as np
from tqdm import tqdm
import json
import torch
import pymilvus
from pymilvus import (
    MilvusClient, utility, connections,
    FieldSchema, CollectionSchema, DataType,
    Collection
)
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

class MilvusManager:
    def __init__(self, collection_name, embedding_model_name, host='localhost', port='19530'):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.embedding_model_name = embedding_model_name
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = self._initialize_embedding_model(self.embedding_model_name)
        self.collection = None
        self.connect_to_server()
        self.initialize_collection()

    def _initialize_embedding_model(self, model_name):
        print(f"Initializing embedding model on device: {self.device}")
        return BGEM3EmbeddingFunction(
            model_name=model_name,
            use_fp16=False, 
            device=self.device)

    def connect_to_server(self):
        print(f"Connecting to Milvus server at {self.host}:{self.port}")
        connections.connect(alias="default", host=self.host, port=self.port)

    def _create_index(self):
        print(f"Creating HNSW index for collection: {self.collection_name}")
        index_params = {'M': 16, 'efConstruction': 32}
        dense_index = {"index_type": "HNSW", "metric_type": "COSINE", "params": index_params}
        self.collection.create_index("dense_vector", dense_index)

    def initialize_collection(self):
        print(f"Initializing collection: {self.collection_name}")
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"Dropped existing collection: {self.collection_name}")
        
        embedding_dim = self.embedding_model.dim['dense']
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=60000),
            FieldSchema(name="law_id", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000)
        ]
        schema = CollectionSchema(fields)
        self.collection = Collection(self.collection_name, schema, consistency_level="Eventually")
        self._create_index()
        self.collection.load()

    def insert_data(self, data):
        print(f"Inserting data into collection: {self.collection_name}")
        dict_list = []
        for item in data:
            chunk_dict = {
                'chunk': item['chunk'],
                'law_id': item['law_id'],
                'title': item['title'],
                'dense_vector': item['dense_vector']
            }
            dict_list.append(chunk_dict)

        if dict_list:
            self.collection.insert(dict_list)
            self.collection.flush()
            print("Data successfully inserted and flushed to Milvus collection.")

class DataProcessor:
    def __init__(self, embedding_model, max_length=60000):
        self.embedding_model = embedding_model
        self.max_length = max_length

    def trim_string(self, input_string):
        return input_string[:self.max_length] if len(input_string) > self.max_length else input_string

    def process_documents(self, data):
        processed_data = []
        for doc in tqdm(data, desc="Processing documents"):
            law_id = doc["Field_id"]
            articles = doc["infor"]

            for article in articles:
                title = article["title"]
                text = title + ". " + article["text"] 
                trimmed_text = self.trim_string(text)

                if len(text) != 0:
                    embeddings = self.embedding_model([text])
                    processed_data.append({
                        'chunk': trimmed_text,
                        'law_id': law_id,
                        'title': title,
                        'dense_vector': embeddings["dense"][0]
                    })
        return processed_data

def load_data_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def main():
    # Initialize Milvus manager
    collection_name = "Hust_Law"
    embedding_model_name = "/point/namnt/md1/NAMNT_DA2/checkpoint/checkpoint_fine_tune2/checkpoint-1000"

    milvus_manager = MilvusManager(collection_name=collection_name, embedding_model_name= embedding_model_name)

    # Load and process data
    file_name = "/home/namnt/DATN/retrieval/data/template.json"
    data = load_data_from_file(file_name)
    
    processor = DataProcessor(embedding_model=milvus_manager.embedding_model)
    processed_data = processor.process_documents(data)  # Example range

    # Insert processed data into Milvus
    milvus_manager.insert_data(processed_data)

if __name__ == "__main__":
    main()
