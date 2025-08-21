from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, MatchValue

class QdrantHelper:

    def __init__(self, url:str, api_key:str):
        self.client = QdrantClient(url=url, api_key=api_key)

    def query_vector_store(self, collection_name:str, query_vector: list, top_k:int = 5):

        print(f"\nQuerying Qdrant collection '{collection_name}' with top_k={top_k} ...")

        documents = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        print(f"\nQdrant returned {len(documents)} document results.\n")

        return documents
    
    def query_vector_store_with_filter(self, collection_name:str, query_vector: list, project_code_filter:str = "", project_name_filter:str = "", top_k:int = 5):

        documents = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(
                should=[
                    FieldCondition(
                        key='metadata.project_code',
                        match=MatchValue(value=project_code_filter)
                    ),
                    FieldCondition(
                        key='metadata.project_name',
                        match=MatchValue(value=project_name_filter)
                    )
                ]
            ),
            limit=top_k
        )

        return documents

    def point_object(self, index:int, embedding_vector:list, text_chunk:str, metadata:dict) -> models.PointStruct:

        point = models.PointStruct(
            id=index,
            vector=embedding_vector,
            payload={
                "content": text_chunk,
                "metadata": {
                    "project_code": metadata["project_code"],
                    "project_name": metadata["project_name"],
                    "filename": metadata["filename"],
                    "summary": metadata["summary"]
                }
            }
        )

        return point
    
    def create_collection(self, collection_name:str, vector_size:int = 1536, force_recreation: bool = False):

        if (force_recreation) or (not self.client.collection_exists(collection_name)):

            print(f"\nCreating Qdrant collection '{collection_name}' ... ")

            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )

            print(f"\nCollection '{collection_name}' created.\n")

        else:

            print(f"\nQdrant collection '{collection_name}' already exists. Skipping creation.\n")

    def create_keyword_filter_index(self, collection_name:str, field_name:str, field_schema:str = "keyword"):

        print(f"\nCreating keyword filter index on field '{field_name}' ... ")

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema
        )

        print("\nKeyword filter index created.\n")

    def ingest_data(self, collection_name:str, points:list):

        print(f"Uploading {len(points)} points into Qdrant collection '{collection_name}' ... ")

        operation_info =  self.client.upsert(
            collection_name=collection_name,
            points=points
        )

        print("\nUpload complete: ")
        print(operation_info)
        print("\n")

        return operation_info