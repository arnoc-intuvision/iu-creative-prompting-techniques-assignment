import os 
import json
import uvicorn
from dotenv import load_dotenv
from typing import Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from commons.utils import HelperUtils
from commons.qdrant_helper import QdrantHelper
from commons.aws_secrets_manager_helper import AWSSecretManagerHelper
from commons.google_genai_helper import GoogleGenaiHelper

load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path="./cloud-credentials/aws_credentials.env")

secrets_helper = AWSSecretManagerHelper()
aws_secret_name = os.getenv("AWS_SECRET_NAME")
secret_result_json = secrets_helper.get_secret(secret_name=aws_secret_name)
google_cloud_service_credential = secret_result_json.get('GOOGLE_CLOUD_SERVICE_CREDENTIAL')
qdrant_api_key = secret_result_json.get('QDRANT_CLOUD_API_KEY')

google_cloud_project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
google_cloud_location = os.getenv('GOOGLE_CLOUD_LOCATION')
google_cloud_service_credential_json = json.loads(google_cloud_service_credential)

qdrant_url = os.getenv('QDRANT_URL')
qdrant_collection_name = os.getenv('QDRANT_PPA_COLLECTION_NAME')
qdrant_recreate_collection = os.getenv('QDRANT_RECREATE_COLLECTION')

helper_utils = HelperUtils()
genai_helper = GoogleGenaiHelper(
    project_id=google_cloud_project_id, 
    location=google_cloud_location, 
    credentials=google_cloud_service_credential_json
)
qdrant_helper = QdrantHelper(
    url=qdrant_url, 
    api_key=qdrant_api_key
)

# Request model
class QueryRequest(BaseModel):
    query: str
    top_k: int

# Response model (optional, but good practice)
class QueryResponse(BaseModel):
    results: Any
    message: str

# Initialize FastAPI app
app = FastAPI(title="PPA Knowledge Base API", version="1.0.0")

@app.post("/query_ppa_knowledge_base", response_model=QueryResponse)
async def query_ppa_knowledge_base(request: QueryRequest) -> QueryResponse:
    """
    Query the PPA knowledge base with the provided query and return top_k results.
    """
    try:
       
        user_query = request.query
        top_k = request.top_k

        print(f"\nReceived new user query:\n{user_query}\n")

        # Extract metadata details from the user's query
        metadata_details = genai_helper.extract_metadata_details_from_user_query_task(
            user_query=user_query
        )

        # Generate a vector embedding for the user's query
        vector_embeddings = genai_helper.generate_embeddings(
            model="gemini-embedding-001", 
            contents=[user_query],
            title=metadata_details['project_name']
        )

        assert len(vector_embeddings.embeddings) > 0, "No vector embeddings returned from Google Vertex AI."

        embeddings = list(map(helper_utils.get_embedding_values,  vector_embeddings.embeddings))
        _, embed_val = embeddings[0]

        # Query the Qdrant PPA knowledge base with the user's query
        document_results = qdrant_helper.query_vector_store(
            collection_name="ppa_knowledge_base", 
            query_vector=embed_val,
            top_k=top_k
        )

        # Format the retrieved document context
        retrieved_docs = "\n".join(
        [f"""
        Document Result Number: {idx + 1}
        Project Code: {doc.payload['metadata']['project_code']}
        Project Name: {doc.payload['metadata']['project_name']}
        Filename: {doc.payload['metadata']['filename']}

        Section Summary: {doc.payload['metadata']['summary']}

        Section Content: \n{doc.payload['content']}\n
        """ 
        for idx, doc in enumerate(document_results) ]
        ) 
        
        results = {
            "query": user_query,
            "top_k": top_k,
            "documents": retrieved_docs
        }
        
        return QueryResponse(
            results=results,
            message="Query processed successfully"
        )
    
    except Exception as e:
        error_message = e.with_traceback(e.__traceback__)
        raise HTTPException(status_code=500, detail=f"\nInternal server error:\n{error_message}\n")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PPA Knowledge Base API is running"}

if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8000)