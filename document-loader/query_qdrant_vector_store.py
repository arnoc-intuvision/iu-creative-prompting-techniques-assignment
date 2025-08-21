import os 
import json
from dotenv import load_dotenv

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

if __name__ == "__main__":

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

    user_query = """ 
    I need your assistance to help me evaluate our customer's request to cancel their 20-year Power Purchase Agreement (PPA) after eight years of operation with SolarAfrica Energy.

    Please provide a detailed analysis with specific references to the relevant contractual clauses in the Procter and Gamble (P&G) PPA agreement.
    """
    # user_query = "What is the total installed capacity across all zones stipulated in the Ford PPA agreement ?"
    # user_query = "What is the performance warranty percentage for the Ezee Tile agreement ?"
    # user_query = "What is the performance warranty percentage for the Yanfeng East London agreement ?"
    print(f"\nUser query:\n{user_query}\n")

    metadata_details = genai_helper.extract_metadata_details_from_user_query_task(
        user_query=user_query
    )
    print(f"Extracted metadata details from user's query: {metadata_details}")

    vector_embeddings = genai_helper.generate_embeddings(
        model="gemini-embedding-001", 
        contents=[user_query],
        title=metadata_details['project_name']
    )

    assert len(vector_embeddings.embeddings) > 0, "No vector embeddings returned from Google Vertex AI."

    embeddings = list(map(helper_utils.get_embedding_values,  vector_embeddings.embeddings))
    _, embed_val = embeddings[0]

    document_results = qdrant_helper.query_vector_store(
        collection_name=qdrant_collection_name, 
        query_vector=embed_val,
        top_k=5
    )

    print("\nDocument Results: ")

    retrieved_docs = "\n".join(
    [f"""
    Document Details:
    Project Code: {doc.payload['metadata']['project_code']}
    Project Name: {doc.payload['metadata']['project_name']}
    PPA Filename: {doc.payload['metadata']['filename']}
    PPA Section Summary: {doc.payload['metadata']['summary']}
    PPA Section Content: \n{doc.payload['content']}\n
    """ 
    for doc in document_results]
    ) 

    print(retrieved_docs)

    print("\nQuery LLM Assistant => ")

    llm_response = genai_helper.ppa_query_task(
        user_query=user_query,
        retrieved_documents=retrieved_docs
    )

    print(f"\nResponse:\n{llm_response}\n")