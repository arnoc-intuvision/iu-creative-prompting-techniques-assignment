import os 
import re
import json
from dotenv import load_dotenv

from commons.utils import HelperUtils
from commons.aws_s3_helper import AWSS3Helper
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
qdrant_cluster_name = os.getenv('QDRANT_CLUSTER_NAME')
qdrant_collection_name = os.getenv('QDRANT_PPA_COLLECTION_NAME')
qdrant_recreate_collection = os.getenv('QDRANT_RECREATE_COLLECTION')

if __name__ == "__main__":

    helper_utils = HelperUtils()

    s3_helper = AWSS3Helper()

    genai_helper = GoogleGenaiHelper(
        project_id=google_cloud_project_id, 
        location=google_cloud_location, 
        credentials=google_cloud_service_credential_json
    )

    qdrant_helper = QdrantHelper(
        url=qdrant_url, 
        api_key=qdrant_api_key
    )

    print("\nCreate Qdrant Collection: ")
    qdrant_helper.create_collection(
        collection_name=qdrant_collection_name, 
        force_recreation=qdrant_recreate_collection
    )

    print("\nDocument Loader Started => \n")
    qdrant_point_index = 0

    for doc in s3_helper.get_documents():

        doc_filename = doc['key']
        doc_content = doc['content']

        # Regex pattern to match the banner line
        banner_pattern = r"<PAGE NUMBER: \d+, PROJECT CODE: \d+, PROJECT NAME: [^,]+, DOCUMENT NAME: [^>]+>"

        # Split the document at each banner
        contents = re.split(f'({banner_pattern})', doc_content)

        # Remove any empty strings from the result
        text_chunks = [chunk.strip() for chunk in contents if chunk.strip()]

        print("\nChunk document text: ")

        qdrant_points = []

        for idx in range( int( len(text_chunks) / 2) ):

            page_banner_idx = idx * 2
            text_chunk_idx = page_banner_idx + 1 

            page_banner = text_chunks[page_banner_idx]
            text_chunk = text_chunks[text_chunk_idx]
            metadata_fields = {}

            # print(f"\n[{page_banner_idx}] Page Banner =>")
            # print(page_banner)

            print(f"\n[{page_banner_idx}] Extract Banner Fields. ")
            metadata_fields = helper_utils.extract_banner_fields(content=page_banner)

            print(f"\n[{text_chunk_idx}] Text Chunk: ")
            print(text_chunk)
            
            print(f"\n[{text_chunk_idx}] Generate Text Summary. ")
            text_summary = genai_helper.summarize_content_task(content=text_chunk)
            # print(text_summary)
            metadata_fields["summary"] = text_summary

            print(f"\n[{text_chunk_idx}] Metadata Fields: ")
            print(metadata_fields)

            embedding_result = genai_helper.generate_embeddings(
                model="gemini-embedding-001",
                contents=[text_chunk],
                title=metadata_fields['project_name']
            )
        
            embeddings = list(map(helper_utils.get_embedding_values, embedding_result.embeddings))

            if len(embeddings) > 0:

                print(f"\n[{text_chunk_idx}] Vector Embedding: ")
                embed_stats, embed_val = embeddings[0]
                print(embed_stats)
                # print(embed_val)

                point = qdrant_helper.point_object(
                    index=qdrant_point_index,
                    embedding_vector=embed_val,
                    text_chunk=text_chunk,
                    metadata=metadata_fields
                )

                qdrant_points.append(point)

                print(f"\nQdrant point added to the points list => Index: {qdrant_point_index}.\n")

                qdrant_point_index += 1

            # if qdrant_point_index >= 1:
            #     break

        qdrant_helper.ingest_data(
            collection_name=qdrant_collection_name,
            points=qdrant_points
        )

        # break

    print("\nCreate Filter Indexes => \n")

    qdrant_helper.create_keyword_filter_index(
        collection_name=qdrant_collection_name, 
        field_name="metadata.project_code"
    )

    qdrant_helper.create_keyword_filter_index(
        collection_name=qdrant_collection_name, 
        field_name="metadata.project_name"
    )