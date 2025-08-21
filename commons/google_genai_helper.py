import json
from pprint import pprint
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.types import EmbedContentConfig
from google.oauth2.service_account import Credentials

class MetadataDetailsSchema(BaseModel):
    project_code: int 
    project_name: str

class GoogleGenaiHelper:

    def __init__(self, project_id:str, location:str, credentials:dict):
        self.project_id = project_id
        self.location = location
        self.credentials = Credentials.from_service_account_info(
            credentials, 
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
            http_options=types.HttpOptions(api_version='v1')
        )

    def generate_embeddings(self, model:str, contents:list, title:str, output_dimensionality:int = 1536):

        print("\nGenerating vector embeddings ...")

        response = self.client.models.embed_content(
            model=model,
            contents=contents,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                title=title,
                output_dimensionality=output_dimensionality
            ),
        )

        print("\nVector embeddings generated.\n")

        return response
    
    def gemini_llm_chat_with_text_response(self, model:str, prompt:dict, max_output_tokens:int = 4096, temperature:float = 0.9):

        system_instructions = prompt["system"]
        user_query = prompt["user"]

        response = self.client.models.generate_content(
            model=model, 
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                max_output_tokens=max_output_tokens,
                temperature=temperature
            ),
        )

        return response.text
    
    def gemini_llm_chat_with_json_response(self, model:str, prompt:dict, response_schema, max_output_tokens:int = 1024, temperature:float = 0.3):

        system_instructions = prompt["system"]
        user_query = prompt["user"]

        response = self.client.models.generate_content(
            model=model, 
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                response_mime_type='application/json',
                response_schema=response_schema,
            ),
        )

        json_response = {
            "project_code": -1,
            "project_name": ""
        }

        try:

            llm_response = response.text
            # print(f"\nLLM's structured output response: \n{llm_response}\n")

            json_response = json.loads(llm_response)

        except Exception as ex:

            error_message = ex.with_traceback(ex.__traceback__)
            print(f"\nError: \n{error_message}\n")

        return json_response
    
    def summarize_content_task(self, content:str):

        summarize_prompt = {
            "system": "You're an expert in summarizing text to capture the core meaning of the content.",
            "user": f"""
            Summarize the below content in less than 30 words:
            {content} 
            """
        }

        llm_text_response = self.gemini_llm_chat_with_text_response(
            model="gemini-2.5-flash-lite", 
            prompt=summarize_prompt,
            max_output_tokens=250,
            temperature=0.5
        )

        return llm_text_response
    
    def extract_metadata_details_from_user_query_task(self, user_query:str):

        print("\nExtracting metadata details from the user's query ...")

        extraction_prompt = {
            "system": "You're a useful AI assistant and your only function is to extract the project name from the user's query.",
            "user": f"""
            Example 1:
            User's query: What is the performance warranty performance percentage for the Brits Industries PPA agreement ?
            Response: {{project_code: -1, project_name: \"Brits Industries\"}}

            Example 2:
            User's query: Explain the force majeure clause in the Outdoor Warehouse (9089) agreement.
            Response: {{project_code: 9089, project_name: \"Outdoor Warehouse\"}}

            Example 3:
            User's query: What is the contractual duration stipulated in the Tiger Brands PPA agreement ?
            Response: {{project_code: -1, project_name: \"Tiger Brands\"}}

            Example 4:
            User's query: Who is the customer for the PPA agreement with project code 5175 ?
            Response: {{project_code: 5175, project_name: null}}

            User's Query: {user_query}
            Response:
            """
        }

        llm_json_response = self.gemini_llm_chat_with_json_response(
            model="gemini-2.5-flash-lite", 
            prompt=extraction_prompt,
            response_schema=MetadataDetailsSchema,
            max_output_tokens=250,
            temperature=0.3
        )

        print("\nMetadata returned: ")
        pprint(llm_json_response)
        print("\n")

        return llm_json_response

    def ppa_query_task(self, user_query:str, retrieved_documents:list):

        ppa_query_prompt = {
            "system": "You're a useful AI assistant and need to answer the user's query from the provided context.",
            "user": f"""
            Context:
            {retrieved_documents} 

            User's Query:
            {user_query}
            """
        }

        llm_text_response = self.gemini_llm_chat_with_text_response(
            model="gemini-2.5-flash", 
            prompt=ppa_query_prompt,
            max_output_tokens=4096,
            temperature=0.9
        )

        return llm_text_response