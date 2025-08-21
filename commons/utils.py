import re
from google.genai.types import ContentEmbedding

class HelperUtils:

    def get_embedding_values(self, embedding_object: ContentEmbedding):
        return embedding_object.statistics, embedding_object.values
    
    def extract_banner_fields(self, content: str) -> dict:

        pattern = (
            r"<PAGE NUMBER: \d+, PROJECT CODE: (?P<project_code>\d+), "
            r"PROJECT NAME: (?P<project_name>[^,]+), DOCUMENT NAME: (?P<filename>[^>]+)>"
        )

        match = re.match(pattern, content)

        if match:

            return {
                "project_code": match.group("project_code").strip(),
                "project_name": match.group("project_name").strip(),
                "filename": match.group("filename").strip()
            }
        
        return {}