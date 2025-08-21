import json
import boto3
from botocore.exceptions import ClientError

class AWSSecretManagerHelper:

    def __init__(self, region_name:str = "eu-central-1"):
        self.region_name = region_name

    def get_secret(self, secret_name: str):

        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=self.region_name
        )

        try:

            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )

        except ClientError as e:
            raise e

        secret_result_json = json.loads(get_secret_value_response['SecretString'])

        return secret_result_json
