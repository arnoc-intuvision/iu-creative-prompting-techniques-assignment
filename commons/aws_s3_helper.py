import boto3

class AWSS3Helper:

    def __init__(self, s3_bucket: str = "ppa-chatbot-knowledge-base", prefix: str = ""):
        self.s3_bucket = s3_bucket
        self.prefix = prefix
        self.s3_client = boto3.client('s3')

    def get_documents(self):

        print(f"\nRead documents from AWS S3 bucket: {self.s3_bucket}/{self.prefix}\n")

        # List objects in the bucket
        response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=self.prefix)

        for obj in response.get('Contents', []):

            key = obj['Key']

            # Download object content
            file_obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)

            content = file_obj['Body'].read().decode('utf-8')
            
            yield {'key': key, 'content': content}