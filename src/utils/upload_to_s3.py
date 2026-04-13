"""
Upload local datasets to AWS S3.
Run this once to upload your data to S3.

Prerequisites:
    pip install boto3
    aws configure  # Set up AWS credentials
"""

import boto3
from pathlib import Path

# Configuration
BUCKET_NAME = "your-bucket-name"  # Change this to your bucket name
AWS_REGION = "us-east-1"  # Change to your preferred region

def upload_directory_to_s3(local_path, s3_prefix):
    """Upload a directory to S3."""
    s3_client = boto3.client('s3')
    local_dir = Path(local_path)

    for file_path in local_dir.rglob('*'):
        if file_path.is_file():
            # Calculate S3 key (path in bucket)
            relative_path = file_path.relative_to(local_dir)
            s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')

            print(f"Uploading {file_path} to s3://{BUCKET_NAME}/{s3_key}")

            # Upload file
            s3_client.upload_file(
                str(file_path),
                BUCKET_NAME,
                s3_key,
                ExtraArgs={'ContentType': 'text/csv'}  # Adjust based on file type
            )

    print(f"✓ Uploaded {local_path} to S3")

def main():
    """Upload all datasets to S3."""
    print(f"Uploading datasets to S3 bucket: {BUCKET_NAME}\n")

    # Upload IMDB data
    upload_directory_to_s3("data/imdb", "datasets/imdb")

    # Upload Goodreads data
    upload_directory_to_s3("data/goodreads", "datasets/goodreads")

    print("\n✓ All datasets uploaded to S3!")
    print(f"\nS3 URLs:")
    print(f"- IMDB: s3://{BUCKET_NAME}/datasets/imdb/")
    print(f"- Goodreads: s3://{BUCKET_NAME}/datasets/goodreads/")

if __name__ == "__main__":
    main()
