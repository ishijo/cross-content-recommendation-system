# AWS S3 Setup Guide

This guide shows you how to host your datasets on AWS S3 and access them directly from your Streamlit app.

## Benefits of Using S3

✅ No local storage needed - data stays in the cloud
✅ Fast access from anywhere
✅ Version control for datasets
✅ Can make data public or keep it private
✅ Pay only for what you use (~$0.023 per GB/month)

## Setup Steps

### 1. Create AWS Account

1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Follow the signup process (requires credit card)

### 2. Create S3 Bucket

1. Go to [AWS S3 Console](https://s3.console.aws.amazon.com)
2. Click "Create bucket"
3. Configure:
   - **Bucket name**: `your-project-datasets` (must be globally unique)
   - **Region**: Choose closest to you (e.g., `us-east-1`)
   - **Public access**: Choose based on option below

### 3. Choose Access Method

#### Option A: Public Bucket (No Authentication)

**Pros**: Simple, anyone can access
**Cons**: Data is publicly visible

1. Uncheck "Block all public access"
2. Add this bucket policy (replace `YOUR-BUCKET-NAME`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::YOUR-BUCKET-NAME/*"
    }
  ]
}
```

3. Update `s3_data_loader.py` to use `load_from_public_s3()`

#### Option B: Private Bucket (Authenticated)

**Pros**: Secure, only authorized users can access
**Cons**: Requires AWS credentials

1. Keep "Block all public access" checked
2. Set up AWS credentials locally:

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter:
#   AWS Access Key ID: [from AWS IAM]
#   AWS Secret Access Key: [from AWS IAM]
#   Default region: us-east-1
#   Default output format: json
```

3. Create IAM user with S3 read access:
   - Go to [IAM Console](https://console.aws.amazon.com/iam)
   - Create user with "AmazonS3ReadOnlyAccess" policy
   - Save Access Key ID and Secret Key

### 4. Upload Data to S3

**Option A: AWS Console (Manual)**

1. Go to your S3 bucket
2. Click "Upload"
3. Drag and drop your `data/imdb/` and `data/goodreads/` folders
4. Click "Upload"

**Option B: Using Script (Recommended)**

```bash
# Install boto3
pip install boto3

# Update bucket name in upload_to_s3.py
# Then run:
python upload_to_s3.py
```

### 5. Update Configuration

Edit `s3_data_loader.py`:

```python
BUCKET_NAME = "your-project-datasets"  # Your bucket name
AWS_REGION = "us-east-1"  # Your region
```

### 6. Install Dependencies

```bash
pip install boto3 s3fs
```

Or add to `requirements.txt`:
```
boto3
s3fs
```

### 7. Run Your App

**Using S3 version:**
```bash
cp main_s3.py main.py  # Replace main.py with S3 version
streamlit run main.py
```

**Or keep both:**
```bash
streamlit run main_s3.py  # S3 version
streamlit run main.py     # Local version
```

## Cost Estimate

For your datasets (~5GB):

- **Storage**: 5GB × $0.023/GB = ~$0.12/month
- **Data transfer**: First 100GB/month free
- **Requests**: GET requests ~$0.0004 per 1000 requests

**Total**: ~$0.15-0.50/month depending on usage

## Troubleshooting

### Error: "NoCredentialsError"
- Run `aws configure` and enter your credentials
- Or make bucket public and use `load_from_public_s3()`

### Error: "Access Denied"
- Check bucket policy for public buckets
- Check IAM user permissions for private buckets
- Verify bucket name and region are correct

### Slow loading
- Use `@st.cache_data` (already implemented)
- Consider using S3 in same region as your app
- Use `s3fs` for very large files

## Security Best Practices

1. **Don't commit AWS credentials** - Use `aws configure` or environment variables
2. **Use IAM roles** for EC2/production deployments
3. **Enable S3 versioning** to protect against accidental deletion
4. **Set up CloudWatch** to monitor costs
5. **Use lifecycle policies** to archive old data

## Next Steps

- Set up CloudFront CDN for faster global access
- Enable S3 versioning for data backup
- Use Athena to query data directly in S3
- Set up automatic Kaggle → S3 sync
