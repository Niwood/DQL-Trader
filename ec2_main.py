import boto3

s3 = boto3.resource('s3')
bucket = s3.Bucket('demo-store-123')


for obj in bucket.objects.all():
    key = obj.key
    print(key)
    # body = obj.get()['Body'].read()