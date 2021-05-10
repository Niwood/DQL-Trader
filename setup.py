import sh
import boto3
from tqdm import tqdm



'''
MAKE DIRS
'''
DATA_DIR = 'data/'
try:
    sh.mkdir(DATA_DIR)
    sh.mkdir(DATA_DIR + 'staged')
    sh.mkdir(DATA_DIR + 'stock')
except Exception as e:
    print(e)
    # quit()


try:
    sh.mkdir('models')
    sh.mkdir('pre_trained_models')
except Exception as e:
    print(e)


'''
LIST AVAILABLE FILES IN S3 BUCKET
'''
BUCKET = 'demo-store-123'
s3_rsc = boto3.resource('s3')
bucket_stocks = s3_rsc.Bucket(BUCKET).objects.filter(Prefix='stock/').all()
bucket_staged = s3_rsc.Bucket(BUCKET).objects.filter(Prefix='staged/').all()

all_stock_files = list()
for i in bucket_stocks:
    all_stock_files.append(i.key.split('/')[1])
    
all_staged_files = list()
for i in bucket_staged:
    all_staged_files.append(i.key.split('/')[1])



'''
DOWNLOAD THE STOCK FILES
'''
s3 = boto3.client('s3')
counter = 0
for _file in tqdm(all_stock_files):
    s3.download_file(
        BUCKET,
        'stock/' + _file,
        DATA_DIR + 'stock/' + _file)

    counter += 1
    if counter == 5:
        break


'''
DOWNLOAD THE STAGED FILES
'''
for _file in tqdm(all_staged_files):
    s3.download_file(
        BUCKET,
        'staged/' + _file,
        DATA_DIR + 'staged/' + _file)