import sh
import boto3
from tqdm import tqdm



'''
MAKE DIRS
'''
DATA_DIR = 'TEST_DIR/'
try:
    sh.mkdir(DATA_DIR)
    sh.mkdir(DATA_DIR + 'staged')
    sh.mkdir(DATA_DIR + 'stock')
except Exception as e:
    print(e)
    quit()


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
bucket = s3_rsc.Bucket(BUCKET).objects.filter(Prefix='stock/').all()

all_files = list()
for i in bucket:
    all_files.append(i.key.split('/')[1])



'''
DOWNLOAD THE FILES FROM BUCKET
'''
s3 = boto3.client('s3')
counter = 0
for _file in tqdm(all_files):
    s3.download_file(
        BUCKET,
        'stock/' + _file,
        DATA_DIR + 'stock/' + _file)

    counter += 1
    if counter == 5:
        break