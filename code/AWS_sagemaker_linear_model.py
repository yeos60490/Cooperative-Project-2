import boto3
import re
from sagemaker import get_execution_role
import io
import numpy as np
import sagemaker.amazon.common as smac
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer, json_deserializer
import sagemaker
import csv
import os



bucket = 'sagemaker-1026'
prefix = 'sagemaker/DEMO-linear-mnist'
key = 'stratified_train.txt'
 
role = get_execution_role()
s3 = boto3.client('s3')

train = []
data = s3.get_object(Bucket=bucket, Key=key)
data = data['Body'].read().decode('utf-8')
datas = data.split('\n')

for i in range(len(datas)):
    datas[i] = datas[i].split('\t')
    for j in range(len(datas[i])):
        datas[i][j] = int(datas[i][j])
    train.append(datas[i])
    

train= np.array(train, dtype='float32')
train = np.transpose(train)

print('train', train, train.shape)

#train= np.loadtxt('stratified_train.txt', unpack=True, dtype='float32')
#print('train', train, train.shape)

vectors = np.transpose(train[0:50]) #shape (1256, 50)
labels = np.transpose(train[50:]) #(1256, 1)
print ("labels", labels.shape)
labels=np.concatenate(labels)  #(1256, )


print(len(labels.shape))


buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)
buf.seek(0)

key = 'recordio-pb-data'
boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))



output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))


container = get_image_uri(boto3.Session().region_name, 'linear-learner')




sess = sagemaker.Session()

linear = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.c4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sess)
linear.set_hyperparameters(feature_dim=50,
                           predictor_type='multiclass_classifier',
                           num_classes=96,
                           mini_batch_size=200,
                           epochs=72,
                           accuracy_top_k=3)

linear.fit({'train': s3_train_data})



linear_predictor = linear.deploy(initial_instance_count=1,
                                 instance_type='ml.m4.xlarge')
