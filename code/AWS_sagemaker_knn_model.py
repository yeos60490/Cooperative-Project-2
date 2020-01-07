import tensorflow as tf
import numpy as np
import os
import csv
import io
import sagemaker.amazon.common as smac
import boto3
import sagemaker
import matplotlib.pyplot as plt
from sagemaker import get_execution_role
from sagemaker.predictor import csv_serializer, json_deserializer
from sagemaker.amazon.amazon_estimator import get_image_uri



learning_rate = 0.001
train= np.loadtxt('stratified_train1.txt', unpack=True, dtype='float32')
test = np.loadtxt('stratified_test1.txt', unpack=True, dtype='float32')
train = np.transpose(train)
test = np.transpose(test)

classes = []
file = open( 'class_name.txt', 'r') 
data = csv.reader(file)
for line in data:
    for one in line:
        classes.append(one)

#np.random.seed(0)
#np.random.shuffle()
#train_size = int(0.9 * train.shape[0])
#print(train_size)

x_data = train[ : , :-1]
y_data = train[ : , -1]

test_data = test[ : , :-1]
test_data_output = test[ : , -1]

print (x_data.shape)
print (y_data.shape)
print (test_data.shape)
print (test_data_output.shape)



print('train_features shape = ', x_data.shape)
print('train_labels shape = ', y_data.shape)

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, x_data, y_data)
buf.seek(0)

bucket = 'sagemaker-pretest'
prefix = 'knn'
key = 'data'

boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))

output_location = 's3://{}/linear-test/output'.format(bucket)


print('train_features shape = ', test_data.shape)
print('train_labels shape = ', test_data_output.shape)

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, test_data, test_data_output)
buf.seek(0)

boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'test', key)).upload_fileobj(buf)
s3_test_data = 's3://{}/{}/test/{}'.format(bucket, prefix, key)
print('uploaded test data location: {}'.format(s3_test_data))


def trained_estimator_from_hyperparams(s3_train_data, hyperparams, output_path, s3_test_data = None):
    containers = {
        'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/knn:1',
        'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/knn:1',
        'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/knn:1',
        'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/knn:1',
        'ap-northeast-1': '351501993468.dkr.ecr.ap-northeast-1.amazonaws.com/knn:1',
        'ap-northeast-2': '835164637446.dkr.ecr.ap-northeast-2.amazonaws.com/knn:1',
        'ap-southeast-2': '712309505854.dkr.ecr.ap-southeast-2.amazonaws.com/knn:1'
    }
    
    knn = sagemaker.estimator.Estimator(get_image_uri(boto3.Session().region_name, "knn"),
        get_execution_role(),
        train_instance_count=1,
        train_instance_type='ml.m5.2xlarge',
        output_path=output_path,
        sagemaker_session=sagemaker.Session())
    knn.set_hyperparameters(**hyperparams)
    
    # train a model. fit_input contains the locations of the train and test data
    fit_input = {'train': s3_train_data}
    if s3_test_data is not None:
        fit_input['test'] = s3_test_data
    #fit_input = {'test': s3_test_data}
    knn.fit(fit_input)
    return knn

hyperparams = {
    'feature_dim': 50,
    'k': 10,
    'sample_size': 1256,
    'predictor_type': 'classifier' 
}
output_path = output_location
knn_estimator = trained_estimator_from_hyperparams(s3_train_data, hyperparams, output_path, s3_test_data=s3_test_data)

