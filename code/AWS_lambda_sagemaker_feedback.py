import json
import csv
import os
import boto3
import datetime


#환경 변수  
#instance_count = 1
#instance_type = ml.c4.xlarge
#training_job_name = linear-learner-2019-10-31-10-17-32-232
#training_job_prefix = linear-feedback

s3 = boto3.client('s3')
sm = boto3.client('sagemaker')

bucket_fb = 'sagemaker-feedback'
bucket_model = 'sagemaker-1026'

keys=[]
key_test = 'stratified_train.txt'


def check_count(bucket):
    data_list = s3.list_objects(Bucket=bucket)['Contents']
    print (len(data_list))
    return len(data_list)
    
    

def get_data(bucket):
    datas = ''
    data_list = s3.list_objects(Bucket=bucket)['Contents']
    for obj in data_list:
        keys.append(obj['Key'])
        data = s3.get_object(Bucket=bucket, Key=obj['Key'])
        data = data['Body'].read().decode('utf-8') 
        datas = datas + '\n' + data
    #print(datas)
    return datas
    
    
def add_to_train_data(data_fb, bucket):
    data_train = s3.get_object(Bucket=bucket, Key=key_test)
    data_train = data_train['Body'].read().decode('utf-8') 
    data_train = data_train + data_fb
    #print(data_train)

    return (data_train)
    
    
def delect_files(count, bucket):
    for i in range(count):
        s3.delete_object(Bucket= bucket,Key= keys[i])
    print ("delect feedback datas")
    
    
def update_train_data(data_added):
    s3.delete_object(Bucket= bucket_model,Key= key_test)
    upload_path = '/tmp/result_{}'.format(key_test)
    with open(upload_path, 'w', encoding = 'utf-8') as text:
        text.write(data_added)
    s3.upload_file(upload_path, bucket_model, key_test)
    print("update train data")
        
        
def re_train():
    training_job_name = os.environ['training_job_name']
    job = sm.describe_training_job(TrainingJobName=training_job_name)

    training_job_prefix = os.environ['training_job_prefix']
    training_job_name = training_job_prefix+str(datetime.datetime.today()).replace(' ', '-').replace(':', '-').rsplit('.')[0]
    job['ResourceConfig']['InstanceType'] = os.environ['instance_type']
    job['ResourceConfig']['InstanceCount'] = int(os.environ['instance_count'])
    
    AlgorithmSpecification = {}
    AlgorithmSpecification['TrainingImage'] = job['AlgorithmSpecification']['TrainingImage']
    AlgorithmSpecification['TrainingInputMode'] = job['AlgorithmSpecification']['TrainingInputMode']
    job['AlgorithmSpecification'] = AlgorithmSpecification
    #job['AlgorithmSpecification']= {'TrainingImage': '835164637446.dkr.ecr.ap-northeast-2.amazonaws.com/linear-learner:1', 'TrainingInputMode': 'File'}    
         
    #print(job['AlgorithmSpecification'])
    

    print("Starting training job %s" % training_job_name)
    
    resp = sm.create_training_job(
            TrainingJobName=training_job_name, 
            AlgorithmSpecification=job['AlgorithmSpecification'], 
            RoleArn=job['RoleArn'],
            InputDataConfig=job['InputDataConfig'], 
            OutputDataConfig=job['OutputDataConfig'],
            ResourceConfig=job['ResourceConfig'], 
            StoppingCondition=job['StoppingCondition'],
            HyperParameters=job['HyperParameters'])
    
    print (resp)
    return resp
    
    
    
    
    
        
def lambda_handler(event, context):
    count = check_count(bucket_fb)
    if (count < 100):
        print ('not enough data for re-train')
        return 0
        
    else:
        data_fb = get_data(bucket_fb)
        data_added = add_to_train_data(data_fb, bucket_model)
        #print("data_add", data_added)
        #print(keys)
        delect_files(count, bucket_fb)
        update_train_data(data_added)
        
        respose = re_train()
        return respose
    
    
    




