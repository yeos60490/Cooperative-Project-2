import json
import csv
import boto3
import os
import sys
import uuid
import numpy as np
from urllib.parse import unquote_plus


#환경변수
#ENDPOINT_NAME = linear-learner-2019-10-31-10-17-32-232


s3 = boto3.client('s3')
bucket = 'sagemaker-1026'
classes_name=[] 
test_datas = []


ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')


#class 이름 가져오기
def get_classes(key):
    classes = s3.get_object(Bucket=bucket, Key=key)
    classes = classes['Body'].read().decode('utf-8') 
    classes_name = classes.split('\n')
    return classes_name
    
    
    
def get_data(key):
    data = s3.get_object(Bucket=bucket, Key=key)
    data = data['Body'].read().decode('utf-8') 
    data = data.replace('\t', ', ')
    test_datas = data.split('\n')
    #print ("test_datas:", test_datas)
    return (test_datas)
    

def lambda_handler(event, context):
    class_key = 'class_name.txt'
    data_key = 'testonly.csv'
    inf = event
    res=str(inf['info'])
    #print(res)
    test_datas=[]
    test_datas.append(res)
    classes_name = get_classes(class_key)
    #test_datas = get_data(data_key)
    #print(test_datas)
    predicted_countries = []
    
    
    
    #test data의 개수만큼 
    for i in range(len(test_datas)):
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='text/csv', Body=test_datas[i])
        result = json.loads(response['Body'].read().decode())
        #print("result", result)
        pred = result['predictions'][0]['score']
        idx = np.argsort(pred)[-3:]
        result={"first":str(idx[2]),
        "second":str(idx[1]),
        "third":str(idx[0])}
        
        
    return result
    
    
    
