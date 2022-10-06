import boto3
import datetime
import os


def lambda_handler(event, context):    
    original_training_job_name = "sms-spam-classifier-mxnet-2022-04-18-15-25-57-981"
    sm = boto3.client('sagemaker')
    job = sm.describe_training_job(TrainingJobName=original_training_job_name)

    training_job_prefix = "sms-spam-classifier-mxnet-"
    training_job_name = training_job_prefix + str(datetime.datetime.today()).replace(' ', '-').replace(':', '-').rsplit('.')[0]
    job['ResourceConfig']['InstanceType'] = "ml.m4.xlarge"
    job['ResourceConfig']['InstanceCount'] = 1

    print("Starting training job %s" % training_job_name)
    
    resp = sm.create_training_job(
        TrainingJobName=training_job_name, AlgorithmSpecification=job['AlgorithmSpecification'], RoleArn=job['RoleArn'],
        InputDataConfig=job['InputDataConfig'], OutputDataConfig=job['OutputDataConfig'],
        ResourceConfig=job['ResourceConfig'], StoppingCondition=job['StoppingCondition'],
        HyperParameters=job['HyperParameters'] if 'HyperParameters' in job else {},
        Tags=job['Tags'] if 'Tags' in job else [])

    print(resp)
    
    waiter = sm.get_waiter('training_job_completed_or_stopped')
    waiter.wait(
        TrainingJobName=training_job_name,
    )
    
    print('Model training complete!')
    model = sm.create_model(
        ModelName=training_job_name,
        PrimaryContainer={
            'ContainerHostname': 'model-Container',
            'Image': '520713654638.dkr.ecr.us-east-1.amazonaws.com/sagemaker-mxnet:1.2.1-cpu-py3',
            'ModelDataUrl': f's3://hlin-model/sms-spam-classifier/output/{training_job_name}/output/model.tar.gz',
            'Environment': {
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                'SAGEMAKER_REGION': os.environ['AWS_REGION'],
                'SAGEMAKER_PROGRAM': 'spam_classifier.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': f's3://hlin-model/sms-spam-classifier/code/{original_training_job_name}/source/sourcedir.tar.gz',
                'MMS_DEFAULT_RESPONSE_TIMEOUT': '500'
            },
        },
        ExecutionRoleArn=job['RoleArn'],
    )
    print('Model creation complete!')
    endpoint_config = sm.create_endpoint_config(
        EndpointConfigName=training_job_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': training_job_name,
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m4.xlarge',
            },
        ],
    )
    endpoint_name = os.environ['ENDPOINT_NAME']
    sm.update_endpoint(EndpointName=endpoint_name,
                      EndpointConfigName=training_job_name)
    waiter = sm.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    print('Model deployment complete!')
