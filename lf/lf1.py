import boto3
from botocore.exceptions import ClientError
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import email.utils
from hashlib import md5
import json
import logging
import numpy as np
import os
import urllib.parse


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

REGION = os.environ['AWS_REGION']
SAGEMAKER_ENDPOINT = os.environ['SAGEMAKER_ENDPOINT']
vocabulary_length = 9013
maketrans = str.maketrans


s3 = boto3.client("s3")
ses = boto3.client('ses', REGION)
sagemaker = boto3.client('runtime.sagemaker')


def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]
    
    
def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower:
        text = text.lower()

    translate_dict = dict((c, split) for c in filters)
    translate_map = maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results


def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data


def read_email(event):
    bucket = event['Records'][0]['s3']['bucket']['name']
    filename = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    logger.debug(f"FILENAME: {filename}")
    fileObj = s3.get_object(Bucket=bucket, Key=filename)
    msg = email.message_from_string(fileObj['Body'].read().decode('utf-8'))
    return msg


def format_reply(original_email):
    receive_date = original_email['Date']
    subject = original_email['Subject']
    body = original_email.get_payload()[0].get_payload()

    # TODO call sagemaker and get prediction
    logger.debug(f"SAGEMAKER ENDPOINT: {SAGEMAKER_ENDPOINT}")
    one_hot_test_messages = one_hot_encode([body.replace('\n', ''), ], vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)
    sm_payload = json.dumps(encoded_test_messages.tolist())
    response = sagemaker.invoke_endpoint(EndpointName=SAGEMAKER_ENDPOINT,
                                         ContentType='application/json',
                                         Body=sm_payload)
    
    result = json.loads(response['Body'].read().decode('utf-8'))
    logger.debug(f"SAGEMAKER RES: {result}")
    conf = float(result['predicted_probability'][0][0])
    if conf > 0.5:
        pred = "Spam"
    else:
        pred = "Ham"
        conf = 1 - conf

    body_sample = '<br />'.join(body[:240].split('\n'))

    return f"""
        We received your email sent at {receive_date} with the subject {subject}.<br />
        <br />
        Here is a 240 character sample of the email body:<br />
        {body_sample}
        <br />
        The email was categorized as {pred} with a {(conf * 100):.2f}% confidence.
    """


def create_reply(original_email):
    # Create a new subject line
    subject = f"RE: {original_email['Subject']}"

    # The body text of the email.
    body_text = format_reply(original_email)

    # Create a MIME container.
    msg = MIMEMultipart()
    # Create a MIME text part.
    text_part = MIMEText(body_text, _subtype="html")
    # Attach the text part to the MIME message.
    msg.attach(text_part)

    # Add subject, from and to lines.
    recipient = email.utils.parseaddr(original_email['From'])[1]
    sender = email.utils.parseaddr(original_email['To'])[1]
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient

    return {
        "Source": sender,
        "Destinations": recipient,
        "Data": msg.as_string()
    }


def send_email(message):
    # Send the email.
    try:
        #Provide the contents of the email.
        response = ses.send_raw_email(
            Source=message['Source'],
            Destinations=[
                message['Destinations']
            ],
            RawMessage={
                'Data': message['Data']
            }
        )

    # Display an error if something goes wrong.
    except ClientError as e:
        output = e.response['Error']['Message']
    else:
        output = f"Email sent with ID: {response['MessageId']}"

    return output


def lambda_handler(event, context):
    logger.debug(f"EVENT: {event}")
    msg = read_email(event)
    reply = create_reply(msg)
    res = send_email(reply)
    logger.debug(f"SEND EMAIL RES: {res}")
