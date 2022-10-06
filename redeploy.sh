sh ./delete.sh
aws --region us-east-1 cloudformation deploy --template-file cf_infra.yml --stack-name hlin-stack --capabilities CAPABILITY_NAMED_IAM
aws --region us-east-1 cloudformation wait stack-create-complete --stack-name hlin-stack
aws ses set-active-receipt-rule-set --rule-set-name hlin-ses-rule-set-dev
