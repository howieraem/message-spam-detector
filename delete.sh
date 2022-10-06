aws s3 rb --force s3://hlin-email-storage-dev
aws ses set-active-receipt-rule-set
sleep 15
aws cloudformation delete-stack --stack-name hlin-stack
aws cloudformation wait stack-delete-complete --stack-name hlin-stack
