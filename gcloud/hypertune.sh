BUCKET_NAME="evenet-fusion-mlengine"
REGION="asia-east1"
JOB_NAME="HYPER1"

gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/$JOB_NAME \
                                                --config ./gcloud/test_config.yaml \
                                                --runtime-version 1.2 \
                                                --module-name trainer.task \
                                                --package-path trainer/ \
                                                --region $REGION \
                                                -- \
                                                --train-files gs://$BUCKET_NAME/data/fusion_1.csv \
                                                --train-steps 250
