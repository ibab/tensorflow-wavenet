# Google Cloud command cheat sheet


```
BUCKET_NAME="evenet-fusion-mlengine"
REGION="asia-east1"
```




```
python trainer/task.py --train-files ../_archive/amen_sprint_2/new_data/fusion/output --job-dir ./output
```




```
JOB_NAME="TRAIN6"
gcloud ml-engine jobs submit training $JOB_NAME --job-dir gs://$BUCKET_NAME/$JOB_NAME \
                                                --runtime-version 1.2 \
                                                --module-name trainer.task \
                                                --package-path trainer/ \
                                                --region $REGION \
                                                -- \
                                                --train-files gs://$BUCKET_NAME/data/fusion_1.csv
```




```
gcloud ml-engine local train --module-name trainer.task \
                             --package-path trainer/ \
                             -- \
                             --train-files gs://$BUCKET_NAME/data/fusion_1.csv \
                             --job-dir ./output \
```
