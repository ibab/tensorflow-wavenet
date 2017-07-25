# Command Cheatsheet

## Run locally

Train the network by executing this command. The data folder must contain .dat, .emo and .pho files.

```
python trainer/task.py --train-files ./data/* --job-dir ./output
```

Generate samples live via:

```
python generate.py output/model.ckpt
```

This opens a Telnet socket that accepts EMO and PHO commands. They are being used in:

```
interactive_generator/launch.sh
```


## Run on Google Cloud

```
BUCKET_NAME="evenet-fusion-mlengine"
REGION="asia-east1"
```

```
JOB_NAME="JOB1"
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
