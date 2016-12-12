#!/bin/bash

cd /home/mogren/sync/code/mogren/c-rnn-gan/run/

source settings2.sh

echo $SETTINGS_LOG_DIR/run_job.sh.error 
echo $SETTINGS_LOG_DIR/run_job.sh.log
/usr/bin/qsub -cwd -l gpu=1 -e $SETTINGS_LOG_DIR/run_job.sh.error -o $SETTINGS_LOG_DIR/run_job.sh.log /home/mogren/sync/code/mogren/c-rnn-gan/run/job2.sh

