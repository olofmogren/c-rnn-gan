#!/bin/bash

cd /home/mogren/sync/code/mogren/pg-s2s/run/
SETTINGS_LOG_DIR=/home/mogren/experiments/2016-rnn-gan/logs/20161207-large_d
mkdir -p $SETTINGS_LOG_DIR

/usr/bin/qsub -cwd -l gpu=1 -e $SETTINGS_LOG_DIR/run_job.sh.error -o $SETTINGS_LOG_DIR/run_job.sh.log /home/mogren/sync/code/mogren/c-rnn-gan/run/job.sh

