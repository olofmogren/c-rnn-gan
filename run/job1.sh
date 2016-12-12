cd /home/mogren/sync/code/mogren/c-rnn-gan/

source run/settings1.sh

python rnn_gan.py --model medium --datadir ~/sync/datasets/midis --traindir $TRAIN_DIR $HYPERPARAMS

