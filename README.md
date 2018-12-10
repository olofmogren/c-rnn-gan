# Implementation of C-RNN-GAN.

Publication:
Title: C-RNN-GAN: Continuous recurrent neural networks with adversarial training
Information: http://mogren.one/publications/2016/c-rnn-gan/

Bibtex:

@inproceedings{mogren2016crnngan, 
  title={C-RNN-GAN: A continuous recurrent neural network with adversarial training}, 
  author={Olof Mogren}, 
  booktitle={Constructive Machine Learning Workshop (CML) at NIPS 2016}, 
  pages={1}, 
  year={2016}
}


A generative adversarial model that works on continuous sequential data.
Implementation uses Python and Tensorflow, and depends on
https://github.com/vishnubob/python-midi for MIDI file IO.

## Requirements

tensorflow, python-midi (or python3-midi)

## How to run?

python rnn_gan.py --datadir "relative-path-to-data" --traindir "path-to-generated-output" --feed_previous --feature_matching --bidirectional_d --learning_rate 0.1 --pretraining_epochs 6

Author: Olof Mogren (olofmogren)
Contributors: Dhruv Sharma (dhruvsharma1992)
