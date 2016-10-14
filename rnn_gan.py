# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""

The hyperparameters used in the model:
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- songlength - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The hyperparameters that could be used in the model:
- init_scale - the initial scale of the weights

To run:

$ python rnn_gan.py --model small|medium|large --datadir simple-examples/data/ --traindir dir-for-checkpoints-and-plots --select_validation_percentage 0-40 --select_test_percentage 0-40

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, datetime, os, sys

import numpy as np
import tensorflow as tf
import cPickle as pkl

from subprocess import call, Popen

import music_data_utils

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "medium",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("datadir", None, "Directory to save and load midi music files.")
flags.DEFINE_string("traindir", None, "Directory to save checkpoints and gnuplot files.")
flags.DEFINE_integer("epochs_per_checkpoint", 10,
                     "How many training epochs to do per checkpoint.")
flags.DEFINE_string("call_after", None, "Call this command after exit.")
flags.DEFINE_integer("exit_after", 200,
                     "exit after this many minutes")
flags.DEFINE_integer("select_validation_percentage", None,
                     "Select random percentage of data as validation set.")
flags.DEFINE_integer("select_test_percentage", None,
                     "Select random percentage of data as test set.")
flags.DEFINE_float("learning_rate", None,
                     "Initial learning rate.")
flags.DEFINE_boolean("sample", False,
                     "Sample output from the model. Assume training was already done. Save sample output to file.")
flags.DEFINE_boolean("variable_ticks", False,
                     "Instead of constant tick length, use variable tick lengths. This means only one note at a time, and zero-velocity tones are put in as breaks.")
FLAGS = flags.FLAGS


def data_type():
  #return tf.float16 if FLAGS.use_fp16 else tf.float32
  return tf.float32

def linear(inp, output_dim, scope=None, stddev=1.0, reuse_scope=False):
  norm = tf.random_normal_initializer(stddev=stddev)
  const = tf.constant_initializer(0.0)
  with tf.variable_scope(scope or 'linear') as scope:
    if reuse_scope:
      scope.reuse_variables()
    w = tf.get_variable('w', [inp.get_shape()[1], output_dim], initializer=norm)
    b = tf.get_variable('b', [output_dim], initializer=const)
  return tf.matmul(inp, w) + b

class RNNGAN(object):
  """The RNNGAN model."""

  def __init__(self, is_training, config, num_song_features=None, num_meta_features=None):
    self.batch_size = batch_size = config.batch_size
    self.songlength = songlength = config.songlength
    size = config.hidden_size
    #self.global_step            = tf.Variable(0, trainable=False)

    with tf.variable_scope('G') as scope:
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)
      if is_training and config.keep_prob < 1:
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
            lstm_cell, output_keep_prob=config.keep_prob)
      cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

      self._initial_state = cell.zero_state(batch_size, data_type())

      metainputs = tf.random_uniform(shape=[batch_size, num_meta_features], minval=0.0, maxval=1.0)
      meta_g = tf.nn.relu(linear(metainputs, config.meta_layer_size, scope='meta_layer', reuse_scope=False))
      meta_softmax_w = tf.get_variable("meta_softmax_w", [config.meta_layer_size, num_meta_features])
      meta_softmax_b = tf.get_variable("meta_softmax_b", [num_meta_features])
      meta_logits = tf.nn.xw_plus_b(meta_g, meta_softmax_w, meta_softmax_b)
      meta_probs = tf.nn.softmax(meta_logits)

      rnninputs = tf.random_uniform(shape=[batch_size, songlength, num_song_features], minval=0.0, maxval=1.0)

      # Make list of tensors. One per step in recurrence.
      # Each tensor is batchsize*numfeatures.
      inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(1, songlength, rnninputs)]
      transformed = [tf.nn.relu(linear(inp, size, scope='input_layer', reuse_scope=(i!=0))) for i,inp in enumerate(inputs)]

      outputs, state = tf.nn.rnn(cell, transformed, initial_state=self._initial_state)

      self._lengths_freqs_velocities = [tf.nn.relu(linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0))) for i,output in enumerate(outputs)]

    self._final_state = state

    self._input_songdata = tf.placeholder(tf.float32, [batch_size, songlength, num_song_features])
    self._input_metadata = tf.placeholder(tf.float32, [batch_size, num_meta_features])

    # These are used both for pretraining and for D/G training further down.
    self._lr = tf.Variable(config.learning_rate, trainable=False)
    self.g_params = [v for v in tf.trainable_variables() if v.name.startswith('model/G/')]
    g_optimizer = tf.train.GradientDescentOptimizer(self._lr)
   
    inputs_attribute_splitted = tf.split(2,3,self._input_songdata)
    inputs_lengts = inputs_attribute_splitted[0]
    inputs_freqs = inputs_attribute_splitted[1]
    inputs_velocities = inputs_attribute_splitted[2]

    generated_attribute_splitted = tf.split(2,3,self._lengths_freqs_velocities)
    generated_lengts = generated_attribute_splitted[0]
    generated_freqs = generated_attribute_splitted[1]
    generated_velocities = generated_attribute_splitted[2]

    self.rnn_pretraining_loss_lengths = tf.reduce_mean(tf.squared_difference(x=tf.pack(tf.transpose(generated_lengts, perm=[1, 0, 2])), 
                                                                 y=inputs_lengts) ) #, reduction_indices=[1])
    self.rnn_pretraining_loss_freqs = tf.reduce_mean(tf.squared_difference(x=tf.pack(tf.transpose(generated_freqs, perm=[1, 0, 2])), 
                                                                 y=inputs_freqs) ) #, reduction_indices=[1])
    self.rnn_pretraining_loss_velocities = tf.reduce_mean(tf.squared_difference(x=tf.pack(tf.transpose(generated_velocities, perm=[1, 0, 2])), 
                                                                 y=inputs_velocities) ) #, reduction_indices=[1])
    self.rnn_pretraining_loss = 2*self.rnn_pretraining_loss_lengths+self.rnn_pretraining_loss_freqs+self.rnn_pretraining_loss_velocities
    pretraining_grads, _ = tf.clip_by_global_norm(tf.gradients(self.rnn_pretraining_loss, self.g_params), config.max_grad_norm)
    self.opt_pretraining = g_optimizer.apply_gradients(zip(pretraining_grads, self.g_params))

    # The discriminator tries to tell the difference between samples from the
    # true data distribution (self.x) and the generated samples (self.z).
    #
    # Here we create two copies of the discriminator network (that share parameters),
    # as you cannot use the same network with different inputs in TensorFlow.
    with tf.variable_scope('D') as scope:
      # Make list of tensors. One per step in recurrence.
      # Each tensor is batchsize*numfeatures.
      print('self._input_songdata shape {}'.format(self._input_songdata.get_shape()))
      print('generated data shape {}'.format(self._lengths_freqs_velocities[0].get_shape()))
      songdata_inputs = [tf.squeeze(input_, [1])
                for input_ in tf.split(1, songlength, self._input_songdata)]
      songdata_inputs = [tf.concat(1, [self._input_metadata, songdata_input]) for songdata_input in songdata_inputs]
      print('metadata inputs shape {}'.format(self._input_metadata.get_shape()))
      print('generated metadata shape {}'.format(meta_probs.get_shape()))
      self.discriminator_for_real_data = self.discriminator(songdata_inputs, config, is_training)
      scope.reuse_variables()
      generated_data = [tf.concat(1, [meta_probs, songdata_input]) for songdata_input in self._lengths_freqs_velocities]
      if songdata_inputs[0].get_shape() != generated_data[0].get_shape():
        print('songdata_inputs shape {} != generated data shape {}'.format(songdata_inputs[0].get_shape(), generated_data[0].get_shape()))
      self.discriminator_for_generated_data = self.discriminator(generated_data, config, is_training)

    # Define the loss for discriminator and generator networks (see the original
    # paper for details), and create optimizers for both
    self.d_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(self.discriminator_for_real_data, 1e-1000, 1.0)) \
                                 -tf.log(1 - tf.clip_by_value(self.discriminator_for_generated_data, 0.0, 1.0-1e-1000000)))
    self.g_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(self.discriminator_for_generated_data, 1e-1000000, 1.0)))

    self.d_params = [v for v in tf.trainable_variables() if v.name.startswith('model/D/')]

    if not is_training:
      return

    d_optimizer = tf.train.GradientDescentOptimizer(self._lr*config.d_lr_factor)
    d_grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.d_params),
                                        config.max_grad_norm)
    self.opt_d = d_optimizer.apply_gradients(zip(d_grads, self.d_params))
    g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params),
                                        config.max_grad_norm)
    self.opt_g = g_optimizer.apply_gradients(zip(g_grads, self.g_params))

    self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def discriminator(self, inputs, config, is_training):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=1.0, state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
    self._initial_state = cell.zero_state(config.batch_size, data_type())
    if is_training and config.keep_prob < 1:
      inputs = [tf.nn.dropout(inp, config.keep_prob) for inp in inputs]
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

    # decision = tf.sigmoid(linear(outputs[-1], 1, 'decision'))
    decisions = [tf.sigmoid(linear(output, 1, 'decision', reuse_scope=(i!=0))) for i,output in enumerate(outputs)]
    decision = tf.reduce_mean(tf.pack(decisions))
    return decision

  
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def lengths_freqs_velocities(self):
    return self._lengths_freqs_velocities

  @property
  def input_songdata(self):
    return self._input_songdata

  @property
  def input_metadata(self):
    return self._input_metadata

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.9
  d_lr_factor = 0.5
  max_grad_norm = 5
  num_layers = 2
  songlength = 200
  meta_layer_size = 300
  hidden_size = 200
  max_epoch = 40
  max_max_epoch = 500
  keep_prob = 1.0
  lr_decay = 1.0
  batch_size = 10
  biscale_slow_layer_ticks = 8
  pretraining_epochs = 20


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 0.9
  d_lr_factor = 0.5
  max_grad_norm = 5
  num_layers = 2
  songlength = 350
  meta_layer_size = 400
  hidden_size = 650
  max_epoch = 60
  max_max_epoch = 500
  keep_prob = 0.5
  lr_decay = 1.0
  batch_size = 20
  biscale_slow_layer_ticks = 8
  pretraining_epochs = 40


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 0.9
  d_lr_factor = 0.5
  max_grad_norm = 10
  num_layers = 2
  songlength = 500
  meta_layer_size = 600
  hidden_size = 1500
  max_epoch = 140
  max_max_epoch = 500
  keep_prob = 0.35
  lr_decay = 1.0
  batch_size = 20
  biscale_slow_layer_ticks = 8
  pretraining_epochs = 40


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  songlength = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 1.0
  batch_size = 20
  biscale_slow_layer_ticks = 8
  pretraining_epochs = 0


def run_epoch(session, model, loader, datasetlabel, eval_op1, eval_op2, pretraining=False, variable_ticks=False, verbose=False):
  """Runs the model on the given data."""
  #epoch_size = ((len(data) // model.batch_size) - 1) // model.songlength
  epoch_start_time = time.time()
  g_loss, d_loss = 10.0, 10.0
  g_losses, d_losses = 0.0, 0.0
  iters = 0
  #state = session.run(model.initial_state)
  loader.rewind(part=datasetlabel)
  [batch_meta, batch_song] = loader.get_batch(model.batch_size, model.songlength, part=datasetlabel, variable_ticks=variable_ticks)
  while batch_meta is not None and batch_song is not None:
    op1 = eval_op1
    op2 = eval_op2
    if datasetlabel == 'train' and not pretraining:
      if d_loss == 0.0 and g_loss == 0.0:
        print('Both G and D train loss are zero. Exiting.')
        break
        #saver.save(session, checkpoint_path, global_step=m.global_step)
        #break
      elif d_loss == 0.0:
        print('D train loss is zero. Pausing optimization. G loss: {:.3f}'.format(g_loss))
        op1 = tf.no_op()
      elif g_loss == 0.0: 
        print('G train loss is zero. Pausing optimization. D loss: {:.3f}'.format(d_loss))
        op2 = tf.no_op()
      elif g_loss < 2.0 or d_loss < 2.0:
        if g_loss*.7 > d_loss:
          print('G train loss is {:.3f}, D train loss is {:.3f}. Pausing optimization of D'.format(g_loss, d_loss))
          op1 = tf.no_op()
        elif d_loss*.7 > g_loss:
          print('G train loss is {:.3f}, D train loss is {:.3f}. Pausing optimization of G'.format(g_loss, d_loss))
        op2 = tf.no_op()
    #fetches = [model.cost, model.final_state, eval_op]
    if pretraining:
      fetches = [model.rnn_pretraining_loss, tf.no_op(), op1, op2]
    else:
      fetches = [model.g_loss, model.d_loss, op1, op2]
    feed_dict = {}
    feed_dict[model.input_songdata] = batch_song
    feed_dict[model.input_metadata] = batch_meta
    #for i, (c, h) in enumerate(model.initial_state):
    #  feed_dict[c] = state[i].c
    #  feed_dict[h] = state[i].h
    #cost, state, _ = session.run(fetches, feed_dict)
    g_loss, d_loss, _, _ = session.run(fetches, feed_dict)
    g_losses += g_loss
    if not pretraining:
      d_losses += d_loss
    iters += 1

    if verbose and iters-1 % 100 == 99:
      if pretraining:
        print("pretraining: %s: %d batch loss: G: {:.3f}, avg loss: G: %{:.3f}, speed: {:.1f} songs per sec".format(
                (datasetlabel, iters, g_loss, g_losses/iters,
                 float(iters * model.batch_size)/(time.time() - epoch_start_time))))
      else:
        print("%s: %d batch loss: G: {:.3f}, D: {:.3f}, avg loss: G: %{:.3f}, D: {:.3f} speed: {:.1f} songs per sec".format(
                (datasetlabel, iters, g_loss, d_loss, g_losses/iters, d_losses/iters,
                 float(iters * model.batch_size)/(time.time() - epoch_start_time))))
    [batch_meta, batch_song] = loader.get_batch(model.batch_size, model.songlength, part=datasetlabel)

  if iters == 0:
    return (None,None)

  g_mean_loss = g_losses/iters
  if pretraining:
    d_mean_loss = None
  else:
    d_mean_loss = d_losses/iters
  return (g_mean_loss, d_mean_loss)


def sample(session, model):
  """Samples from the generative model."""
  #state = session.run(model.initial_state)
  fetches = [model.lengths_freqs_velocities, tf.no_op()]
  feed_dict = {}
  lengths_freqs_velocities,_ = session.run(fetches, feed_dict)
  # squeeze batch dim:
  print( lengths_freqs_velocities)
  print( lengths_freqs_velocities[0].shape)
  # The following worked when batch_size=1.
  # lengths_freqs_velocities = [np.squeeze(x, axis=0) for x in lengths_freqs_velocities]
  # If batch_size != 1, we just pick the first sample. Wastefull, yes.
  lengths_freqs_velocities = [x[0,:] for x in lengths_freqs_velocities]
  return lengths_freqs_velocities

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.datadir:
    raise ValueError("Must set --datadir to midi music dir.")
  if not FLAGS.traindir:
    raise ValueError("Must set --traindir to dir where I can save model and plots.")
  
  summaries_dir = None
  plots_dir = None
  if FLAGS.traindir:
    summaries_dir = os.path.join(FLAGS.traindir, 'summaries')
    plots_dir = os.path.join(FLAGS.traindir, 'plots')
    try: os.makedirs(FLAGS.traindir)
    except: pass
    try: os.makedirs(summaries_dir)
    except: pass
    try: os.makedirs(plots_dir)
    except: pass
  
  if os.path.exists(os.path.join(FLAGS.traindir, 'global_step.pkl')):
    with open(os.path.join(FLAGS.traindir, 'global_step.pkl'), 'r') as f:
      global_step = pkl.load(f)
  else:
    global_step = 0

  songfeatures_filename = os.path.join(FLAGS.traindir, 'num_song_features.pkl')
  metafeatures_filename = os.path.join(FLAGS.traindir, 'num_meta_features.pkl')
  if FLAGS.sample and os.path.exists(songfeatures_filename) and os.path.exists(metafeatures_filename):
    with open(songfeatures_filename, 'r') as f:
      num_song_features = pkl.load(f)
    with open(metafeatures_filename, 'r') as f:
      num_meta_features = pkl.load(f)
  else:
    loader = music_data_utils.MusicDataLoader(FLAGS.datadir, FLAGS.select_validation_percentage, FLAGS.select_test_percentage)
    num_song_features = loader.get_num_song_features()
    print('num_song_features:{}'.format(num_song_features))
    num_meta_features = loader.get_num_meta_features()
    print('num_meta_features:{}'.format(num_meta_features))
    if not os.path.exists(songfeatures_filename):
      with open(songfeatures_filename, 'w') as f:
        pkl.dump(num_song_features, f)
    if not os.path.exists(metafeatures_filename):
      with open(metafeatures_filename, 'w') as f:
        pkl.dump(num_meta_features, f)

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  #eval_config.songlength = 500

  if FLAGS.learning_rate is not None:
    config.learning_rate = FLAGS.learning_rate

  train_start_time = time.time()
  checkpoint_path = os.path.join(FLAGS.traindir, "model.ckpt")

  with tf.Graph().as_default(), tf.Session() as session:
    with tf.variable_scope("model", reuse=None):
      m = RNNGAN(is_training=True, config=config, num_song_features=num_song_features, num_meta_features=num_meta_features)

    saver = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state(FLAGS.traindir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      saver.restore(session, ckpt.model_checkpoint_path)
    else:
      print("Created model with fresh parameters.")
      session.run(tf.initialize_all_variables())

    if not FLAGS.sample:
      train_g_loss,train_d_loss = 1.0,1.0
      for i in range(global_step, config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        save = False
        exit = False

        print("Epoch: {} Learning rate: {:.3f}, pretraining: {}".format(i + 1, session.run(m.lr), (i<config.pretraining_epochs)))
        if i<config.pretraining_epochs:
          train_g_loss,train_d_loss = run_epoch(session, m, loader, 'train', m.opt_pretraining, tf.no_op(), pretraining = True, variable_ticks=FLAGS.variable_ticks, verbose=True)
          print("Epoch: {} Pretraining loss: G: {:.3f}".format(i + 1, train_g_loss))
        else:
          train_g_loss,train_d_loss = run_epoch(session, m, loader, 'train', m.opt_d, m.opt_g, variable_ticks=FLAGS.variable_ticks, verbose=True)
          try:
            print("Epoch: {} Train loss: G: {:.3f}, D: {:.3f}".format(i + 1, train_g_loss, train_d_loss))
          except:
            print("Epoch: {} Train loss: G: {}, D: {}".format(i + 1, train_g_loss, train_d_loss))
          valid_g_loss,valid_d_loss = run_epoch(session, m, loader, 'validation', tf.no_op(), tf.no_op(), variable_ticks=FLAGS.variable_ticks)
          try:
            print("Epoch: {} Valid loss: G: {:.3f}, D: {:.3f}".format(i + 1, valid_g_loss, valid_d_loss))
          except:
            print("Epoch: {} Valid loss: G: {}, D: {}".format(i + 1, valid_g_loss, valid_d_loss))
        
        if train_d_loss == 0.0 and train_g_loss == 0.0:
          print('Both G and D train loss are zero. Exiting.')
          save = True
          exit = True
        if i % FLAGS.epochs_per_checkpoint == 0:
          save = True

        if save:
          saver.save(session, checkpoint_path, global_step=i)
          print('{}: Saving done!'.format(i))
          step_time, loss = 0.0, 0.0
          if train_d_loss is None: #pretraining
            train_d_loss = 0.0
            valid_d_loss = 0.0
            valid_g_loss = 0.0
          if plots_dir:
            if not os.path.exists(os.path.join(plots_dir, 'gnuplot-input.txt')):
              with open(os.path.join(plots_dir, 'gnuplot-input.txt'), 'w') as f:
                f.write('# global-step learning-rate train-g-loss train-d-loss valid-g-loss valid-d-loss\n')
            with open(os.path.join(plots_dir, 'gnuplot-input.txt'), 'a') as f:
              try:
                f.write('{} {:.4f} {:.2f} {:.2f} {:.3} {:.3f}\n'.format(i, m.lr.eval(), train_g_loss, train_d_loss, valid_g_loss, valid_d_loss))
              except:
                f.write('{} {} {} {} {} {}\n'.format(i, m.lr.eval(), train_g_loss, train_d_loss, valid_g_loss, valid_d_loss))
            if not os.path.exists(os.path.join(plots_dir, 'gnuplot-commands.txt')):
              with open(os.path.join(plots_dir, 'gnuplot-commands.txt'), 'a') as f:
                f.write('set terminal postscript eps color butt "Times" 14\nset yrange [0:400]\nset output "loss.eps"\nplot \'gnuplot-input.txt\' using ($1):($3) title \'train G\' with linespoints, \'gnuplot-input.txt\' using ($1):($4) title \'train D\' with linespoints, \'gnuplot-input.txt\' using ($1):($5) title \'valid G\' with linespoints, \'gnuplot-input.txt\' using ($1):($6) title \'valid D\' with linespoints, \n')
            Popen(['gnuplot','gnuplot-commands.txt'], cwd=plots_dir)
            
            song_data = sample(session, m)
            filename = os.path.join(plots_dir, 'out-{}-{}.mid'.format(i, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
            music_data_utils.save_data(filename, song_data, variable_ticks=FLAGS.variable_ticks)
            print('Saved {}.'.format(filename))
          
          if FLAGS.exit_after > 0 and time.time() - train_start_time > FLAGS.exit_after*60:
            print("%s: Has been running for %d seconds. Will exit (exiting after %d minutes)."%(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), (int)(time.time() - train_start_time), FLAGS.exit_after))
            exit = True
        if exit:
          with open(os.path.join(FLAGS.traindir, 'global_step.pkl'), 'w') as f:
            pkl.dump(i+1, f)
          if FLAGS.call_after is not None:
            print("%s: Will call \"%s\" before exiting."%(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), FLAGS.call_after))
            res = call(FLAGS.call_after.split(" "))
            print ('{}: call returned {}.'.format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), res))
          exit()
        sys.stdout.flush()

      test_g_loss,test_d_loss = run_epoch(session, m, loader, 'test', tf.no_op(), tf.no_op(), variable_ticks=FLAGS.variable_ticks)
      print("Test loss G: %.3f, D: %.3f" %(test_g_loss, test_d_loss))
      with open(os.path.join(FLAGS.traindir, 'global_step.pkl'), 'w') as f:
        pkl.dump(i+1, f)

    song_data = sample(session, m)
    filename = os.path.join(plots_dir, 'out-{}-{}.mid'.format(i, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
    music_data_utils.save_data(filename, song_data, variable_ticks=FLAGS.variable_ticks)
    print('Saved {}.'.format(filename))



if __name__ == "__main__":
  tf.app.run()

