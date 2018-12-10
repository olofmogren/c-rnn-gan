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
- epochs_before_decay - the number of epochs trained with the initial learning rate
- max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "epochs_before_decay"
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
import pickle as pkl
from subprocess import call, Popen

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline


import music_data_utils
from midi_statistics import get_all_stats

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("datadir", None, "Directory to save and load midi music files.")
flags.DEFINE_string("traindir", None, "Directory to save checkpoints and gnuplot files.")
flags.DEFINE_integer("epochs_per_checkpoint", 2,
                     "How many training epochs to do per checkpoint.")
flags.DEFINE_boolean("log_device_placement", False,           #
                   "Outputs info on device placement.")
flags.DEFINE_string("call_after", None, "Call this command after exit.")
flags.DEFINE_integer("exit_after", 1440,
                     "exit after this many minutes")
flags.DEFINE_integer("select_validation_percentage", None,
                     "Select random percentage of data as validation set.")
flags.DEFINE_integer("select_test_percentage", None,
                     "Select random percentage of data as test set.")
flags.DEFINE_boolean("sample", False,
                     "Sample output from the model. Assume training was already done. Save sample output to file.")
flags.DEFINE_integer("works_per_composer", None,
                     "Limit number of works per composer that is loaded.")
flags.DEFINE_boolean("disable_feed_previous", False,
                     "Feed output from previous cell to the input of the next. In the generator.")
flags.DEFINE_float("init_scale", 0.05,                # .1, .04
                   "the initial scale of the weights")
flags.DEFINE_float("learning_rate", 0.1,              # .05,.1,.9 
                   "Learning rate")
flags.DEFINE_float("d_lr_factor", 0.5,                # .5
                   "Learning rate decay")
flags.DEFINE_float("max_grad_norm", 5.0,              # 5.0, 10.0
                   "the maximum permissible norm of the gradient")
flags.DEFINE_float("keep_prob", 0.5,                  # 1.0, .35
                   "Keep probability. 1.0 disables dropout.")
flags.DEFINE_float("lr_decay", 1.0,                   # 1.0
                   "Learning rate decay after each epoch after epochs_before_decay")
flags.DEFINE_integer("num_layers_g", 2,                 # 2
                   "Number of stacked recurrent cells in G.")
flags.DEFINE_integer("num_layers_d", 2,                 # 2
                   "Number of stacked recurrent cells in D.")
flags.DEFINE_integer("songlength", 100,               # 200, 500
                   "Limit song inputs to this number of events.")
flags.DEFINE_integer("meta_layer_size", 200,          # 300, 600
                   "Size of hidden layer for meta information module.")
flags.DEFINE_integer("hidden_size_g", 100,              # 200, 1500
                   "Hidden size for recurrent part of G.")
flags.DEFINE_integer("hidden_size_d", 100,              # 200, 1500
                   "Hidden size for recurrent part of D. Default: same as for G.")
flags.DEFINE_integer("epochs_before_decay", 60,       # 40, 140
                   "Number of epochs before starting to decay.")
flags.DEFINE_integer("max_epoch", 500,                # 500, 500
                   "Number of epochs before stopping training.")
flags.DEFINE_integer("batch_size", 20,                # 10, 20
                   "Batch size.")
flags.DEFINE_integer("biscale_slow_layer_ticks", 8,   # 8
                   "Biscale slow layer ticks. Not implemented yet.")
flags.DEFINE_boolean("multiscale", False,             #
                   "Multiscale RNN. Not implemented.")
flags.DEFINE_integer("pretraining_epochs", 6,        # 20, 40
                   "Number of epochs to run lang-model style pretraining.")
flags.DEFINE_boolean("pretraining_d", False,          #
                   "Train D during pretraining.")
flags.DEFINE_boolean("initialize_d", False,           #
                   "Initialize variables for D, no matter if there are trained versions in checkpoint.")
flags.DEFINE_boolean("ignore_saved_args", False,      #
                   "Tells the program to ignore saved arguments, and instead use the ones provided as CLI arguments.")
flags.DEFINE_boolean("pace_events", False,            #
                   "When parsing input data, insert one dummy event at each quarter note if there is no tone.")
flags.DEFINE_boolean("minibatch_d", False,            #
                   "Adding kernel features for minibatch diversity.")
flags.DEFINE_boolean("unidirectional_d", False,        #
                   "Unidirectional RNN instead of bidirectional RNN for D.")
flags.DEFINE_boolean("profiling", False,              #
                   "Profiling. Writing a timeline.json file in plots dir.")
flags.DEFINE_boolean("float16", False,                #
                   "Use floa16 data type. Otherwise, use float32.")

flags.DEFINE_boolean("adam", False,                   #
                   "Use Adam optimizer.")
flags.DEFINE_boolean("feature_matching", False,       #
                   "Feature matching objective for G.")
flags.DEFINE_boolean("disable_l2_regularizer", False,       #
                   "L2 regularization on weights.")
flags.DEFINE_float("reg_scale", 1.0,       #
                   "L2 regularization scale.")
flags.DEFINE_boolean("synthetic_chords", False,       #
                   "Train on synthetically generated chords (three tones per event).")
flags.DEFINE_integer("tones_per_cell", 1,             # 2,3
                   "Maximum number of tones to output per RNN cell.")
flags.DEFINE_string("composer", None, "Specify exactly one composer, and train model only on this.")
flags.DEFINE_boolean("generate_meta", False, "Generate the composer and genre as part of output.")
flags.DEFINE_float("random_input_scale", 1.0,       #
                   "Scale of random inputs (1.0=same size as generated features).")
flags.DEFINE_boolean("end_classification", False, "Classify only in ends of D. Otherwise, does classification at every timestep and mean reduce.")

FLAGS = flags.FLAGS

model_layout_flags = ['num_layers_g', 'num_layers_d', 'meta_layer_size', 'hidden_size_g', 'hidden_size_d', 'biscale_slow_layer_ticks', 'multiscale', 'multiscale', 'disable_feed_previous', 'pace_events', 'minibatch_d', 'unidirectional_d', 'feature_matching', 'composer']

def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell,
                  state_is_tuple=True,
                  reuse=False):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the RNN.
    dropout_keep_prob: The float probability to keep the output of any given sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
    state_is_tuple: A boolean specifying whether to use tuple of hidden matrix
        and cell matrix as a state instead of a concatenated matrix.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units, state_is_tuple=state_is_tuple, reuse=reuse)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=state_is_tuple, reuse=reuse)

  return cell
def restore_flags(save_if_none_found=True):
  if FLAGS.traindir:
    saved_args_dir = os.path.join(FLAGS.traindir, 'saved_args')
    if save_if_none_found:
      try: os.makedirs(saved_args_dir)
      except: pass
    for arg in FLAGS.__flags:
      if arg not in model_layout_flags:
        continue
      if FLAGS.ignore_saved_args and os.path.exists(os.path.join(saved_args_dir, arg+'.pkl')):
        print('{:%Y-%m-%d %H:%M:%S}: saved_args: Found {} setting from saved state, but using CLI args ({}) and saving (--ignore_saved_args).'.format(datetime.datetime.today(), arg, getattr(FLAGS, arg)))
      elif os.path.exists(os.path.join(saved_args_dir, arg+'.pkl')):
        with open(os.path.join(saved_args_dir, arg+'.pkl'), 'rb') as f:
          setattr(FLAGS, arg, pkl.load(f))
          print('{:%Y-%m-%d %H:%M:%S}: saved_args: {} from saved state ({}), ignoring CLI args.'.format(datetime.datetime.today(), arg, getattr(FLAGS, arg)))
      elif save_if_none_found:
        print('{:%Y-%m-%d %H:%M:%S}: saved_args: Found no {} setting from saved state, using CLI args ({}) and saving.'.format(datetime.datetime.today(), arg, getattr(FLAGS, arg)))
        with open(os.path.join(saved_args_dir, arg+'.pkl'), 'wb') as f:
            print(getattr(FLAGS, arg),arg)
            pkl.dump(getattr(FLAGS, arg), f)
      else:
        print('{:%Y-%m-%d %H:%M:%S}: saved_args: Found no {} setting from saved state, using CLI args ({}) but not saving.'.format(datetime.datetime.today(), arg, getattr(FLAGS, arg)))



def data_type():
  return tf.float16 if FLAGS.float16 else tf.float32
  #return tf.float16

def my_reduce_mean(what_to_take_mean_over):
  return tf.reshape(what_to_take_mean_over, shape=[-1])[0]
  denom = 1.0
  #print(what_to_take_mean_over.get_shape())
  for d in what_to_take_mean_over.get_shape():
    #print(d)
    if type(d) == tf.Dimension:
      denom = denom*d.value
    else:
      denom = denom*d
  return tf.reduce_sum(what_to_take_mean_over)/denom

def linear(inp, output_dim, scope=None, stddev=1.0, reuse_scope=False):
  norm = tf.random_normal_initializer(stddev=stddev, dtype=data_type())
  const = tf.constant_initializer(0.0, dtype=data_type())
  with tf.variable_scope(scope or 'linear') as scope:
    scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
    if reuse_scope:
      scope.reuse_variables()
    #print('inp.get_shape(): {}'.format(inp.get_shape()))
    w = tf.get_variable('w', [inp.get_shape()[1], output_dim], initializer=norm, dtype=data_type())
    b = tf.get_variable('b', [output_dim], initializer=const, dtype=data_type())
  return tf.matmul(inp, w) + b

def minibatch(inp, num_kernels=25, kernel_dim=10, scope=None, msg='', reuse_scope=False):
  """
   Borrowed from http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
  """
  with tf.variable_scope(scope or 'minibatch_d') as scope:
    scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
    if reuse_scope:
      scope.reuse_variables()
  
    inp = tf.Print(inp, [inp],
            '{} inp = '.format(msg), summarize=20, first_n=20)
    x = tf.sigmoid(linear(inp, num_kernels * kernel_dim, scope))
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    activation = tf.Print(activation, [activation],
            '{} activation = '.format(msg), summarize=20, first_n=20)
    diffs = tf.expand_dims(activation, 3) - \
                tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    diffs = tf.Print(diffs, [diffs],
            '{} diffs = '.format(msg), summarize=20, first_n=20)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    abs_diffs = tf.Print(abs_diffs, [abs_diffs],
            '{} abs_diffs = '.format(msg), summarize=20, first_n=20)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    minibatch_features = tf.Print(minibatch_features, [tf.reduce_min(minibatch_features), tf.reduce_max(minibatch_features)],
            '{} minibatch_features (min,max) = '.format(msg), summarize=20, first_n=20)
  return tf.concat( [inp, minibatch_features],1)

class RNNGAN(object):
  """The RNNGAN model."""

  def __init__(self, is_training, num_song_features=None, num_meta_features=None):
    batch_size = FLAGS.batch_size
    self.batch_size =  batch_size
	
    songlength = FLAGS.songlength
    self.songlength = songlength#self.global_step            = tf.Variable(0, trainable=False)

    print('songlength: {}'.format(self.songlength))
    self._input_songdata = tf.placeholder(shape=[batch_size, songlength, num_song_features], dtype=data_type())
    self._input_metadata = tf.placeholder(shape=[batch_size, num_meta_features], dtype=data_type())
    #_split = tf.split(self._input_songdata,songlength,1)[0]
    print("self._input_songdata",self._input_songdata, 'songlength',songlength)
    #print(tf.squeeze(_split,[1]))
    songdata_inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(self._input_songdata,songlength,1)]
  
    
    with tf.variable_scope('G') as scope:
      scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
      #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size_g, forget_bias=1.0, state_is_tuple=True)
      if is_training and FLAGS.keep_prob < 1:
        #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
        #    lstm_cell, output_keep_prob=FLAGS.keep_prob)
        cell = make_rnn_cell([FLAGS.hidden_size_g]*FLAGS.num_layers_g,dropout_keep_prob=FLAGS.keep_prob)
      else:
         cell = make_rnn_cell([FLAGS.hidden_size_g]*FLAGS.num_layers_g)	  

      #cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range( FLAGS.num_layers_g)], state_is_tuple=True)
      self._initial_state = cell.zero_state(batch_size, data_type())

      # TODO: (possibly temporarily) disabling meta info
      if FLAGS.generate_meta:
        metainputs = tf.random_uniform(shape=[batch_size, int(FLAGS.random_input_scale*num_meta_features)], minval=0.0, maxval=1.0)
        meta_g = tf.nn.relu(linear(metainputs, FLAGS.meta_layer_size, scope='meta_layer', reuse_scope=False))
        meta_softmax_w = tf.get_variable("meta_softmax_w", [FLAGS.meta_layer_size, num_meta_features])
        meta_softmax_b = tf.get_variable("meta_softmax_b", [num_meta_features])
        meta_logits = tf.nn.xw_plus_b(meta_g, meta_softmax_w, meta_softmax_b)
        meta_probs = tf.nn.softmax(meta_logits)

      random_rnninputs = tf.random_uniform(shape=[batch_size, songlength, int(FLAGS.random_input_scale*num_song_features)], minval=0.0, maxval=1.0, dtype=data_type())

      # Make list of tensors. One per step in recurrence.
      # Each tensor is batchsize*numfeatures.
      
      random_rnninputs = [tf.squeeze(input_, [1]) for input_ in tf.split( random_rnninputs,songlength,1)]
      
      # REAL GENERATOR:
      state = self._initial_state
      # as we feed the output as the input to the next, we 'invent' the initial 'output'.
      generated_point = tf.random_uniform(shape=[batch_size, num_song_features], minval=0.0, maxval=1.0, dtype=data_type())
      outputs = []
      self._generated_features = []
      for i,input_ in enumerate(random_rnninputs):
        if i > 0: scope.reuse_variables()
        concat_values = [input_]
        if not FLAGS.disable_feed_previous:
          concat_values.append(generated_point)
        if FLAGS.generate_meta:
          concat_values.append(meta_probs)
        if len(concat_values):
          input_ = tf.concat(axis=1, values=concat_values)
        input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_g,
                            scope='input_layer', reuse_scope=(i!=0)))
        output, state = cell(input_, state)
        outputs.append(output)
        #generated_point = tf.nn.relu(linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0)))
        generated_point = linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0))
        self._generated_features.append(generated_point)
      
      
      # PRETRAINING GENERATOR, will feed inputs, not generated outputs:
      scope.reuse_variables()
      # as we feed the output as the input to the next, we 'invent' the initial 'output'.
      prev_target = tf.random_uniform(shape=[batch_size, num_song_features], minval=0.0, maxval=1.0, dtype=data_type())
      outputs = []
      self._generated_features_pretraining = []
      for i,input_ in enumerate(random_rnninputs):
        concat_values = [input_]
        if not FLAGS.disable_feed_previous:
          concat_values.append(prev_target)
        if FLAGS.generate_meta:
          concat_values.append(self._input_metadata)
        if len(concat_values):
          input_ = tf.concat(axis=1, values=concat_values)
        input_ = tf.nn.relu(linear(input_, FLAGS.hidden_size_g, scope='input_layer', reuse_scope=(i!=0)))
        output, state = cell(input_, state)
        outputs.append(output)
        #generated_point = tf.nn.relu(linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0)))
        generated_point = linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0))
        self._generated_features_pretraining.append(generated_point)
        prev_target = songdata_inputs[i]
      
      #outputs, state = tf.nn.rnn(cell, transformed, initial_state=self._initial_state)

      #self._generated_features = [tf.nn.relu(linear(output, num_song_features, scope='output_layer', reuse_scope=(i!=0))) for i,output in enumerate(outputs)]

    self._final_state = state

    # These are used both for pretraining and for D/G training further down.
    self._lr = tf.Variable(FLAGS.learning_rate, trainable=False, dtype=data_type())
    self.g_params = [v for v in tf.trainable_variables() if v.name.startswith('model/G/')]
    if FLAGS.adam:
      g_optimizer = tf.train.AdamOptimizer(self._lr)
    else:
      g_optimizer = tf.train.GradientDescentOptimizer(self._lr)
   
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.1  # Choose an appropriate one.
    reg_loss = reg_constant * sum(reg_losses)
    reg_loss = tf.Print(reg_loss, reg_losses,
                  'reg_losses = ', summarize=20, first_n=20)
    #if not FLAGS.disable_l2_regularizer:
    #  print('L2 regularization. Reg losses: {}'.format([v.name for v in reg_losses]))
   
    # ---BEGIN, PRETRAINING. ---
    
    print(tf.transpose(tf.stack(self._generated_features_pretraining), perm=[1, 0, 2]).get_shape())
    print(self._input_songdata.get_shape())
    self.rnn_pretraining_loss = tf.reduce_mean(tf.squared_difference(x=tf.transpose(tf.stack(self._generated_features_pretraining), perm=[1, 0, 2]), y=self._input_songdata))
    if not FLAGS.disable_l2_regularizer:
      self.rnn_pretraining_loss = self.rnn_pretraining_loss+reg_loss
    
    
    pretraining_grads, _ = tf.clip_by_global_norm(tf.gradients(self.rnn_pretraining_loss, self.g_params), FLAGS.max_grad_norm)
    self.opt_pretraining = g_optimizer.apply_gradients(zip(pretraining_grads, self.g_params))

    # ---END, PRETRAINING---

    # The discriminator tries to tell the difference between samples from the
    # true data distribution (self.x) and the generated samples (self.z).
    #
    # Here we create two copies of the discriminator network (that share parameters),
    # as you cannot use the same network with different inputs in TensorFlow.
    with tf.variable_scope('D') as scope:
      scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
      # Make list of tensors. One per step in recurrence.
      # Each tensor is batchsize*numfeatures.
      # TODO: (possibly temporarily) disabling meta info
      print('self._input_songdata shape {}'.format(self._input_songdata.get_shape()))
      print('generated data shape {}'.format(self._generated_features[0].get_shape()))
      # TODO: (possibly temporarily) disabling meta info
      if FLAGS.generate_meta:
        songdata_inputs = [tf.concat([self._input_metadata, songdata_input],1) for songdata_input in songdata_inputs]
      #print(songdata_inputs[0])
      #print(songdata_inputs[0])
      #print('metadata inputs shape {}'.format(self._input_metadata.get_shape()))
      #print('generated metadata shape {}'.format(meta_probs.get_shape()))
      self.real_d,self.real_d_features = self.discriminator(songdata_inputs, is_training, msg='real')
      scope.reuse_variables()
      # TODO: (possibly temporarily) disabling meta info
      if FLAGS.generate_meta:
        generated_data = [tf.concat([meta_probs, songdata_input],1) for songdata_input in self._generated_features]
      else:
        generated_data = self._generated_features
      if songdata_inputs[0].get_shape() != generated_data[0].get_shape():
        print('songdata_inputs shape {} != generated data shape {}'.format(songdata_inputs[0].get_shape(), generated_data[0].get_shape()))
      self.generated_d,self.generated_d_features = self.discriminator(generated_data, is_training, msg='generated')

    # Define the loss for discriminator and generator networks (see the original
    # paper for details), and create optimizers for both
    self.d_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(self.real_d, 1e-1000000, 1.0)) \
                                 -tf.log(1 - tf.clip_by_value(self.generated_d, 0.0, 1.0-1e-1000000)))
    self.g_loss_feature_matching = tf.reduce_sum(tf.squared_difference(self.real_d_features, self.generated_d_features))
    self.g_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(self.generated_d, 1e-1000000, 1.0)))

    if not FLAGS.disable_l2_regularizer:
      self.d_loss = self.d_loss+reg_loss
      self.g_loss_feature_matching = self.g_loss_feature_matching+reg_loss
      self.g_loss = self.g_loss+reg_loss
    self.d_params = [v for v in tf.trainable_variables() if v.name.startswith('model/D/')]

    if not is_training:
      return

    d_optimizer = tf.train.GradientDescentOptimizer(self._lr*FLAGS.d_lr_factor)
    d_grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.d_params),
                                        FLAGS.max_grad_norm)
    self.opt_d = d_optimizer.apply_gradients(zip(d_grads, self.d_params))
    if FLAGS.feature_matching:
      g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss_feature_matching,
                                                       self.g_params),
                                        FLAGS.max_grad_norm)
    else:
      g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.g_loss, self.g_params),
                                        FLAGS.max_grad_norm)
    self.opt_g = g_optimizer.apply_gradients(zip(g_grads, self.g_params))

    self._new_lr = tf.placeholder(shape=[], name="new_learning_rate", dtype=data_type())
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def discriminator(self, inputs, is_training, msg=''):
    # RNN discriminator:
    #for i in xrange(len(inputs)):
    #  print('shape inputs[{}] {}'.format(i, inputs[i].get_shape()))
    #inputs[0] = tf.Print(inputs[0], [inputs[0]],
    #        '{} inputs[0] = '.format(msg), summarize=20, first_n=20)
    if is_training and FLAGS.keep_prob < 1:
      inputs = [tf.nn.dropout(input_, FLAGS.keep_prob) for input_ in inputs]
    
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size_d, forget_bias=1.0, state_is_tuple=True)
    if is_training and FLAGS.keep_prob < 1:
      #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
      #lstm_cell, output_keep_prob=FLAGS.keep_prob)
      cell_fw = make_rnn_cell([FLAGS.hidden_size_d]* FLAGS.num_layers_d,dropout_keep_prob=FLAGS.keep_prob)
      
      cell_bw = make_rnn_cell([FLAGS.hidden_size_d]* FLAGS.num_layers_d,dropout_keep_prob=FLAGS.keep_prob)
    else:
      cell_fw = make_rnn_cell([FLAGS.hidden_size_d]* FLAGS.num_layers_d)
      
      cell_bw = make_rnn_cell([FLAGS.hidden_size_d]* FLAGS.num_layers_d)
    #cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range( FLAGS.num_layers_d)], state_is_tuple=True)
    self._initial_state_fw = cell_fw.zero_state(self.batch_size, data_type())
    if not FLAGS.unidirectional_d:
      #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size_g, forget_bias=1.0, state_is_tuple=True)
      #if is_training and FLAGS.keep_prob < 1:
      #  lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
      #      lstm_cell, output_keep_prob=FLAGS.keep_prob)
      #cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range( FLAGS.num_layers_d)], state_is_tuple=True)
      self._initial_state_bw = cell_bw.zero_state(self.batch_size, data_type())
      print("cell_fw",cell_fw.output_size)
      #print("cell_bw",cell_bw.output_size)
      #print("inputs",inputs)
      #print("initial_state_fw",self._initial_state_fw)
      #print("initial_state_bw",self._initial_state_bw)
      outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw, cell_bw, inputs, initial_state_fw=self._initial_state_fw, initial_state_bw=self._initial_state_bw)
      #outputs[0] = tf.Print(outputs[0], [outputs[0]],
      #        '{} outputs[0] = '.format(msg), summarize=20, first_n=20)
      #state = tf.concat(state_fw, state_bw)
      #endoutput = tf.concat(concat_dim=1, values=[outputs[0],outputs[-1]])
    else:
      outputs, state = tf.nn.rnn(cell_fw, inputs, initial_state=self._initial_state_fw)
      #state = self._initial_state
	  
      #outputs, state = cell_fw(tf.convert_to_tensor (inputs),state)
      #endoutput = outputs[-1]

    if FLAGS.minibatch_d:
      outputs = [minibatch(tf.reshape(outp, shape=[FLAGS.batch_size, -1]), msg=msg, reuse_scope=(i!=0)) for i,outp in enumerate(outputs)]
    # decision = tf.sigmoid(linear(outputs[-1], 1, 'decision'))
    if FLAGS.end_classification:
      decisions = [tf.sigmoid(linear(output, 1, 'decision', reuse_scope=(i!=0))) for i,output in enumerate([outputs[0], outputs[-1]])]
      decisions = tf.stack(decisions)
      decisions = tf.transpose(decisions, perm=[1,0,2])
      print('shape, decisions: {}'.format(decisions.get_shape()))
    else:
      decisions = [tf.sigmoid(linear(output, 1, 'decision', reuse_scope=(i!=0))) for i,output in enumerate(outputs)]
      decisions = tf.stack(decisions)
      decisions = tf.transpose(decisions, perm=[1,0,2])
      print('shape, decisions: {}'.format(decisions.get_shape()))
    decision = tf.reduce_mean(decisions, reduction_indices=[1,2])
    decision = tf.Print(decision, [decision],
            '{} decision = '.format(msg), summarize=20, first_n=20)
    return (decision,tf.transpose(tf.stack(outputs), perm=[1,0,2]))
      

  
  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def generated_features(self):
    return self._generated_features

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



def run_epoch(session, model, loader, datasetlabel, eval_op_g, eval_op_d, pretraining=False, verbose=False, run_metadata=None, pretraining_d=False):
  """Runs the model on the given data."""
  #epoch_size = ((len(data) // model.batch_size) - 1) // model.songlength
  epoch_start_time = time.time()
  g_loss, d_loss = 10.0, 10.0
  g_losses, d_losses = 0.0, 0.0
  iters = 0
  #state = session.run(model.initial_state)
  time_before_graph = None
  time_after_graph = None
  times_in_graph = []
  times_in_python = []
  #times_in_batchreading = []
  loader.rewind(part=datasetlabel)
  [batch_meta, batch_song] = loader.get_batch(model.batch_size, model.songlength, part=datasetlabel)

  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

  while batch_meta is not None and batch_song is not None:
    op_g = eval_op_g
    op_d = eval_op_d
    if datasetlabel == 'train' and not pretraining: # and not FLAGS.feature_matching:
      if d_loss == 0.0 and g_loss == 0.0:
        print('Both G and D train loss are zero. Exiting.')
        break
        #saver.save(session, checkpoint_path, global_step=m.global_step)
        #break
      elif d_loss == 0.0:
        #print('D train loss is zero. Freezing optimization. G loss: {:.3f}'.format(g_loss))
        op_g = tf.no_op()
      elif g_loss == 0.0: 
        #print('G train loss is zero. Freezing optimization. D loss: {:.3f}'.format(d_loss))
        op_d = tf.no_op()
      elif g_loss < 2.0 or d_loss < 2.0:
        if g_loss*.7 > d_loss:
          #print('G train loss is {:.3f}, D train loss is {:.3f}. Freezing optimization of D'.format(g_loss, d_loss))
          op_g = tf.no_op()
        #elif d_loss*.7 > g_loss:
          #print('G train loss is {:.3f}, D train loss is {:.3f}. Freezing optimization of G'.format(g_loss, d_loss))
        op_d = tf.no_op()
    #fetches = [model.cost, model.final_state, eval_op]
    if pretraining:
      if pretraining_d:
        fetches = [model.rnn_pretraining_loss, model.d_loss, op_g, op_d]
      else:
        fetches = [model.rnn_pretraining_loss, tf.no_op(), op_g, op_d]
    else:
      fetches = [model.g_loss, model.d_loss, op_g, op_d]
    feed_dict = {}
    feed_dict[model.input_songdata.name] = batch_song
    feed_dict[model.input_metadata.name] = batch_meta
    #print(batch_song)
    #print(batch_song.shape)
    
    #for i, (c, h) in enumerate(model.initial_state):
    #  feed_dict[c] = state[i].c
    #  feed_dict[h] = state[i].h
    #cost, state, _ = session.run(fetches, feed_dict)
    time_before_graph = time.time()
    if iters > 0:
      times_in_python.append(time_before_graph-time_after_graph)
    if run_metadata:
      g_loss, d_loss, _, _ = session.run(fetches, feed_dict, options=run_options, run_metadata=run_metadata)
    else:
      g_loss, d_loss, _, _ = session.run(fetches, feed_dict)
    time_after_graph = time.time()
    if iters > 0:
      times_in_graph.append(time_after_graph-time_before_graph)
    g_losses += g_loss
    if not pretraining:
      d_losses += d_loss
    iters += 1

    if verbose and iters % 10 == 9:
      songs_per_sec = float(iters * model.batch_size)/float(time.time() - epoch_start_time)
      avg_time_in_graph = float(sum(times_in_graph))/float(len(times_in_graph))
      avg_time_in_python = float(sum(times_in_python))/float(len(times_in_python))
      #avg_time_batchreading = float(sum(times_in_batchreading))/float(len(times_in_batchreading))
      if pretraining:
        print("{}: {} (pretraining) batch loss: G: {:.3f}, avg loss: G: {:.3f}, speed: {:.1f} songs/s, avg in graph: {:.1f}, avg in python: {:.1f}.".format(datasetlabel, iters, g_loss, float(g_losses)/float(iters), songs_per_sec, avg_time_in_graph, avg_time_in_python))
      else:
        print("{}: {} batch loss: G: {:.3f}, D: {:.3f}, avg loss: G: {:.3f}, D: {:.3f} speed: {:.1f} songs/s, avg in graph: {:.1f}, avg in python: {:.1f}.".format(datasetlabel, iters, g_loss, d_loss, float(g_losses)/float(iters), float(d_losses)/float(iters),songs_per_sec, avg_time_in_graph, avg_time_in_python))
    #batchtime = time.time()
    [batch_meta, batch_song] = loader.get_batch(model.batch_size, model.songlength, part=datasetlabel)
    #times_in_batchreading.append(time.time()-batchtime)

  if iters == 0:
    return (None,None)

  g_mean_loss = g_losses/iters
  if pretraining and not pretraining_d:
    d_mean_loss = None
  else:
    d_mean_loss = d_losses/iters
  return (g_mean_loss, d_mean_loss)


def sample(session, model, batch=False):
  """Samples from the generative model."""
  #state = session.run(model.initial_state)
  fetches = [model.generated_features]
  feed_dict = {}
  generated_features, = session.run(fetches, feed_dict)
  #print( generated_features)
  print( generated_features[0].shape)
  # The following worked when batch_size=1.
  # generated_features = [np.squeeze(x, axis=0) for x in generated_features]
  # If batch_size != 1, we just pick the first sample. Wastefull, yes.
  returnable = []
  if batch:
    for batchno in range(generated_features[0].shape[0]):
      returnable.append([x[batchno,:] for x in generated_features])
  else:
    returnable = [x[0,:] for x in generated_features]
  return returnable

def main(_):
  if not FLAGS.datadir:
    raise ValueError("Must set --datadir to midi music dir.")
  if not FLAGS.traindir:
    raise ValueError("Must set --traindir to dir where I can save model and plots.")
 
  restore_flags()
 
  summaries_dir = None
  plots_dir = None
  generated_data_dir = None
  summaries_dir = os.path.join(FLAGS.traindir, 'summaries')
  plots_dir = os.path.join(FLAGS.traindir, 'plots')
  generated_data_dir = os.path.join(FLAGS.traindir, 'generated_data')
  try: os.makedirs(FLAGS.traindir)
  except: pass
  try: os.makedirs(summaries_dir)
  except: pass
  try: os.makedirs(plots_dir)
  except: pass
  try: os.makedirs(generated_data_dir)
  except: pass
  directorynames = FLAGS.traindir.split('/')
  experiment_label = ''
  while not experiment_label:
    experiment_label = directorynames.pop()
  
  global_step = -1
  if os.path.exists(os.path.join(FLAGS.traindir, 'global_step.pkl')):
    with open(os.path.join(FLAGS.traindir, 'global_step.pkl'), 'r') as f:
      global_step = pkl.load(f)
  global_step += 1

  songfeatures_filename = os.path.join(FLAGS.traindir, 'num_song_features.pkl')
  metafeatures_filename = os.path.join(FLAGS.traindir, 'num_meta_features.pkl')
  synthetic=None
  if FLAGS.synthetic_chords:
    synthetic = 'chords'
    print('Training on synthetic chords!')
  if FLAGS.composer is not None:
    print('Single composer: {}'.format(FLAGS.composer))
  loader = music_data_utils.MusicDataLoader(FLAGS.datadir, FLAGS.select_validation_percentage, FLAGS.select_test_percentage, FLAGS.works_per_composer, FLAGS.pace_events, synthetic=synthetic, tones_per_cell=FLAGS.tones_per_cell, single_composer=FLAGS.composer)
  if FLAGS.synthetic_chords:
    # This is just a print out, to check the generated data.
    batch = loader.get_batch(batchsize=1, songlength=400)
    loader.get_midi_pattern([batch[1][0][i] for i in xrange(batch[1].shape[1])])

  num_song_features = loader.get_num_song_features()
  print('num_song_features:{}'.format(num_song_features))
  num_meta_features = loader.get_num_meta_features()
  print('num_meta_features:{}'.format(num_meta_features))

  train_start_time = time.time()
  checkpoint_path = os.path.join(FLAGS.traindir, "model.ckpt")

  songlength_ceiling = FLAGS.songlength

  if global_step < FLAGS.pretraining_epochs:
    FLAGS.songlength = int(min(((global_step+10)/10)*10,songlength_ceiling))
    FLAGS.songlength = int(min((global_step+1)*4,songlength_ceiling))
 
  with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as session:
    with tf.variable_scope("model", reuse=None) as scope:
      scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
      m = RNNGAN(is_training=True, num_song_features=num_song_features, num_meta_features=num_meta_features)


    if FLAGS.initialize_d:
      vars_to_restore = {}
      for v in tf.trainable_variables():
        if v.name.startswith('model/G/'):
          print(v.name[:-2])
          vars_to_restore[v.name[:-2]] = v
      saver = tf.train.Saver(vars_to_restore)
      ckpt = tf.train.get_checkpoint_state(FLAGS.traindir)
      if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path,end=" ")
        saver.restore(session, ckpt.model_checkpoint_path)
        session.run(tf.initialize_variables([v for v in tf.trainable_variables() if v.name.startswith('model/D/')]))
      else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
      saver = tf.train.Saver(tf.all_variables())
    else:
      saver = tf.train.Saver(tf.all_variables())
      ckpt = tf.train.get_checkpoint_state(FLAGS.traindir)
      if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
      else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())

    run_metadata = None
    if FLAGS.profiling:
      run_metadata = tf.RunMetadata()
    if not FLAGS.sample:
      train_g_loss,train_d_loss = 1.0,1.0
      for i in range(global_step, FLAGS.max_epoch):
        lr_decay = FLAGS.lr_decay ** max(i - FLAGS.epochs_before_decay, 0.0)

        if global_step < FLAGS.pretraining_epochs:
          #new_songlength = int(min(((i+10)/10)*10,songlength_ceiling))
          new_songlength = int(min((i+1)*4,songlength_ceiling))
        else:
          new_songlength = songlength_ceiling
        if new_songlength != FLAGS.songlength:
          print('Changing songlength, now training on {} events from songs.'.format(new_songlength))
          FLAGS.songlength = new_songlength
          with tf.variable_scope("model", reuse=True) as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale))
            m = RNNGAN(is_training=True, num_song_features=num_song_features, num_meta_features=num_meta_features)

        if not FLAGS.adam:
          m.assign_lr(session, FLAGS.learning_rate * lr_decay)

        save = False
        do_exit = False

        print("Epoch: {} Learning rate: {:.3f}, pretraining: {}".format(i, session.run(m.lr), (i<FLAGS.pretraining_epochs)))
        if i<FLAGS.pretraining_epochs:
          opt_d = tf.no_op()
          if FLAGS.pretraining_d:
            opt_d = m.opt_d
          train_g_loss,train_d_loss = run_epoch(session, m, loader, 'train', m.opt_pretraining, opt_d, pretraining = True, verbose=True, run_metadata=run_metadata, pretraining_d=FLAGS.pretraining_d)
          if FLAGS.pretraining_d:
            try:
              print("Epoch: {} Pretraining loss: G: {:.3f}, D: {:.3f}".format(i, train_g_loss, train_d_loss))
            except:
              print(train_g_loss)
              print(train_d_loss)
          else:
            print("Epoch: {} Pretraining loss: G: {:.3f}".format(i, train_g_loss))
        else:
          train_g_loss,train_d_loss = run_epoch(session, m, loader, 'train', m.opt_d, m.opt_g, verbose=True, run_metadata=run_metadata)
          try:
            print("Epoch: {} Train loss: G: {:.3f}, D: {:.3f}".format(i, train_g_loss, train_d_loss))
          except:
            print("Epoch: {} Train loss: G: {}, D: {}".format(i, train_g_loss, train_d_loss))
        valid_g_loss,valid_d_loss = run_epoch(session, m, loader, 'validation', tf.no_op(), tf.no_op())
        try:
          print("Epoch: {} Valid loss: G: {:.3f}, D: {:.3f}".format(i, valid_g_loss, valid_d_loss))
        except:
          print("Epoch: {} Valid loss: G: {}, D: {}".format(i, valid_g_loss, valid_d_loss))
        
        if train_d_loss == 0.0 and train_g_loss == 0.0:
          print('Both G and D train loss are zero. Exiting.')
          save = True
          do_exit = True
        if i % FLAGS.epochs_per_checkpoint == 0:
          save = True
        if FLAGS.exit_after > 0 and time.time() - train_start_time > FLAGS.exit_after*60:
          print("%s: Has been running for %d seconds. Will exit (exiting after %d minutes)."%(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), (int)(time.time() - train_start_time), FLAGS.exit_after))
          save = True
          do_exit = True

        if save:
          saver.save(session, checkpoint_path, global_step=i)
          with open(os.path.join(FLAGS.traindir, 'global_step.pkl'), 'wb') as f:
            pkl.dump(i, f)
          if FLAGS.profiling:
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(os.path.join(plots_dir, 'timeline.json'), 'w') as f:
              f.write(ctf)
          print('{}: Saving done!'.format(i))

        step_time, loss = 0.0, 0.0
        if train_d_loss is None: #pretraining
          train_d_loss = 0.0
          valid_d_loss = 0.0
          valid_g_loss = 0.0
        if not os.path.exists(os.path.join(plots_dir, 'gnuplot-input.txt')):
          with open(os.path.join(plots_dir, 'gnuplot-input.txt'), 'w') as f:
            f.write('# global-step learning-rate train-g-loss train-d-loss valid-g-loss valid-d-loss\n')
        with open(os.path.join(plots_dir, 'gnuplot-input.txt'), 'a') as f:
          try:
            f.write('{} {:.4f} {:.2f} {:.2f} {:.3} {:.3f}\n'.format(i, m.lr.eval(), train_g_loss, train_d_loss, valid_g_loss, valid_d_loss))
          except:
            f.write('{} {} {} {} {} {}\n'.format(i, m.lr.eval(), train_g_loss, train_d_loss, valid_g_loss, valid_d_loss))
        if not os.path.exists(os.path.join(plots_dir, 'gnuplot-commands-loss.txt')):
          with open(os.path.join(plots_dir, 'gnuplot-commands-loss.txt'), 'a') as f:
            f.write('set terminal postscript eps color butt "Times" 14\nset yrange [0:400]\nset output "loss.eps"\nplot \'gnuplot-input.txt\' using ($1):($3) title \'train G\' with linespoints, \'gnuplot-input.txt\' using ($1):($4) title \'train D\' with linespoints, \'gnuplot-input.txt\' using ($1):($5) title \'valid G\' with linespoints, \'gnuplot-input.txt\' using ($1):($6) title \'valid D\' with linespoints, \n')
        if not os.path.exists(os.path.join(plots_dir, 'gnuplot-commands-midistats.txt')):
          with open(os.path.join(plots_dir, 'gnuplot-commands-midistats.txt'), 'a') as f:
            f.write('set terminal postscript eps color butt "Times" 14\nset yrange [0:127]\nset xrange [0:70]\nset output "midistats.eps"\nplot \'midi_stats.gnuplot\' using ($1):(100*$3) title \'Scale consistency, %\' with linespoints, \'midi_stats.gnuplot\' using ($1):($6) title \'Tone span, halftones\' with linespoints, \'midi_stats.gnuplot\' using ($1):($10) title \'Unique tones\' with linespoints, \'midi_stats.gnuplot\' using ($1):($23) title \'Intensity span, units\' with linespoints, \'midi_stats.gnuplot\' using ($1):(100*$24) title \'Polyphony, %\' with linespoints, \'midi_stats.gnuplot\' using ($1):($12) title \'3-tone repetitions\' with linespoints\n')
        try:
          Popen(['gnuplot','gnuplot-commands-loss.txt'], cwd=plots_dir)
          Popen(['gnuplot','gnuplot-commands-midistats.txt'], cwd=plots_dir)
        except:
          print('failed to run gnuplot. Please do so yourself: gnuplot gnuplot-commands.txt cwd={}'.format(plots_dir))
        
        song_data = sample(session, m, batch=True)
        midi_patterns = []
        print('formatting midi...')
        midi_time = time.time()
        for d in song_data:
          midi_patterns.append(loader.get_midi_pattern(d))
        print('done. time: {}'.format(time.time()-midi_time))
        
        filename = os.path.join(generated_data_dir, 'out-{}-{}-{}.mid'.format(experiment_label, i, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
        loader.save_midi_pattern(filename, midi_patterns[0])
  
        stats = []
        print('getting stats...')
        stats_time = time.time()
        for p in midi_patterns:
          stats.append(get_all_stats(p))
        print('done. time: {}'.format(time.time()-stats_time))
        #print(stats)
        stats = [stat for stat in stats if stat is not None]
        if len(stats):
          stats_keys_string = ['scale']
          stats_keys = ['scale_score', 'tone_min', 'tone_max', 'tone_span', 'freq_min', 'freq_max', 'freq_span', 'tones_unique', 'repetitions_2', 'repetitions_3', 'repetitions_4', 'repetitions_5', 'repetitions_6', 'repetitions_7', 'repetitions_8', 'repetitions_9', 'estimated_beat', 'estimated_beat_avg_ticks_off', 'intensity_min', 'intensity_max', 'intensity_span', 'polyphony_score', 'top_2_interval_difference', 'top_3_interval_difference', 'num_tones']
          statsfilename = os.path.join(plots_dir, 'midi_stats.gnuplot')
          if not os.path.exists(statsfilename):
            with open(statsfilename, 'a') as f:
              f.write('# Average numers over one minibatch of size {}.\n'.format(FLAGS.batch_size))
              f.write('# global-step {} {}\n'.format(' '.join([s.replace(' ', '_') for s in stats_keys_string]), ' '.join(stats_keys)))
          with open(statsfilename, 'a') as f:
            f.write('{} {} {}\n'.format(i, ' '.join(['{}'.format(stats[0][key].replace(' ', '_')) for key in stats_keys_string]), ' '.join(['{:.3f}'.format(sum([s[key] for s in stats])/float(len(stats))) for key in stats_keys])))
          print('Saved {}.'.format(filename))
          
        if do_exit:
          if FLAGS.call_after is not None:
            print("%s: Will call \"%s\" before exiting."%(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), FLAGS.call_after))
            res = call(FLAGS.call_after.split(" "))
            print ('{}: call returned {}.'.format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), res))
          exit()
        sys.stdout.flush()


      test_g_loss,test_d_loss = run_epoch(session, m, loader, 'test', tf.no_op(), tf.no_op())
      print("Test loss G: %.3f, D: %.3f" %(test_g_loss, test_d_loss))

    song_data = sample(session, m)
    filename = os.path.join(generated_data_dir, 'out-{}-{}-{}.mid'.format(experiment_label, i, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
    loader.save_data(filename, song_data)
    print('Saved {}.'.format(filename))



if __name__ == "__main__":
  tf.app.run()

