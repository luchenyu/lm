# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math, os, random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.client import timeline

import data_utils, model_utils


class Seq2SeqModel(object):
  """Sequence-to-class model with multiple buckets.

     implements multiple classifiers
  """

  def __init__(self, vocab_size, buckets, size, num_layers, max_gradient, batch_size, 
               learning_rate, learning_rate_decay_factor, data_dir, cell_type="GRU",
               is_decoding=False):
    """Create the model.

    Args:
      vocab_size: size of the vocabulary.
      num_class: num of output classes
      buckets: a list of size of the input sequence
      size: number of units in each layer of the model.
      num_layers: number of layers.
      max_gradient: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      cell_type: choose between LSTM cells and GRU cells.
      is_decoding: if set, we do decoding instead of training.
    """
    self.vocab_size = vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    with tf.variable_scope("OptimizeLoss"):
      self.learning_rate = tf.get_variable("learning_rate", [], trainable=False, 
          initializer=tf.constant_initializer(learning_rate))
    self.global_step = tf.Variable(0, trainable=False)
    self.is_decoding = is_decoding
    global is_training
    is_training = not is_decoding


    def get_batch(bucket_id, num_epoch=None):
      bucket_path = os.path.join(data_dir, "train_tfrecords", "bucket_"+str(bucket_id))
      filenames = [os.path.join(bucket_path, f) for f in os.listdir(bucket_path)]
      filename_queue = tf.train.string_input_producer(filenames, num_epochs=None, shuffle=True, capacity=10000)
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)
      features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
              'seq': tf.FixedLenFeature([], tf.string),
          })
      seq = tf.decode_raw(features['seq'], tf.int32)
      seq.set_shape([buckets[i]])
      batch_seq = tf.train.shuffle_batch(
          [seq], batch_size=batch_size, num_threads=5,
          capacity=100000 + 10 * batch_size,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=100000)
      return batch_seq

    # The rnn classifier function: we use rnn to embed the query
    def lvm_lm(targets, is_decoding):

      target_len = len(targets)
      batch_size = tf.shape(targets[0])[0]

      # embedding parameters
      with tf.variable_scope("embed"):
        char_embedding = tf.contrib.framework.model_variable("char_embedding",
                                                             shape=[self.vocab_size, size],
                                                             dtype=tf.float32,
                                                             initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                             collections=tf.GraphKeys.WEIGHTS,
                                                             trainable=True)
        latent_size = 10000
        latent_embedding = tf.contrib.framework.model_variable("latent_embedding",
                                                             shape=[latent_size, size],
                                                             dtype=tf.float32,
                                                             initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                             collections=tf.GraphKeys.WEIGHTS,
                                                             trainable=True)

      # Create the internal multi-layer cell for our RNN.
      cell = model_utils.create_cell(size, num_layers, cell_type, is_training=is_training)

      with tf.variable_scope("encoder") as scope:
        # encode the target seq
        encoder_inputs = [tf.nn.embedding_lookup(char_embedding, i) for i in targets]
        encoder_inputs = tf.expand_dims(tf.pack(encoder_inputs, 1), 1)
        encoder_outputs = model_utils.cnn(encoder_inputs, size, size, 2, [1, 3], [1, 2], 
            is_training=is_training, scope="cnn")
        target_states = tf.squeeze(encoder_outputs, [1])
        target_embed = tf.reduce_max(target_states, 1)

      length = target_states.get_shape()[1].value

      # Actor post
      with tf.variable_scope("actor_post") as scope:
        values = tf.nn.relu(target_states)
        keys = model_utils.fully_connected(values, size, activation_fn=None, 
            is_training=is_training, scope="keys")
        memory = (keys, values)
        state = (tf.zeros([batch_size, size]), target_embed)
        inputs = tf.zeros([batch_size, size])
        actor_post_probs = []
        actor_post_samples = []
        actor_post_log_probs = []
        for _ in xrange(length):
          inputs = tf.stop_gradient(inputs)
          outputs, state = model_utils.attention_iter(inputs, state, memory, cell,
              not is_decoding)
          logits = tf.matmul(outputs, tf.stop_gradient(tf.transpose(latent_embedding)))
          actor_post_probs.append(tf.nn.softmax(logits))
          log_probs = tf.nn.log_softmax(logits)
          sampleids = tf.squeeze(tf.multinomial(logits, 1), [1])
          indices = tf.to_int64(tf.range(tf.shape(sampleids)[0])) * latent_size + sampleids
          actor_post_log_probs.append(tf.gather(tf.reshape(log_probs, [-1]), indices))
          samples = tf.nn.embedding_lookup(latent_embedding, sampleids)
          actor_post_samples.append(samples)
          inputs = tf.nn.relu(samples)
          scope.reuse_variables()

      # Actor prior
      with tf.variable_scope("actor_prior") as scope:
        memory = (tf.zeros([batch_size, 1, size]), tf.zeros([batch_size, 1, size]))
        state = (tf.zeros([batch_size, size]), tf.zeros([batch_size, size]))
        inputs = tf.zeros([batch_size, size])
        actor_prior_samples = []
        loss_prior = 0.0
        for i in xrange(length):
          inputs = tf.stop_gradient(inputs)
          outputs, state = model_utils.attention_iter(inputs, state, memory, cell,
              not is_decoding)
          logits = tf.matmul(outputs, tf.stop_gradient(tf.transpose(latent_embedding)))
          if self.is_decoding:
            sampleids = tf.squeeze(tf.multinomial(logits, 1), [1])
            samples = tf.nn.embedding_lookup(latent_embedding, sampleids)
            actor_prior_samples.append(samples)
            inputs = tf.nn.relu(samples)
          else:
            loss_prior += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,
                tf.stop_gradient(actor_post_probs[i])))
            inputs = tf.nn.relu(actor_post_samples[i])
          scope.reuse_variables()

      # Generator
      with tf.variable_scope("decoder") as scope:
        actor_samples = actor_prior_samples if self.is_decoding else actor_post_samples
        actor_samples = tf.pack(actor_samples, axis=1)
        values = tf.nn.relu(actor_samples)
        keys = model_utils.fully_connected(values, size, is_training=not is_decoding, 
            scope = "keys")
        memory = (keys, values)
        state = (tf.zeros([batch_size, size]), tf.zeros([batch_size, size]))
        if self.is_decoding:
          iter_fn = lambda inputs, state, memory: model_utils.attention_iter(inputs, state, memory, cell, 
              not is_decoding)
          outputs = model_utils.beam_dec_v2(target_len, state, memory, iter_fn, char_embedding, 
              beam_size=1000, topn=100)
          return outputs
        else:
          decoder_inputs = [tf.zeros([batch_size, size])]
          decoder_inputs += [tf.nn.relu(tf.nn.embedding_lookup(char_embedding, i)) for i in targets[:-1]]
          decoder_outputs = []
          for inputs in decoder_inputs:
            outputs, state = model_utils.attention_iter(inputs, state, memory, cell,
                not is_decoding)
            decoder_outputs.append(outputs)
            scope.reuse_variables()
          logits = tf.unpack(tf.reshape(tf.matmul(tf.concat(0, decoder_outputs), 
              tf.transpose(char_embedding)), [target_len, batch_size, self.vocab_size]))
          losses = tf.nn.seq2seq.sequence_loss_by_example(logits, targets,
              [tf.ones([batch_size])] * len(logits), True)
          loss_generator = tf.reduce_mean(losses)
          loss_post = tf.reduce_mean(-sum(actor_post_log_probs) * tf.stop_gradient(tf.exp(-losses)))
          loss_prior = loss_prior / float(length)
          return loss_prior + loss_post + loss_generator


    # The basic lm
    def reinforce_lm(targets, is_decoding):

      length = len(targets)
      batch_size = tf.shape(targets[0])[0]
      # embedding parameters
      with tf.variable_scope("embed"):
        char_embedding = tf.contrib.framework.model_variable("char_embedding",
                                                             shape=[self.vocab_size, size],
                                                             dtype=tf.float32,
                                                             initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                             collections=tf.GraphKeys.WEIGHTS,
                                                             trainable=True)
      # Create the internal multi-layer cell for our RNN.
      cell = model_utils.create_cell(size, num_layers, cell_type, is_training=is_training)

      with tf.variable_scope("decoder") as scope:
        memory = (tf.zeros([2*batch_size, 1, size]), tf.zeros([2*batch_size, 1, size]))
        state = tuple([tf.zeros([2*batch_size, size])] * (num_layers+1))
        inputs = tf.zeros([2*batch_size, size])
        samples = []
        log_probs = []
        loss_xe = 0.0
        for i in xrange(length):
          outputs, state = model_utils.attention_iter(inputs, state, memory, cell, 
              not is_decoding)
          logits = tf.matmul(outputs, tf.transpose(char_embedding))
          logits_target, logits_sample = tf.split(0, 2, logits)
          loss_xe += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_target, targets[i]))
          sampleids = tf.to_int32(tf.squeeze(tf.multinomial(logits_sample, 1), [1]))
          indices = tf.range(tf.shape(sampleids)[0]) * self.vocab_size + sampleids
          log_probs.append(tf.gather(tf.reshape(tf.nn.log_softmax(logits_sample), [-1]), indices))
          samples.append(sampleids)
          inputs = tf.nn.relu(tf.nn.embedding_lookup(char_embedding, tf.concat(0, [targets[i], sampleids])))
          scope.reuse_variables()

      with tf.variable_scope("critic") as scope:
        encoder_inputs = [tf.concat(0, [tf.nn.embedding_lookup(char_embedding, x) 
            for x in [targets[i], samples[i]]]) for i in xrange(length)]
        encoder_inputs = tf.stop_gradient(tf.expand_dims(tf.pack(encoder_inputs, 1), 1))
        encoder_outputs = model_utils.ResCNN(encoder_inputs, size, 2, [1, 3], [1, 2], 
            is_training=is_training, scope="ResCNN")
        encoder_states = tf.squeeze(encoder_outputs, [1])
        encoder_embeds = tf.reduce_max(encoder_states, 1)
        embed_proj = model_utils.ResDNN(tf.nn.relu(encoder_embeds), size, 2, is_training=not is_decoding, 
            scope="embed_proj")
        logits = tf.squeeze(model_utils.fully_connected(tf.nn.relu(embed_proj), 1, is_training=not is_decoding, 
            scope="logits"), [1])

      if self.is_decoding:
        return tf.pack(samples, axis=1)
      else:
        targets = tf.concat(0, [tf.ones([batch_size]), -tf.ones([batch_size])])
        loss_critic = 0.5 * tf.reduce_mean(tf.square(tf.tanh(logits) - targets))
        _, logits = tf.split(0, 2, logits)
        reward = tf.tanh(logits)
        loss_pg = tf.reduce_mean(-sum(log_probs) * tf.stop_gradient(reward))
        return loss_critic + loss_pg + loss_xe

    # The basic lm
    def basic_lm(targets, is_decoding):

      length = len(targets)
      batch_size = tf.shape(targets[0])[0]
      # embedding parameters
      with tf.variable_scope("embed"):
        char_embedding = tf.contrib.framework.model_variable("char_embedding",
                                                             shape=[self.vocab_size, size],
                                                             dtype=tf.float32,
                                                             initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                                             collections=tf.GraphKeys.WEIGHTS,
                                                             trainable=True)
      # Create the internal multi-layer cell for our RNN.
      cell = model_utils.create_cell(size, num_layers, cell_type, is_training=is_training)

      with tf.variable_scope("decoder") as scope:
        iter_fn = lambda inputs, state, memory: model_utils.attention_iter(inputs, state, memory, cell, 
            is_training)
        # output the target sequence
        if is_decoding:
          decoder = model_utils.stochastic_dec
          memory = (tf.zeros([batch_size, 1, size]), tf.zeros([batch_size, 1, size]))
          state = tuple([tf.zeros([batch_size, size])] * (num_layers+1))
          outputs = decoder(length, state, memory, iter_fn, 
              char_embedding, topn=1)
          return outputs
        # output the xent loss
        else:
          memory = (tf.zeros([batch_size, 1, size]), tf.zeros([batch_size, 1, size]))
          decoder_inputs = [tf.zeros([batch_size, size])]
          decoder_inputs += [tf.nn.relu(tf.nn.embedding_lookup(char_embedding, i)) for i in targets[:-1]]
          decoder_outputs = []
          state = tuple([tf.zeros([batch_size, size])] * (num_layers+1))
          for inputs in decoder_inputs:
            outputs, state = iter_fn(inputs, state, memory)
            decoder_outputs.append(outputs)
            scope.reuse_variables()
          logits = tf.matmul(tf.concat(0, decoder_outputs), tf.transpose(char_embedding))
          logits = tf.unpack(tf.reshape(logits, [length, -1, self.vocab_size]))
          loss_xe = tf.nn.seq2seq.sequence_loss(logits, targets, [tf.ones([batch_size])]*length, False, True)
          return loss_xe


    model = reinforce_lm
    # build the buckets
    self.outputs, self.losses, self.updates = [], [], []
    for i in xrange(len(buckets)):
      inputs_len = buckets[i]
      if self.is_decoding:
        inputs = [tf.zeros([batch_size], dtype=tf.int32)] * inputs_len
        outputs = model(inputs, self.is_decoding)
        self.outputs.append(outputs)
      else:
        batch_seq = get_batch(i)
        inputs = tf.unpack(batch_seq, axis=1)
        outputs = model(inputs, self.is_decoding)
        self.losses.append(outputs)
        model_utils.params_decay(1.0 - self.learning_rate)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.learning_rate.assign(
            tf.maximum(self.learning_rate*learning_rate_decay_factor, 1e-5)))
        self.updates.append(tf.contrib.layers.optimize_loss(self.losses[i], self.global_step, 
            tf.identity(self.learning_rate), 'Adam', gradient_noise_scale=None, clip_gradients=None, 
            moving_average_decay=None, name="OptimizeLoss"))
      tf.get_variable_scope().reuse_variables()
    if not self.is_decoding:
      with tf.variable_scope("OptimizeLoss"):
        self.learning_rate = tf.get_variable("learning_rate")

    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session,
           bucket_id, update=False, do_profiling=False):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      targets: target sequence
      bucket_id: which bucket of the model to use.
      update: whether to do the update or not.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """

    # Output feed.
    if self.is_decoding:
      output_feed = [self.outputs[bucket_id]]
    elif update:
      output_feed = [self.losses[bucket_id],  # Update Op that does SGD.
                     self.updates[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.

    if do_profiling:
      self.run_metadata = tf.RunMetadata()
      outputs = session.run(output_feed,  
          options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=self.run_metadata)
      trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
      trace_file = open('timeline.ctf.json', 'w')
      trace_file.write(trace.generate_chrome_trace_format())
      trace_file.close()
    else:
      outputs = session.run(output_feed)

    return outputs[0]

