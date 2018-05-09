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

from utils import data_utils, model_utils


class LM_Model(object):
    """Sequence-to-class model with multiple buckets.

       implements multiple classifiers
    """

    def __init__(self,
                 session,
                 model_dir,
                 seqs,
                 learning_rate, global_step, training,
                 vocab_size, vocab_dim,
                 size, num_layers, max_gradient,
                 cell_type="LSTM",
                 scope="lm"):
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
        self.seqs = seqs
        self.vocab_size = vocab_size
        self.vocab_dim = vocab_dim
        self.learning_rate = learning_rate
        self.global_step = global_step
        self.training = training


        def bi_lm(seqs):

            batch_size = tf.shape(seqs)[0]
            seq_length = tf.shape(seqs)[1]

            with tf.variable_scope("embed"):
                char_embedding = tf.get_variable(
                    "char_embedding",
                    shape=[self.vocab_size, self.vocab_dim],
                    dtype=tf.float32,
                    initializer=tf.initializers.truncated_normal(0.0, 0.01),
                    trainable=True,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.WEIGHTS])
                input_embedding = tf.concat([tf.zeros([1, self.vocab_dim]), char_embedding[1:]], axis=0)
                output_embedding = char_embedding
                char_embeds = tf.nn.embedding_lookup(input_embedding, seqs)
                padded_seqs = tf.pad(seqs, [[0,0], [2,2]])
                inputs = tf.nn.embedding_lookup(input_embedding, padded_seqs)

            with tf.variable_scope("bilstm"):
                fw_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                    num_layers=2,
                    num_units=size,
                    direction="unidirectional")
                bw_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                    num_layers=2,
                    num_units=size,
                    direction="unidirectional")
                for var in fw_lstm.trainable_variables+bw_lstm.trainable_variables:
                    tf.add_to_collection(
                        tf.GraphKeys.WEIGHTS,
                        var)
                inputs = tf.transpose(inputs, [1, 0, 2])
                outputs_fw, states_fw = fw_lstm(inputs, training=(self.training != False))
                outputs_bw, states_bw = bw_lstm(tf.reverse(inputs, [0]), training=(self.training != False))
                outputs_bw = tf.reverse(outputs_bw, [0])
                lstm_embeds = tf.transpose(
                    tf.concat([outputs_fw[2:-2], outputs_bw[2:-2]], axis=-1), [1, 0, 2])
                outputs = tf.concat([outputs_fw[:-2], outputs_bw[2:]], axis=-1)
                outputs_proj = model_utils.fully_connected(
                    outputs,
                    self.vocab_dim,
                    is_training=self.training,
                    scope="output_proj")
                logits = tf.matmul(
                    tf.reshape(outputs_proj, [(seq_length+2)*batch_size, self.vocab_dim]),
                    output_embedding,
                    transpose_b=True)
                logits = tf.reshape(logits, [seq_length+2, batch_size, self.vocab_size])
                logits_valid = tf.reshape(tf.transpose(logits[1:-1], [1, 0, 2]), [-1, self.vocab_size])
                sample_seqs = tf.multinomial(logits_valid, 1, output_dtype=tf.int32)
                sample_seqs = tf.reshape(sample_seqs, [batch_size, seq_length])
                logits = tf.transpose(logits, [1, 0, 2])
                labels = tf.pad(seqs, [[0,0], [1,1]])
                weights = tf.pad(tf.to_float(tf.not_equal(seqs, 0)), [[0,0], [2,0]], constant_values=1.0)
                loss = tf.losses.sparse_softmax_cross_entropy(
                    labels,
                    logits,
                    weights=weights)
            return loss, char_embeds, lstm_embeds, sample_seqs

        model = bi_lm

        with tf.variable_scope(scope):
            self.losses, self.char_embeds, self.lstm_embeds, self.sample_seqs = model(self.seqs)
            if self.training != False:
                model_utils.params_decay(1.0 - self.learning_rate)
                self.updates = tf.contrib.layers.optimize_loss(self.losses, self.global_step, 
                    tf.identity(self.learning_rate), 'Adam', gradient_noise_scale=None, clip_gradients=None, 
                    name="OptimizeLoss")

        self.saver = tf.train.Saver(tf.global_variables(scope=scope))

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path+'.meta'):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
            try:
                global_step = int(ckpt.model_checkpoint_path.split('-')[-1])
                session.run(self.global_step.assign(global_step))
            except:
                pass
        else:
            print("Created model with fresh parameters. ")
            session.run(tf.variables_initializer(tf.global_variables(scope=scope)))

    def train_one_step(self, session, input_feed, do_profiling=False):
        """Run a step of the model feeding the given inputs.

        Args:
            session: tensorflow session to use.
            seqs: input sequence

        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.

        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
        """

        # Input feed.
        if type(self.training) != bool:
            input_feed[self.training] = True

        # Output feed.
        output_feed = [self.losses,  # Update Op that does SGD.
                       self.updates]  # Loss for this batch.

        if do_profiling:
          self.run_metadata = tf.RunMetadata()
          outputs = session.run(output_feed, input_feed, 
              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=self.run_metadata)
          trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
          trace_file = open('timeline.ctf.json', 'w')
          trace_file.write(trace.generate_chrome_trace_format())
          trace_file.close()
        else:
          outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def valid_one_step(self, session, input_feed, do_profiling=False):
        """Run a step of the model feeding the given inputs.

        Args:
            session: tensorflow session to use.
            seqs: input sequence

        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.

        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
        """

        # Input feed.
        if type(self.training) != bool:
            input_feed[self.training] = False

        # Output feed.
        output_feed = [self.losses]  # Update Op that does SGD.

        if do_profiling:
          self.run_metadata = tf.RunMetadata()
          outputs = session.run(output_feed, input_feed, 
              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=self.run_metadata)
          trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
          trace_file = open('timeline.ctf.json', 'w')
          trace_file.write(trace.generate_chrome_trace_format())
          trace_file.close()
        else:
          outputs = session.run(output_feed, input_feed)

        return outputs[0]

    def sample_one_step(self, session, input_feed, do_profiling=False):
        """Run a step of the model feeding the given inputs.

        Args:
            session: tensorflow session to use.
            seqs: input sequence

        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.

        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
        """

        # Input feed.
        if type(self.training) != bool:
            input_feed[self.training] = False

        # Output feed.
        output_feed = [self.sample_seqs]  # Update Op that does SGD.

        if do_profiling:
          self.run_metadata = tf.RunMetadata()
          outputs = session.run(output_feed, input_feed, 
              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=self.run_metadata)
          trace = timeline.Timeline(step_stats=self.run_metadata.step_stats)
          trace_file = open('timeline.ctf.json', 'w')
          trace_file.write(trace.generate_chrome_trace_format())
          trace_file.close()
        else:
          outputs = session.run(output_feed, input_feed)

        return outputs[0]
