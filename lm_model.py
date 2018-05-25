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
                 seqs,
                 trainable,
                 vocab_size, vocab_dim,
                 size, num_layers,
                 embedding_init=None,
                 dropout=0.0,
                 learning_rate=0.001, clr_period=10000,
                 reuse=None,
                 scope="lm"):
        """Create the model.

        Args:
            session: tf session.
            model_dir: directory to load and save the model.
            seqs: input sequences.
            trainable: if the model is trainable.
            vocab_size: size of the vocabulary.
            vocab_dim: dimension of the vocabulary.
            size: number of units in each layer of the model.
            num_layers: number of layers.
            embedding_init: the pretrained embedding to load.
            dropout: dropout rate.
            learning_rate: learning rate to start with.
            clr_period: period of cyclic learning rate.
            scope: scope name of the model.
        """
        self.seqs = seqs
        self.vocab_size = vocab_size
        self.vocab_dim = vocab_dim

        with tf.variable_scope(scope, reuse=reuse) as sc:

            """ Training flag """
            collections = [tf.GraphKeys.GLOBAL_VARIABLES]
            self.training = False
            if trainable:
                collections.append(tf.GraphKeys.WEIGHTS)
                self.training = tf.get_variable(
                    "training",
                    shape=[],
                    dtype=tf.bool,
                    initializer=tf.initializers.constant(True),
                    trainable=False)

            """ Global step counter """
            self.global_step = tf.get_variable(
                "global_step",
                shape=[],
                dtype=tf.int32,
                initializer=tf.initializers.constant(0),
                trainable=False)

            """ Learning rate schedule """
            cycle = tf.floor(1.0 + tf.to_float(self.global_step) / clr_period)
            learning_rate = (
                1e-1*learning_rate + \
                (learning_rate - 1e-1*learning_rate) * \
                tf.maximum((
                    1.0-tf.abs(tf.to_float(self.global_step) / tf.to_float(clr_period // 2)
                        - 2.0*cycle + 1.0)), 0))
            self.learning_rate = learning_rate

            """ lm model """
            def bi_lm(seqs):

                batch_size = tf.shape(seqs)[0]
                seq_length = tf.shape(seqs)[1]

                with tf.variable_scope("embed"):
                    if type(embedding_init) == np.ndarray:
                        initializer = tf.initializers.constant(embedding_init)
                    else:
                        initializer = tf.initializers.truncated_normal(0.0, 0.01)
                    char_embedding = tf.get_variable(
                        "char_embedding",
                        shape=[self.vocab_size, self.vocab_dim],
                        dtype=tf.float32,
                        initializer=initializer,
                        trainable=trainable,
                        collections=collections)
                    input_embedding = tf.concat([tf.zeros([1, self.vocab_dim]), char_embedding[1:]], axis=0)
                    output_embedding = char_embedding
                    padded_seqs = tf.pad(seqs, [[0,0], [2,2]])
                    if dropout > 0.0:
                        noise_seqs = tf.where(
                            tf.less(tf.random_uniform(tf.shape(padded_seqs)), 0.5),
                            tf.zeros(tf.shape(padded_seqs), dtype=tf.int32),
                            tf.random_uniform(tf.shape(padded_seqs), maxval=self.vocab_size, dtype=tf.int32))
                        corrupt_seqs = tf.where(
                            tf.less(tf.random_uniform(tf.shape(padded_seqs)), dropout),
                            noise_seqs,
                            padded_seqs)
                    else:
                        corrupt_seqs = padded_seqs
                    inputs = tf.nn.embedding_lookup(input_embedding, padded_seqs)
                    corrupt_inputs = tf.nn.embedding_lookup(input_embedding, corrupt_seqs)

                with tf.variable_scope("encode"):
                    lstms = []
                    for i in range(num_layers):
                        with tf.variable_scope("layer_"+str(i)):
                            lstms.append(model_utils.cudnn_lstm(
                                num_layers=1,
                                num_units=size,
                                direction="bidirectional",
                                input_shape=tf.TensorShape([None, None, self.vocab_dim+i*32]),
                                trainable=trainable))
                    def encode(inputs):
                        inputs = tf.transpose(inputs, [1, 0, 2])
                        for i in range(num_layers):
                            outputs, states = lstms[i](inputs, training=trainable)
                            glu_feats = model_utils.GLU(
                                outputs,
                                inputs.get_shape()[-1].value+32,
                                is_training=self.training,
                                scope="glu_feats_"+str(i))
                            switchs = model_utils.fully_connected(
                                outputs,
                                inputs.get_shape()[-1].value+32,
                                activation_fn=tf.sigmoid,
                                is_training=self.training,
                                scope="switchs_"+str(i))
                            inputs = (1.0 - switchs) * glu_feats + \
                                switchs * tf.concat([inputs, tf.zeros([seq_length+4, batch_size, 32])], axis=-1)
                        encodes = tf.transpose(inputs, [1, 0, 2])
                        return encodes
                    encoder_outputs = encode(inputs)
                    encodes = encoder_outputs[:,2:-2]
                    tf.get_variable_scope().reuse_variables()
                    corrupt_encoder_outputs = encode(corrupt_inputs)

                with tf.variable_scope("lm"):
                    inputs = tf.transpose(corrupt_encoder_outputs, [1, 0, 2])
                    lstm = model_utils.cudnn_lstm(
                        num_layers=1,
                        num_units=size,
                        direction="bidirectional",
                        input_shape=tf.TensorShape([None,None,inputs.get_shape()[-1].value]),
                        trainable=trainable)
                    outputs, states = lstm(inputs, training=trainable)
                    outputs_fw, outputs_bw = tf.split(outputs, 2, axis=2)
                    outputs = tf.concat([outputs_fw[:-2], outputs_bw[2:]], axis=-1)
                    outputs = model_utils.GLU(
                        outputs,
                        inputs.get_shape()[-1].value,
                        is_training=self.training,
                        scope="outputs_pred")
                    outputs = tf.transpose(outputs, [1, 0, 2])
                    weights = tf.pad(tf.to_float(tf.not_equal(seqs, 0)), [[0,0], [2,0]], constant_values=1.0)
                    targets = encoder_outputs[:,1:-1]
                    loss_mse = 0.5 * tf.reduce_sum(
                        weights*tf.reduce_sum(tf.square(outputs - targets), axis=-1)) / \
                        tf.reduce_sum(weights)
                    outputs = model_utils.GLU(
                        outputs,
                        self.vocab_dim,
                        is_training=self.training,
                        scope="outputs_vocab")
                    logits = tf.matmul(
                        tf.reshape(outputs, [batch_size*(seq_length+2), self.vocab_dim]),
                        output_embedding,
                        transpose_b=True)
                    logits = tf.reshape(logits, [batch_size, seq_length+2, self.vocab_size])
                    logits_valid = tf.reshape(
                        tf.nn.log_softmax(logits[:,1:-1]),
                        [batch_size, seq_length*self.vocab_size])
                    samples = tf.multinomial(logits_valid, 1, output_dtype=tf.int32)
                    sample_posits = tf.one_hot(tf.squeeze(samples // self.vocab_size, 1), seq_length,
                        on_value=True, off_value=False, dtype=tf.bool)
                    sample_ids = tf.tile(samples % self.vocab_size, [1, seq_length])
                    sample_seqs = tf.where(sample_posits, sample_ids, seqs)
                    labels = tf.pad(seqs, [[0,0], [1,1]])
                    loss_ce = tf.losses.sparse_softmax_cross_entropy(
                        labels,
                        logits,
                        weights=weights)
                return loss_ce+loss_mse, encodes, sample_seqs

            model = bi_lm

            self.loss, self.encodes, self.sample_seqs = model(
                self.seqs)
            if trainable:
                self.trainable_variables = sc.trainable_variables()
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate)
                self.update = model_utils.optimize_loss(
                    self.loss,
                    self.global_step,
                    self.optimizer,
                    self.trainable_variables)

            self.global_variables = sc.global_variables()
            self.saver = tf.train.Saver(self.global_variables)

    def init(self, session, model_dir=''):
        """Initialize the model graph.

        """

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path+'.meta'):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters. ")
            session.run(tf.variables_initializer(self.global_variables))

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
        output_feed = [self.loss,  # Update Op that does SGD.
                       self.update]  # Loss for this batch.

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
        output_feed = [self.loss]  # Update Op that does SGD.

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
