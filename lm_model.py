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
                 segs=None,
                 pos_labels=None,
                 num_pos_tags=30,
                 block_type="transformer2",
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
        self.num_pos_tags = num_pos_tags

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
                            tf.less(tf.random_uniform(tf.shape(padded_seqs)), 0.95),
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
                return loss_ce+loss_mse, encodes, sample_seqs, None

            """ lm model """
            def multi_lm(seqs, segs, pos_labels):

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
                    output_embedding = model_utils.GLU(
                        input_embedding,
                        size,
                        is_training=self.training,
                        scope="embed_proj")
                    pos_embedding = tf.get_variable(
                        "pos_embedding",
                        shape=[self.num_pos_tags, size],
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal(0.0, 0.01),
                        trainable=trainable,
                        collections=collections)
                    transition_params = tf.get_variable(
                        "transition_params",
                        shape=[self.num_pos_tags, self.num_pos_tags],
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal(0.0, 0.01),
                        trainable=trainable,
                        collections=collections)
                   # if dropout > 0.0:
                   #     noise_seqs = tf.where(
                   #         tf.less(tf.random_uniform(tf.shape(padded_seqs)), 0.95),
                   #         tf.zeros(tf.shape(padded_seqs), dtype=tf.int32),
                   #         tf.random_uniform(tf.shape(padded_seqs), maxval=self.vocab_size, dtype=tf.int32))
                   #     corrupt_seqs = tf.where(
                   #         tf.less(tf.random_uniform(tf.shape(padded_seqs)), dropout),
                   #         noise_seqs,
                   #         padded_seqs)
                   # else:
                   #     corrupt_seqs = padded_seqs
                   # corrupt_inputs = tf.nn.embedding_lookup(input_embedding, corrupt_seqs)


                with tf.variable_scope("seg"):
                    lstm = model_utils.cudnn_lstm(
                        num_layers=1,
                        num_units=size,
                        direction="bidirectional",
                        input_shape=tf.TensorShape([None, None, self.vocab_dim]),
                        trainable=trainable)

                    padded_seqs = tf.pad(seqs, [[0,0], [1,1]])
                    masks = tf.greater(padded_seqs, 0)
                    seg_masks = tf.reduce_any(tf.stack([masks[:,:-1], masks[:,1:]], axis=-1), axis=-1)
                    inputs = tf.nn.embedding_lookup(input_embedding, padded_seqs)
                    inputs = tf.transpose(inputs, [1, 0, 2])
                    outputs, states = lstm(inputs, training=trainable)
                    padded_char_encodes = tf.transpose(outputs, [1, 0, 2])
                    padded_char_encodes *= tf.expand_dims(tf.to_float(masks), axis=-1)
                    char_encodes = padded_char_encodes[:,1:-1]
                    adjacent_encodes = tf.concat([padded_char_encodes[:,:-1], padded_char_encodes[:,1:]], axis=-1)
                    seg_logits = model_utils.MLP(
                        adjacent_encodes,
                        2,
                        size,
                        1,
                        is_training=self.training,
                        scope="seg_logits")
                    seg_logits = tf.squeeze(seg_logits, -1)

                    weights = tf.to_float(seg_masks)
                    if segs != None:
                        seg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=segs,
                            logits=seg_logits)
                        seg_loss = tf.reduce_sum(seg_loss*weights) / tf.reduce_sum(weights)
                        segmented_seqs_gold = model_utils.slice_words(
                            seqs, segs[:,1:-1])
                    else:
                        seg_loss = 0.0
                        segmented_seqs_gold = None
                    seg_predicts = tf.cast(tf.greater(seg_logits, 0.0), tf.float32)*weights
                    segmented_seqs_hyp = model_utils.slice_words(
                        seqs, seg_predicts[:,1:-1])


                def embed_words(segmented_seqs, reuse=None):

                    batch_size = tf.shape(segmented_seqs)[0]
                    max_word_length = tf.shape(segmented_seqs)[1]
                    max_char_length = tf.shape(segmented_seqs)[2]
                    masks = tf.greater(segmented_seqs, 0)

                    with tf.variable_scope("word_embedder", reuse=reuse):
                        inputs = tf.nn.embedding_lookup(input_embedding, segmented_seqs)
                        cnn_outputs = model_utils.convolution2d(
                            inputs,
                            [size]*4,
                            [[1,1],[1,2],[1,3],[1,4]],
                            activation_fn=tf.nn.relu,
                            is_training=self.training,
                            scope="convs")
                        cnn_outputs = tf.reduce_max(cnn_outputs, axis=2)
                        word_embeds = model_utils.fully_connected(
                            cnn_outputs,
                            self.vocab_dim,
                            is_training=self.training,
                            scope="projs")
                        word_embeds = tf.contrib.layers.layer_norm(word_embeds, begin_norm_axis=-1)

                        weights = tf.to_float(tf.reduce_any(masks, axis=2, keepdims=True))
                        word_embeds *= weights
                    return word_embeds

                lstm_encode = model_utils.cudnn_lstm(
                    num_layers=1,
                    num_units=size,
                    direction="bidirectional",
                    input_shape=tf.TensorShape([None,None,self.vocab_dim]),
                    trainable=trainable)

                def encode_words(segmented_seqs):

                    batch_size = tf.shape(segmented_seqs)[0]
                    max_length = tf.shape(segmented_seqs)[1]
                    masks = tf.reduce_any(tf.greater(segmented_seqs, 0), axis=-1)
                    padded_seqs = tf.pad(segmented_seqs, [[0,0],[1,1],[0,0]])
                    padded_word_embeds = embed_words(padded_seqs)
                    word_embeds = padded_word_embeds[:,1:-1]
                    if block_type == "lstm":
                        inputs = tf.transpose(padded_word_embeds, [1, 0, 2])
                        outputs, states = lstm_encode(inputs, training=trainable)
                        outputs_fw, outputs_bw = tf.split(outputs, 2, axis=2)
                        contexts = tf.concat([outputs_fw[:-2], outputs_bw[2:]], axis=-1)
                        contexts = tf.transpose(contexts, [1,0,2])
                        contexts = model_utils.fully_connected(
                            contexts,
                            self.vocab_dim,
                            is_training=self.training,
                            scope="context_projs")
                    elif block_type == "transformer":
                        inputs = padded_word_embeds[:,:-2]
                        outputs_fw = model_utils.transformer(
                            inputs,
                            3,
                            is_training=self.training,
                            scope="contexts_fw")
                        inputs = tf.reverse(padded_word_embeds[:,2:], axis=[1])
                        outputs_bw = model_utils.transformer(
                            inputs,
                            3,
                            is_training=self.training,
                            scope="contexts_bw")
                        outputs_bw = tf.reverse(outputs_bw, axis=[1])
                        contexts = outputs_fw + outputs_bw
                    elif block_type == "transformer2":
                        attn_masks = tf.tile(tf.expand_dims(masks, 1), [1,max_length,1])
                        attn_masks = tf.logical_and(attn_masks, tf.expand_dims(masks, -1))
                        contexts = model_utils.transformer2(
                            word_embeds,
                            2,
                            attn_masks,
                            is_training=self.training,
                            scope="contexts")
                    word_encodes = tf.concat([contexts, word_embeds], axis=-1)
                    return word_embeds, contexts, word_encodes

                lstm_decode = model_utils.cudnn_lstm(
                    num_layers=1,
                    num_units=size,
                    direction="unidirectional",
                    input_shape=tf.TensorShape([None,None,2*self.vocab_dim]),
                    trainable=trainable)

                def logit_fn(outputs):
                    outputs = model_utils.GLU(
                        outputs,
                        size,
                        is_training=self.training,
                        scope="output_projs")
                    logits = tf.matmul(outputs, output_embedding, transpose_b=True)
                    return logits

                def get_char_loss(contexts, segmented_seqs, reuse=None):
                    with tf.variable_scope("generator", reuse=reuse):
                        batch_size = tf.shape(segmented_seqs)[0]
                        max_word_seq_length = tf.shape(segmented_seqs)[1]
                        max_char_seq_length = tf.shape(segmented_seqs)[2]+1
                        masks = tf.greater(segmented_seqs, 0)
                        weights = tf.pad(tf.to_float(masks), [[0,0],[0,0],[1,0]], constant_values=1.0)
                        masks = tf.pad(tf.reduce_any(masks, axis=-1), [[0,0],[1,0]])
                        masks = tf.reduce_any(tf.stack([masks[:,:-1], masks[:,1:]], axis=-1), axis=-1)
                        weights *= tf.expand_dims(tf.to_float(masks), axis=-1)
                        segmented_seqs = tf.pad(segmented_seqs, [[0,0],[0,0],[1,1]])
                        ids = tf.reshape(
                            segmented_seqs[:,:,:-1], [batch_size*max_word_seq_length, max_char_seq_length])
                        inputs = tf.nn.embedding_lookup(input_embedding, ids)
                        contexts = tf.reshape(contexts, [batch_size*max_word_seq_length, 1, self.vocab_dim])
                        contexts = tf.tile(contexts, [1, max_char_seq_length, 1])
                        inputs = tf.concat([inputs, contexts], axis=-1)
                        inputs = tf.transpose(inputs, [1,0,2])
                        outputs, _ = lstm_decode(inputs, training=trainable)
                        outputs = tf.transpose(outputs, [1,0,2])
                        outputs = tf.reshape(outputs, [batch_size*max_word_seq_length*max_char_seq_length, size])
                        logits = logit_fn(outputs)
                        logits = tf.reshape(
                            logits, [batch_size, max_word_seq_length, max_char_seq_length, self.vocab_size])
                        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=segmented_seqs[:,:,1:], logits=logits)
                        losses *= weights
                        loss = tf.reduce_sum(losses) / tf.reduce_sum(weights)
                    return loss

                def make_lstm_cell(cudnn_lstm, contexts):
                    def lstm_cell(inputs, state):
                        tile_num = tf.stack([tf.shape(inputs)[0] / tf.shape(contexts)[0], 1], axis=0)
                        tiled_contexts = tf.tile(contexts, tf.to_int32(tile_num))
                        inputs = tf.concat([inputs, tiled_contexts], axis=-1)
                        inputs = tf.expand_dims(inputs, 0)
                        state = tuple(map(lambda i: tf.expand_dims(i, 0), state))
                        outputs, state = cudnn_lstm(inputs, state)
                        outputs = tf.squeeze(outputs, 0)
                        state = tuple(map(lambda i: tf.squeeze(i, 0), state))
                        return outputs, state
                    return lstm_cell

                def decode(word_embeds, reuse=None):
                    with tf.variable_scope("generator", reuse=reuse):
                        batch_size = tf.shape(word_embeds)[0]
                        max_word_seq_length = tf.shape(word_embeds)[1]
                        contexts = tf.reshape(word_embeds, [batch_size*max_word_seq_length, self.vocab_dim])
                        lstm_cell = make_lstm_cell(lstm_decode, contexts)
                        initial_state = tf.zeros([batch_size*max_word_seq_length, size])
                        initial_state = (initial_state, initial_state)
                        sample_seqs, sample_scores = model_utils.stochastic_dec(
                            5,
                            initial_state,
                            input_embedding,
                            lstm_cell,
                            logit_fn)
                        sample_seqs = tf.reshape(sample_seqs, [batch_size, max_word_seq_length, 5])
                        sample_scores = tf.reshape(sample_scores, [batch_size, max_word_seq_length])
                    return sample_seqs, sample_scores

                def sample(segmented_seqs):
                    batch_size = tf.shape(segmented_seqs)[0]
                    for i in range(1):
                        max_char_length = tf.shape(segmented_seqs)[2]
                        segmented_seqs = tf.pad(segmented_seqs, [[0,0],[0,1],[0,tf.nn.relu(5-max_char_length)]])
                        masks = tf.reduce_any(tf.greater(segmented_seqs, 0), axis=-1)
                        masks = tf.cast(tf.cumprod(tf.to_int32(masks), axis=1), tf.bool)
                        masks = tf.pad(masks, [[0,0],[1,0]], constant_values=True)
                        masks = masks[:,:-1]
                        seq_lengths = tf.reduce_sum(tf.to_int32(masks), axis=1)
                        max_seq_length = tf.reduce_max(seq_lengths)
                        segmented_seqs = segmented_seqs[:,:max_seq_length]
                        masks = masks[:,:max_seq_length]
                        max_char_length = tf.shape(segmented_seqs)[2]
                        word_embeds, contexts, word_encodes = encode_words(segmented_seqs)
                        sample_seqs, sample_scores = decode(contexts)
                        sample_seqs = tf.pad(sample_seqs, [[0,0],[0,0],[0,max_char_length-5]])
                        sample_scores -= 1e6 * tf.to_float(tf.logical_not(masks))
                        choices = tf.squeeze(tf.multinomial(sample_scores, 1), axis=-1)
                        masks = tf.one_hot(choices, max_seq_length)
                        masks = tf.cast(masks, tf.bool)
                        segmented_seqs = tf.where(
                            tf.reshape(masks, [batch_size*max_seq_length]),
                            tf.reshape(sample_seqs, [batch_size*max_seq_length, -1]),
                            tf.reshape(segmented_seqs, [batch_size*max_seq_length, -1]))
                        segmented_seqs = tf.reshape(segmented_seqs, [batch_size, max_seq_length, -1])
                        seqs = model_utils.stitch_chars(segmented_seqs)
                    return seqs

                def get_pos_logits(word_encodes):
                    batch_size = tf.shape(word_encodes)[0]
                    max_length = tf.shape(word_encodes)[1]
                    pos_outputs = model_utils.GLU(
                        word_encodes,
                        size,
                        is_training=self.training,
                        scope="pos_outputs")
                    pos_logits = tf.matmul(
                        tf.reshape(pos_outputs, [batch_size*max_length, size]),
                        pos_embedding,
                        transpose_b=True)
                    pos_logits = tf.reshape(pos_logits, [batch_size, max_length, self.num_pos_tags])
                    return pos_logits

                segmented_seqs_hyp = tf.pad(segmented_seqs_hyp, [[0,0],[0,1],[0,0]])
                word_embeds_hyp, contexts_hyp, word_encodes_hyp = encode_words(
                    segmented_seqs_hyp)
                lm_hyp_loss = get_char_loss(contexts_hyp, segmented_seqs_hyp)
                segmented_seq_masks_hyp = tf.reduce_any(tf.greater(segmented_seqs_hyp, 0), axis=-1)
                segmented_seq_lengths_hyp = tf.reduce_sum(tf.to_int32(segmented_seq_masks_hyp), axis=-1)
                pos_logits_hyp = get_pos_logits(word_encodes_hyp)
                viterbi_tags, viterbi_scores = tf.contrib.crf.crf_decode(
                    pos_logits_hyp, transition_params, segmented_seq_lengths_hyp)

                tf.get_variable_scope().reuse_variables()
                sample_seqs = sample(segmented_seqs_hyp)

                lm_gold_loss = 0.0
                pos_loss = 0.0
                if segmented_seqs_gold != None:                        
                    segmented_seqs_gold = tf.pad(segmented_seqs_gold, [[0,0],[0,1],[0,0]])
                    word_embeds_gold, contexts_gold, word_encodes_gold = encode_words(
                        segmented_seqs_gold)
                    lm_gold_loss = get_char_loss(contexts_gold, segmented_seqs_gold)
                    if pos_labels != None:
                        segmented_seq_masks_gold = tf.reduce_any(tf.greater(segmented_seqs_gold, 0), axis=-1)
                        segmented_seq_lengths_gold = tf.reduce_sum(tf.to_int32(segmented_seq_masks_gold), axis=-1)
                        pos_logits_gold = get_pos_logits(word_encodes_gold[:,:-1])
                        log_probs, _ = tf.contrib.crf.crf_log_likelihood(
                            pos_logits_gold, pos_labels, segmented_seq_lengths_gold,
                            transition_params=transition_params)
                        pos_loss = tf.reduce_mean(-log_probs)


                sup_loss = seg_loss+lm_gold_loss+pos_loss
                unsup_loss = lm_hyp_loss
                return sup_loss, unsup_loss, word_encodes_hyp, seg_predicts, viterbi_tags, sample_seqs


            model = multi_lm

            with tf.variable_scope("model"):
                self.sup_loss, self.unsup_loss, self.encodes, self.segs, self.pos_labels, self.sample_seqs = model(
                    self.seqs, segs, pos_labels)
            if trainable:
                self.trainable_variables = sc.trainable_variables()
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate)
                self.update = model_utils.optimize_loss(
                    self.sup_loss if type(self.sup_loss) != float else self.unsup_loss,
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
        output_feed = [self.sup_loss,  # Update Op that does SGD.
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
        output_feed = [self.sup_loss]  # Update Op that does SGD.

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

    def tag_one_step(self, session, input_feed, do_profiling=False):
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
        output_feed = [self.segs, self.pos_labels]  # Update Op that does SGD.

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

        return outputs
