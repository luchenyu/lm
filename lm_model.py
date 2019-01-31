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

MAX_CHAR_LEN = 10

class LM_Model(object):
    """Sequence-to-class model with multiple buckets.

       implements multiple classifiers
    """

    def __init__(self,
                 seqs,
                 trainable,
                 vocab_size, vocab_dim,
                 size, num_layers,
                 word_ids_array=None,
                 word_count_array=None,
                 segs=None,
                 pos_labels=None,
                 num_pos_tags=30,
                 block_type="transformer2",
                 decoder_type="attn",
                 loss_type="sup",
                 model="ultra_lm",
                 embedding_init=None,
                 dropout=0.0,
                 learning_rate=0.0001, clr_period=100000,
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
        if segs == None:
            segs = [None] * len(seqs)
        self.pieces = zip(seqs, segs)
        self.word_ids_array = tf.constant(word_ids_array, dtype=tf.int32)
        self.word_count_array = word_count_array
        self.vocab_size = vocab_size
        self.vocab_dim = vocab_dim
        self.num_pos_tags = num_pos_tags
        self.size = size
        self.num_layers = num_layers
        self.dropout = dropout


        with tf.variable_scope(scope, reuse=reuse) as self.scope:

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
            ratio = 0.3
            learning_rate = tf.get_variable(
                "learning_rate",
                dtype=tf.float32,
                initializer=learning_rate,
                trainable=False)
            self.max_learning_rate = learning_rate
            clr_period = tf.get_variable(
                "clr_period",
                dtype=tf.float32,
                initializer=float(clr_period),
                trainable=False)
            self.clr_period = clr_period
            x = (tf.to_float(self.global_step) % clr_period) / clr_period
            learning_rate = tf.cond(
                tf.less(x, ratio),
                lambda: 1e-1*learning_rate + (learning_rate - 1e-1*learning_rate)*(x/ratio),
                lambda: learning_rate + (1e-1*learning_rate - learning_rate)*(x-ratio)/(1.0-ratio))
            self.learning_rate = learning_rate

            """ lm model """
            def simple_lm(seqs, segs, pos_labels):

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
                    self.input_embedding = input_embedding
                    output_embedding = model_utils.fully_connected(
                        input_embedding,
                        size,
                        is_training=self.training,
                        scope="embed_proj")
                    output_embedding = tf.contrib.layers.layer_norm(
                        output_embedding, begin_norm_axis=-1,
                        variables_collections=collections, trainable=trainable)
                    self.output_embedding = output_embedding

                lstm_decoder = model_utils.CudnnLSTMCell(
                    num_layers=2,
                    num_units=size,
                    direction="unidirectional",
                    is_training=self.training)
                bilstm_decoder = model_utils.CudnnLSTMCell(
                    num_layers=2,
                    num_units=size,
                    direction="bidirectional",
                    is_training=self.training)
                attn_decoder = model_utils.AttentionCell(
                    size,
                    num_layer=2,
                    dropout=1.0-dropout,
                    is_training=self.training)
                def logit_fn(outputs):
                    logits = tf.matmul(outputs, output_embedding, transpose_b=True)
                    return logits

                with tf.variable_scope("generator"):
                    weights = tf.pad(
                        tf.to_float(tf.greater(seqs, 0)),
                        [[0,0],[1,0]], constant_values=1.0)
                    seqs = tf.pad(seqs, [[0,0],[1,1]])
                    labels = seqs[:,1:]
                    inputs = tf.nn.embedding_lookup(input_embedding, seqs[:,:-1])
                    if decoder_type == "lstm":
                        initial_state = tuple([tf.zeros([batch_size, size])]*2)
                        outputs, state = lstm_decoder(inputs, initial_state)
                    elif decoder_type == "bilstm":
                        initial_state = tuple([tf.zeros([batch_size, 2*size])]*2)
                        padded_seqs = tf.pad(seqs, [[0,0],[0,1]])
                        padded_inputs = tf.nn.embedding_lookup(input_embedding, padded_seqs)
                        outputs, state = bilstm_decoder(padded_inputs, initial_state)
                        outputs_fw, outputs_bw = tf.split(outputs, 2, axis=-1)
                        outputs = outputs_fw[:,:-2]+outputs_bw[:,2:]
                    elif decoder_type == "attn":
                        dec_inputs = tf.TensorArray(tf.float32, 0,
                            dynamic_size=True, clear_after_read=False, infer_shape=False)
                        initial_state = (dec_inputs,)
                        outputs, state = attn_decoder(inputs, initial_state)
                    encodes = tf.expand_dims(
                        tf.concat([inputs[:,1:], outputs[:,:-1]], axis=-1), axis=2)
                    logits = logit_fn(tf.reshape(outputs, [batch_size*(seq_length+1), size]))
                    logits = tf.reshape(
                        logits, [batch_size, seq_length+1, self.vocab_size])
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits)
                    loss = tf.reduce_sum(losses*weights) / tf.reduce_sum(weights)
                return loss, encodes, tf.expand_dims(seqs[:,1:-1], 2), None, None, seqs


            """ lm components """
            def segment(seqs, num_sample=2, segs=None, reuse=None):
                """
                segment seqs
                """
                with tf.variable_scope("seg", reuse=reuse):

                    batch_size = tf.shape(seqs)[0]
                    length = tf.shape(seqs)[1]

                    if trainable:
                        uni_lstm_fw = tf.contrib.cudnn_rnn.CudnnLSTM(
                            num_layers=1,
                            num_units=self.size,
                            direction="unidirectional")
                        uni_lstm_bw = tf.contrib.cudnn_rnn.CudnnLSTM(
                            num_layers=1,
                            num_units=self.size,
                            direction="unidirectional")
                        bi_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                            num_layers=1,
                            num_units=self.size,
                            direction="bidirectional")
                    else:
                        single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.size)
                        lstm_fw_cell = [single_cell() for _ in range(1)]
                        lstm_bw_cell = [single_cell() for _ in range(1)]
                        def uni_lstm_fw(inputs, training=False):
                            with tf.variable_scope("cudnn_lstm"):
                                outputs, state = tf.nn.dynamic_rnn(
                                    tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(1)]),
                                    inputs,
                                    dtype=tf.float32,
                                    time_major=True)
                            return outputs, tf.expand_dims(state, axis=0)
                        def uni_lstm_bw(inputs, training=False):
                            with tf.variable_scope("cudnn_lstm"):
                                outputs, state = tf.nn.dynamic_rnn(
                                    tf.nn.rnn_cell.MultiRNNCell([single_cell() for _ in range(1)]),
                                    inputs,
                                    dtype=tf.float32,
                                    time_major=True)
                            return outputs, tf.expand_dims(state, axis=0)
                        def bi_lstm(inputs, training=False):
                            with tf.variable_scope("cudnn_lstm"):
                                outputs, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                                    lstm_fw_cell,
                                    lstm_bw_cell,
                                    inputs,
                                    dtype=tf.float32,
                                    time_major=True)
                            return outputs, tf.stack([state_fw, state_bw], axis=0)

                    padded_seqs = tf.pad(seqs, [[0,0], [1,1]])
                    masks = tf.not_equal(padded_seqs, 0)
                    seg_masks_and = tf.logical_or(masks[:,:-1], masks[:,1:])
                    seg_masks_or = tf.logical_xor(masks[:,:-1], masks[:,1:])
                    seg_weights = tf.to_float(tf.logical_and(masks[:,:-1], masks[:,1:]))
                    inputs = tf.nn.embedding_lookup(self.input_embedding, tf.maximum(seqs, 0))

                    with tf.variable_scope("base"):
                        outputs, state = bi_lstm(tf.transpose(inputs, [1,0,2]), training=trainable)
                        outputs = tf.transpose(outputs, [1,0,2])
                        adjacent_encodes = tf.concat([outputs[:,:-1], outputs[:,1:]], axis=-1)
                        seg_logits_base = model_utils.MLP(
                            adjacent_encodes,
                            2,
                            self.size,
                            1,
                            dropout=1.0-self.dropout,
                            is_training=self.training,
                            scope="seg_logits")
                        seg_logits_base = tf.pad(tf.squeeze(seg_logits_base, -1), [[0,0],[1,1]])

                    def body(seg_logits, seg_predicts, seg_predicts_prev, i):
                        seg_predicts_prev = seg_predicts
                        seg_feats = tf.expand_dims(seg_predicts, axis=2)
                        fw_feats = tf.concat([inputs, seg_feats[:,:-1]], axis=2)
                        bw_feats = tf.concat([inputs, seg_feats[:,1:]], axis=2)
                        with tf.variable_scope("fw"):
                            fw_feats = tf.transpose(fw_feats, [1, 0, 2])
                            fw_outputs, state = uni_lstm_fw(fw_feats, training=trainable)
                            fw_outputs = tf.transpose(fw_outputs, [1, 0, 2])
                        with tf.variable_scope("bw"):
                            bw_feats = tf.reverse(tf.transpose(bw_feats, [1, 0, 2]), axis=[0])
                            bw_outputs, state = uni_lstm_bw(bw_feats, training=trainable)
                            bw_outputs = tf.reverse(tf.transpose(bw_outputs, [1, 0, 2]), axis=[1])
                        adjacent_encodes = tf.concat([fw_outputs[:,:-1], bw_outputs[:,1:]], axis=-1)
                        seg_logits = model_utils.MLP(
                            adjacent_encodes,
                            2,
                            self.size,
                            1,
                            dropout=1.0-self.dropout,
                            is_training=self.training,
                            scope="seg_logits")
                        seg_logits = tf.pad(tf.squeeze(seg_logits, -1), [[0,0],[1,1]])
                        seg_logits += seg_logits_base
                        seg_probs = tf.sigmoid(seg_logits)
                        seg_predicts = tf.greater(seg_probs, tf.random_uniform(tf.shape(seg_probs)))
                        seg_predicts = tf.logical_or(tf.logical_and(seg_predicts, seg_masks_and), seg_masks_or)
                        seg_predicts = tf.to_float(seg_predicts)
                        i += 1
                        return seg_logits, seg_predicts, seg_predicts_prev, i

                    seg_predicts_list = []
                    if segs != None:
                        seg_logits, seg_predicts, _, _ = body(None, segs, segs, 0)
                        seg_predicts_list.append(seg_predicts)
                        tf.get_variable_scope().reuse_variables()
                    else:
                        seg_logits = tf.zeros([batch_size, length+1])

                    seg_predicts = tf.greater(tf.sigmoid(seg_logits_base), tf.random_uniform(tf.shape(seg_logits_base)))
                    seg_predicts = tf.logical_or(tf.logical_and(seg_predicts, seg_masks_and), seg_masks_or)
                    seg_predicts = tf.to_float(seg_predicts)
                    seg_predicts_prev = tf.zeros([batch_size, length+1])
                    i = 0
                    for _ in range(num_sample):
                        for _ in range(10):
                            _, seg_predicts, seg_predicts_prev, i = body(
                                None, seg_predicts, seg_predicts_prev, i)
                            seg_predicts = tf.where(
                                tf.greater(tf.random_uniform(tf.shape(seg_predicts)), 0.9/float(i)),
                                seg_predicts,
                                seg_predicts_prev)
                            tf.get_variable_scope().reuse_variables()
                        seg_predicts_list.append(seg_predicts)

                    if segs != None:
                        seg_loss_gold = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=segs,
                            logits=seg_logits)
                        seg_loss_gold += tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=segs,
                            logits=seg_logits_base)
                        segmented_seqs_gold, segment_idxs_gold = model_utils.slice_words(
                            seqs, segs[:,1:-1], get_idxs=True)
                        segmented_seqs_gold = tf.stop_gradient(segmented_seqs_gold)
                        max_char_length = tf.shape(segmented_seqs_gold)[2]
                        segmented_seqs_gold = tf.cond(
                            tf.less(max_char_length, MAX_CHAR_LEN),
                            lambda: segmented_seqs_gold,
                            lambda: segmented_seqs_gold[:,:,:MAX_CHAR_LEN])
                       # seg_predicts.append(segs)
                       # num_sample += 1
                    else:
                        seg_loss_gold = 0.0
                        segmented_seqs_gold = None
                    seg_predicts = tf.concat(seg_predicts_list, axis=0)
                    segmented_seqs_hyp, segment_idxs_hyp = model_utils.slice_words(
                        tf.tile(seqs, [num_sample,1]), seg_predicts[:,1:-1], get_idxs=True)
                    segmented_seqs_hyp = tf.stop_gradient(segmented_seqs_hyp)
                    max_char_length = tf.shape(segmented_seqs_hyp)[2]
                    segmented_seqs_hyp = tf.cond(
                        tf.less(max_char_length, MAX_CHAR_LEN),
                        lambda: segmented_seqs_hyp,
                        lambda: segmented_seqs_hyp[:,:,:MAX_CHAR_LEN])
                    seg_loss_hyp = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.stop_gradient(seg_predicts),
                        logits=tf.tile(seg_logits, [num_sample,1]))
                    seg_loss_hyp *= tf.tile(seg_weights, [num_sample,1])
                    seg_loss_hyp = tf.stack(tf.split(tf.reduce_sum(seg_loss_hyp, axis=1), num_sample, axis=0), axis=1)

                    return seg_loss_gold, seg_loss_hyp, segmented_seqs_gold, segmented_seqs_hyp, seg_weights

            def embed_words(segmented_seqs, reuse=None):
                """
                embed seq of words into vectors
                args:
                    segmented_seqs: batch_size x word_length x char_length
                """
                with tf.variable_scope("word_embedder", reuse=reuse):

                    batch_size = tf.shape(segmented_seqs)[0]
                    max_word_length = tf.shape(segmented_seqs)[1]
                    max_char_length = tf.shape(segmented_seqs)[2]
                    masks = tf.reduce_any(tf.not_equal(segmented_seqs, 0), axis=2)

                    char_embeds = tf.nn.embedding_lookup(self.input_embedding, tf.maximum(segmented_seqs, 0))
                    l1_embeds = model_utils.convolution2d(
                        char_embeds,
                        [self.size]*2,
                        [[1,2],[1,3]],
                        activation_fn=tf.nn.relu,
                        is_training=self.training,
                        scope="l1_convs")
                    char_embeds = tf.nn.max_pool(char_embeds, [1,1,2,1], [1,1,2,1], padding='SAME')
                    l2_embeds = model_utils.convolution2d(
                        char_embeds,
                        [self.size]*2,
                        [[1,2],[1,3]],
                        activation_fn=tf.nn.relu,
                        is_training=self.training,
                        scope="l2_convs")
                    concat_embeds = tf.concat(
                        [tf.reduce_max(tf.nn.relu(char_embeds), axis=2),
                         tf.reduce_max(l1_embeds, axis=2),
                         tf.reduce_max(l2_embeds, axis=2)],
                        axis=-1)
                    word_embeds = model_utils.highway(
                        concat_embeds,
                        2,
                        activation_fn=tf.nn.relu,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="highway")
                    word_embeds = tf.contrib.layers.layer_norm(
                        word_embeds, begin_norm_axis=-1,
                        variables_collections=collections, trainable=trainable)
                    word_embeds = model_utils.fully_connected(
                        word_embeds,
                        self.size,
                        is_training=self.training,
                        scope="projs")
                    word_embeds += model_utils.MLP(
                        word_embeds,
                        2,
                        2*self.size,
                        self.size,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="MLP")
                    word_embeds = tf.contrib.layers.layer_norm(
                        word_embeds, begin_norm_axis=-1,
                        variables_collections=collections, trainable=trainable)

                    masksLeft = tf.pad(masks, [[0,0],[1,0]])[:,:-1]
                    masksRight = tf.pad(masks, [[0,0],[0,1]])[:,1:]
                    word_masks = tf.logical_or(masks, tf.logical_or(masksLeft, masksRight))
                return word_embeds, word_masks, tf.zeros([], dtype=tf.int32), tf.zeros([])

            def embed_words2(segmented_seqs, reuse=None):
                """
                embed seq of words into vectors
                """
                with tf.variable_scope("word_embedder", reuse=reuse):

                    batch_size = tf.shape(segmented_seqs)[0]
                    max_word_length = tf.shape(segmented_seqs)[1]
                    max_char_length = tf.shape(segmented_seqs)[2]
                    masks = tf.reduce_any(tf.not_equal(segmented_seqs, 0), axis=2)

                    char_embeds = tf.nn.embedding_lookup(self.input_embedding, tf.maximum(segmented_seqs, 0))
                    conv_embeds = model_utils.convolution2d(
                        char_embeds,
                        [self.size]*4,
                        [[1,1],[1,2],[1,3],[1,4]],
                        activation_fn=tf.nn.relu,
                        is_training=self.training,
                        scope="conv_embeds")
                    word_embeds = tf.reduce_max(conv_embeds, axis=2)
                    word_embeds = model_utils.highway(
                        word_embeds,
                        2,
                        activation_fn=tf.nn.relu,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="highway")
                    word_embeds = tf.contrib.layers.layer_norm(
                        word_embeds, begin_norm_axis=-1,
                        variables_collections=collections, trainable=trainable)
                    word_embeds = model_utils.fully_connected(
                        word_embeds,
                        self.size,
                        is_training=self.training,
                        scope="projs")

                    num_splits = 16
                    num_categories = self.word_embedding_value.get_shape()[0].value
                    word_assign_norms = tf.norm(
                        tf.stack(tf.split(tf.expand_dims(word_embeds, axis=-2) - self.word_embedding_value,
                                          num_splits, axis=-1), axis=-2),
                        axis=-1)
                    word_assign_ids = tf.argmin(word_assign_norms, axis=-2, output_type=tf.int32)
                    inverse_assign_ids = tf.argmin(
                        tf.reshape(word_assign_norms, [batch_size*max_word_length, num_categories, num_splits]),
                        axis=0, output_type=tf.int32)
                    word_embeds_quantised = []
                    word_embed_loss = []
                    for we, wid, ev, iid in zip(tf.split(word_embeds, num_splits, axis=-1)[1:],
                                                tf.unstack(word_assign_ids, axis=-1)[1:],
                                                tf.split(self.word_embedding_value, num_splits, axis=1)[1:],
                                                tf.unstack(inverse_assign_ids, axis=0)[1:]):
                        quantised_embeds = tf.nn.embedding_lookup(ev, wid)
                        word_embeds_quantised.append(quantised_embeds)
                        mean_embedding = tf.unsorted_segment_mean(
                            we, wid, num_categories)
                        inverse_embedding = tf.nn.embedding_lookup(
                            tf.reshape(we, [batch_size*max_word_length, int(self.size/num_splits)]),
                            iid)
                        target_embedding = tf.where(
                            tf.reduce_all(tf.equal(mean_embedding, 0), axis=-1),
                            inverse_embedding,
                            mean_embedding)
                        word_embed_loss.append(
                            tf.reduce_mean(
                                tf.reduce_sum(tf.square(ev - tf.stop_gradient(target_embedding)), axis=-1)))

                    word_embeds_quantised = tf.split(word_embeds, num_splits, axis=-1)[:1] + word_embeds_quantised
                    word_embeds_quantised = tf.concat(word_embeds_quantised, axis=-1)
                    word_embed_loss = sum(word_embed_loss)
                    word_embed_loss += 0.25*tf.reduce_sum(
                        tf.square(word_embeds - tf.stop_gradient(word_embeds_quantised)), axis=-1)

                    @tf.custom_gradient
                    def quantise_embeds(word_embeds, word_embeds_quantised):
                        def grad_fn(dy):
                            return [dy, None]
                        return word_embeds_quantised, grad_fn
                    word_embeds_quantised = quantise_embeds(word_embeds, word_embeds_quantised)

                    masksLeft = tf.pad(masks, [[0,0],[1,0]])[:,:-1]
                    masksRight = tf.pad(masks, [[0,0],[0,1]])[:,1:]
                    word_masks = tf.logical_or(masks, tf.logical_or(masksLeft, masksRight))
                    word_embed_loss = tf.reduce_sum(word_embed_loss*tf.to_float(word_masks)) / \
                                      tf.reduce_sum(tf.to_float(word_masks))
                return word_embeds_quantised, word_masks, word_assign_ids, word_embed_loss

            def embed_words3(segmented_seqs, reuse=None):
                """
                embed seq of words into vectors
                """
                with tf.variable_scope("word_embedder", reuse=reuse):

                    batch_size = tf.shape(segmented_seqs)[0]
                    max_word_length = tf.shape(segmented_seqs)[1]
                    max_char_length = tf.shape(segmented_seqs)[2]
                    masks = tf.reduce_any(tf.not_equal(segmented_seqs, 0), axis=2)

                    char_embeds = tf.nn.embedding_lookup(self.input_embedding, tf.maximum(segmented_seqs, 0))
                    l1_embeds = model_utils.convolution2d(
                        char_embeds,
                        [self.size]*2,
                        [[1,2],[1,3]],
                        activation_fn=tf.nn.relu,
                        is_training=self.training,
                        scope="l1_convs")
                    char_embeds = tf.nn.max_pool(char_embeds, [1,1,2,1], [1,1,2,1], padding='SAME')
                    l2_embeds = model_utils.convolution2d(
                        char_embeds,
                        [self.size]*2,
                        [[1,2],[1,3]],
                        activation_fn=tf.nn.relu,
                        is_training=self.training,
                        scope="l2_convs")
                    concat_embeds = tf.concat(
                        [tf.reduce_max(tf.nn.relu(char_embeds), axis=2),
                         tf.reduce_max(l1_embeds, axis=2),
                         tf.reduce_max(l2_embeds, axis=2)],
                        axis=-1)
                    word_embeds = model_utils.highway(
                        concat_embeds,
                        2,
                        activation_fn=tf.nn.relu,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="highway")
                    word_embeds = tf.contrib.layers.layer_norm(
                        word_embeds, begin_norm_axis=-1,
                        variables_collections=collections, trainable=trainable)
                    word_embeds = model_utils.fully_connected(
                        word_embeds,
                        self.size,
                        is_training=self.training,
                        scope="projs")
                    word_embeds += model_utils.MLP(
                        word_embeds,
                        2,
                        2*self.size,
                        self.size,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="MLP")
                    word_embeds_base = tf.contrib.layers.layer_norm(
                        word_embeds, begin_norm_axis=-1,
                        variables_collections=collections, trainable=trainable)
                    word_embeds_match = model_utils.fully_connected(
                        word_embeds_base,
                        self.size*self.num_categories,
                        is_training=self.training,
                        scope="match_projs")

                    word_assign_norms = tf.matmul(
                        tf.transpose(
                            tf.reshape(
                                word_embeds_match,
                                [batch_size*max_word_length, self.num_categories, self.size]),
                            [1,0,2]),
                        tf.math.l2_normalize(
                            tf.reshape(
                                self.word_embedding_value,
                                [self.num_categories, self.category_size, self.size]),
                            axis=-1),
                        transpose_b=True)
                    word_assign_norms = tf.reshape(
                        word_assign_norms, [self.num_categories, batch_size, max_word_length, self.category_size])
                    word_assign_ids = tf.argmax(word_assign_norms, axis=-1, output_type=tf.int32)
                    word_embeds_quantised = []
                    word_embed_loss = []
                    for we, wid, ev in zip(tf.split(word_embeds_match, self.num_categories, axis=-1),
                                           tf.unstack(word_assign_ids, axis=0),
                                           tf.split(self.word_embedding_value, self.num_categories, axis=0)):
                        quantised_embeds = tf.math.l2_normalize(tf.nn.embedding_lookup(ev, wid), axis=-1)
                        we_norm = tf.norm(we, axis=-1)
                        we = tf.math.l2_normalize(we, axis=-1)
                        word_embed_loss.append(
                            we_norm*(1.0 - tf.reduce_sum(tf.stop_gradient(we)*quantised_embeds, axis=-1)))
                        word_embeds_quantised.append(tf.expand_dims(we_norm, axis=-1)*quantised_embeds)

                    word_embeds_quantised = sum(word_embeds_quantised)
                    word_embed_loss = sum(word_embed_loss)

                    @tf.custom_gradient
                    def quantise_embeds(word_embeds, word_embeds_quantised):
                        def grad_fn(dy):
                            return [dy, None]
                        return word_embeds_quantised, grad_fn
                    word_embeds_quantised = quantise_embeds(
                        sum(tf.split(word_embeds_match, self.num_categories, axis=-1)), word_embeds_quantised)
                    word_embeds_final = word_embeds_base + word_embeds_quantised

                    masksLeft = tf.pad(masks, [[0,0],[1,0]])[:,:-1]
                    masksRight = tf.pad(masks, [[0,0],[0,1]])[:,1:]
                    word_masks = tf.logical_or(masks, tf.logical_or(masksLeft, masksRight))
                    word_embed_loss = tf.reduce_sum(word_embed_loss*tf.to_float(word_masks)) / \
                                      tf.reduce_sum(tf.to_float(word_masks))
                return word_embeds_final, word_masks, tf.transpose(word_assign_ids, [1,2,0]), word_embed_loss

            embed_words_fn = embed_words
            self.num_categories = 0
            self.category_size = 0

           # embed_words_fn = embed_words2
           # self.num_categories = 1
           # self.category_size = 16

           # embed_words_fn = embed_words3
           # self.num_categories = 1
           # self.category_size = 128

            def encode_words(field_embeds, value_embeds, attn_masks, reuse=None):
                """
                encode seq of words, include embeds and contexts
                field_embeds: batch_size x seq_length x dim
                value_embeds: batch_size x seq_length x dim
                attn_masks: batch_size x seq_length
                """

                with tf.variable_scope("encoder", reuse=reuse):
                    encodes = model_utils.transformer(
                        field_embeds,
                        self.num_layers,
                        values=value_embeds,
                        masks=attn_masks,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="transformer")
                return encodes

            def match_embeds(encodes, token_embeds, reuse=None):
                """
                outputs the degree of matchness of the word embeds and contexts
                encodes: batch_size x dim
                token_embeds: batch_size x num_candidates x dim or num_candidates x dim
                task_embeds: dim
                """

                with tf.variable_scope("matcher", reuse=reuse):
                    dim = encodes.get_shape()[-1].value
                    encodes = model_utils.fully_connected(
                        encodes,
                        dim,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="enc_projs")
                    token_embeds = model_utils.fully_connected(
                        token_embeds,
                        dim,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="tok_projs")

                    if len(token_embeds.get_shape()) == 2:
                        outputs = tf.matmul(
                            encodes, token_embeds, transpose_b=True)
                    else:
                        outputs = tf.matmul(
                            token_embeds, tf.expand_dims(encodes, axis=-1))
                        outputs = tf.squeeze(outputs, axis=-1)
                    outputs /= tf.sqrt(float(dim))
                return outputs

            def train_generator(encodes, encMasks, targetSeqs, reuse=None):
                """
                encodes: batch_size x enc_seq_length x dim
                encMasks: batch_size x enc_seq_length
                targetSeqs: batch_size x dec_seq_length
                """

                with tf.variable_scope("generator", reuse=reuse):

                    attn_cell = model_utils.AttentionCell(
                        size,
                        num_layer=2,
                        dropout=1.0-dropout,
                        is_training=self.training)

                    batch_size = tf.shape(encodes)[0]
                    dim = self.spellout_embedding.get_shape()[-1].value

                    decInputs = tf.TensorArray(tf.float32, 0,
                        dynamic_size=True, clear_after_read=False, infer_shape=False)
                    initialState = (decInputs, encodes, encMasks)
                    inputs = tf.nn.embedding_lookup(
                        self.spellin_embedding,
                        tf.pad(tf.maximum(targetSeqs, 0), [[0,0],[1,0]]))
                    targetIds = tf.pad(tf.maximum(targetSeqs, 0), [[0,0],[0,1]])
                    decMasks = tf.not_equal(tf.pad(targetSeqs, [[0,0],[0,1]]), 0)
                    decMasks = tf.logical_or(decMasks, tf.pad(tf.not_equal(targetIds, 0), [[0,0],[1,0]])[:,:-1])
                    decLength = tf.shape(inputs)[1]
                    outputs, state = attn_cell(inputs, initialState)
                    logits = tf.matmul(
                        tf.reshape(outputs, [batch_size*decLength, outputs.get_shape()[-1].value]),
                        self.spellout_embedding,
                        transpose_b=True) / tf.sqrt(float(dim))
                    logits = tf.reshape(logits, [batch_size, decLength, self.vocab_size])
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=targetIds,
                        logits=logits) * tf.to_float(decMasks)
                    loss = tf.reduce_sum(losses) / tf.reduce_sum(tf.to_float(decMasks))
                return loss

            def generate_word(encodes, encMasks, length=6, beam_size=64, num_candidates=1, reuse=None):
                """
                args:
                    encodes: batch_size x enc_seq_length x dim
                    encMasks: batch_size x enc_seq_length

                return:
                    candidates: batch_size x num_candidates x length
                    scores: batch_size x num_candidates
                """

                with tf.variable_scope("generator", reuse=reuse):

                    attn_cell = model_utils.AttentionCell(
                        size,
                        num_layer=2,
                        dropout=1.0-dropout,
                        is_training=self.training)

                    decInputs = tf.TensorArray(tf.float32, 0,
                        dynamic_size=True, clear_after_read=False, infer_shape=False)
                    initialState = (decInputs, encodes, encMasks)

                    logit_fn = lambda outputs: tf.matmul(outputs, self.spellout_embedding, transpose_b=True) / \
                        tf.sqrt(float(self.spellout_embedding.get_shape()[-1].value))

                    candidates, scores = model_utils.beam_dec(
                        length,
                        initialState,
                        self.spellin_embedding,
                        attn_cell,
                        logit_fn,
                        gamma=0.65,
                        beam_size=beam_size,
                        num_candidates=num_candidates)
                return candidates, scores

            def sample_field(input_embeds, input_mask, field_id=0, word_length=20, reuse=None):
                """
                args:
                    input_embeds: batch_size x input_length x dim

                return:
                    candidates: batch_size x seq_length
                """

                batch_size = tf.shape(input_embeds)[0]

                posit_ids = tf.tile(tf.expand_dims(tf.range(word_length), 0), [batch_size, 1])
                posit_embeds = model_utils.embed_position(
                    posit_ids,
                    self.size)
                field_embeds = field_embedding[field_id] + posit_embeds
                field_embeds = tf.concat([field_embeds, input_embeds], axis=1)
                word_embeds = tf.zeros([batch_size, 0, self.size])
                word_list = []
                reuse = reuse
                for i in range(word_length):
                    attn_masks = tf.concat(
                        [tf.sequence_mask(tf.ones([batch_size], dtype=tf.int32)*(i+1), maxlen=word_length),
                         input_mask],
                        axis=1)
                    word_embeds = tf.pad(word_embeds, [[0,0],[0,tf.shape(field_embeds)[1]-i],[0,0]])
                    encodes = encode_words(field_embeds, word_embeds, attn_masks, reuse=reuse)
                    word = generate_word(encodes[i:i+1], tf.ones([batch_size, 1], dtype=tf.bool), reuse=reuse)
                    word = tf.squeeze(word, axis=1)
                    word_list.append(word)
                    reuse = True
                return candidates

            """ lm model """
            def ultra_lm(pieces):
                """
                pieces: [(seqs0, segs0), (seqs1, segs1)...]
                """

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
                    self.input_embedding = input_embedding
                    spellin_embedding = model_utils.MLP(
                        char_embedding,
                        2,
                        self.size,
                        self.size,
                        is_training=self.training,
                        scope="spellin")
                    spellin_embedding = tf.contrib.layers.layer_norm(
                        spellin_embedding, begin_norm_axis=-1,
                        variables_collections=collections, trainable=trainable)
                    self.spellin_embedding = spellin_embedding
                    spellout_embedding = model_utils.fully_connected(
                        spellin_embedding,
                        self.size,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="spellout")
                    self.spellout_embedding = spellout_embedding

                    word_embedding_key = tf.get_variable(
                        "word_embedding_key",
                        shape=[self.num_categories*self.category_size, self.size],
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal(0.0, 0.01),
                        trainable=trainable,
                        collections=collections)
                    word_embedding_value = tf.get_variable(
                        "word_embedding_value",
                        shape=[self.num_categories*self.category_size, self.size],
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal(0.0, 0.01),
                        trainable=trainable,
                        collections=collections)
                    self.word_embedding_key = word_embedding_key
                    self.word_embedding_value = word_embedding_value

                    field_embedding = tf.get_variable(
                        "field_embedding",
                        shape=[10, self.size],
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal(0.0, 0.01),
                        trainable=trainable,
                        collections=collections)
                    self.field_embedding = field_embedding

                    task_embedding = tf.get_variable(
                        "task_embedding",
                        shape=[10, self.size],
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal(0.0, 0.01),
                        trainable=trainable,
                        collections=collections)
                    self.task_embedding = task_embedding

                num_sample = 1
                seg_loss_gold_list, seg_loss_hyp_list, segmented_seqs_gold_list, segmented_seqs_hyp_list, \
                    seg_weights_list = [],[],[],[],[]
                for seqs, segs in pieces:
                    seg_loss_gold, seg_loss_hyp, segmented_seqs_gold, segmented_seqs_hyp, seg_weights = segment(seqs, num_sample=num_sample, segs=segs)
                    seg_loss_gold_list.append(seg_loss_gold)
                    seg_loss_hyp_list.append(seg_loss_hyp)
                    segmented_seqs_gold_list.append(segmented_seqs_gold)
                    segmented_seqs_hyp_list.append(segmented_seqs_hyp)
                    seg_weights_list.append(seg_weights)
                seg_loss_gold = sum(
                    map(lambda (loss, weights): tf.reduce_sum(loss*weights) / tf.reduce_sum(weights),
                        zip(seg_loss_gold_list, seg_weights_list)))

                def logit_fn(outputs):
                    logits = tf.matmul(outputs, output_embedding, transpose_b=True)
                    return logits

                def decode(contexts, length=5, num_candidates=1, reuse=None):
                    with tf.variable_scope("generator", reuse=reuse):
                        batch_size = tf.shape(contexts)[0]
                        max_word_seq_length = tf.shape(contexts)[1]
                        if decoder_type == "attn":
                            dec_inputs = tf.TensorArray(tf.float32, 0,
                                dynamic_size=True, clear_after_read=False, infer_shape=False)
                            encodes = tf.reshape(contexts, [batch_size*max_word_seq_length, 1, 2*self.vocab_dim])
                            masks = tf.ones([batch_size*max_word_seq_length, 1], dtype=tf.bool)
                            initial_state = (dec_inputs, encodes, masks)
                            decoder_cell = attn_decoder
                        sample_seqs, sample_scores = model_utils.stochastic_dec(
                            length,
                            initial_state,
                            input_embedding,
                            decoder_cell,
                            logit_fn,
                            num_candidates=num_candidates)
                        sample_seqs = tf.reshape(
                            sample_seqs, [batch_size, max_word_seq_length, num_candidates, length])
                        sample_scores = tf.reshape(
                            sample_scores, [batch_size, max_word_seq_length, num_candidates])
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
                        sample_seqs = tf.squeeze(sample_seqs, axis=2)
                        sample_scores = tf.squeeze(sample_scores, axis=2)
                        sample_seqs = tf.pad(sample_seqs, [[0,0],[0,0],[0,max_char_length-5]])
                        sample_scores -= 1e20 * tf.to_float(tf.logical_not(masks))
                        choices = tf.squeeze(tf.multinomial(sample_scores, 1), axis=-1)
                        masks = tf.one_hot(choices, max_seq_length)
                        masks = tf.cast(masks, tf.bool)
                        segmented_seqs = tf.where(
                            tf.reshape(masks, [batch_size*max_seq_length]),
                            tf.reshape(sample_seqs, [batch_size*max_seq_length, -1]),
                            tf.reshape(segmented_seqs, [batch_size*max_seq_length, -1]))
                        segmented_seqs = tf.reshape(segmented_seqs, [batch_size, max_seq_length, -1])
                        #seqs = model_utils.stitch_chars(segmented_seqs)
                        seqs = tf.reshape(segmented_seqs, [batch_size, -1])
                    return seqs

                if loss_type == 'unsup' or segmented_seqs_gold == None:                        
                    max_word_length = tf.shape(segmented_seqs_hyp)[1]
                    max_char_length = tf.shape(segmented_seqs_hyp)[2]

                    word_embeds_hyp, contexts_hyp, word_encodes_hyp = encode_words(
                        segmented_seqs_hyp)
                    masks = tf.reduce_any(tf.greater(segmented_seqs_hyp, 0), axis=-1)
                    positive_logits = discriminate(word_embeds_hyp, contexts_hyp)
                    positive_probs_mean = tf.reduce_sum(
                        tf.log_sigmoid(positive_logits)*tf.to_float(masks), axis=1) / \
                        tf.reduce_sum(tf.to_float(masks), axis=1)
                    positive_probs_mean = tf.stack(tf.split(positive_probs_mean, num_sample, axis=0), axis=1)
                    idx = tf.argmax(positive_probs_mean, axis=-1, output_type=tf.int32)
                    idx = tf.stack([tf.range(batch_size), idx], axis=1)

                    seg_loss_hyp = tf.reduce_sum(tf.reshape(tf.gather_nd(seg_loss_hyp, idx), [batch_size])) / \
                        tf.reduce_sum(seg_weights)
                    segmented_seqs_hyp = tf.stack(tf.split(segmented_seqs_hyp, num_sample, axis=0), axis=1)
                    segmented_seqs_hyp = tf.reshape(
                        tf.gather_nd(segmented_seqs_hyp, idx), [batch_size, max_word_length, max_char_length])
                    segment_idxs_hyp = tf.stack(tf.split(segment_idxs_hyp, num_sample, axis=0), axis=1)
                    segment_idxs_hyp = tf.reshape(
                        tf.gather_nd(segment_idxs_hyp, idx), [batch_size, -1])
                    word_embeds_hyp = tf.stack(tf.split(word_embeds_hyp, num_sample, axis=0), axis=1)
                    word_embeds_hyp = tf.reshape(
                        tf.gather_nd(word_embeds_hyp, idx), [batch_size, max_word_length, 2*self.vocab_dim])
                    contexts_hyp = tf.stack(tf.split(contexts_hyp, num_sample, axis=0), axis=1)
                    contexts_hyp = tf.reshape(
                        tf.gather_nd(contexts_hyp, idx),
                        [batch_size, max_word_length, self.num_layers*2*self.vocab_dim])
                    word_encodes_hyp = tf.stack(tf.split(word_encodes_hyp, num_sample, axis=0), axis=1)
                    word_encodes_hyp = tf.reshape(
                        tf.gather_nd(word_encodes_hyp, idx),
                        [batch_size, max_word_length, (self.num_layers+1)*2*self.vocab_dim])
                    
                    masks = tf.reduce_any(tf.greater(segmented_seqs_hyp, 0), axis=-1)
                    valid_word_embeds_hyp = tf.boolean_mask(word_embeds_hyp, masks)
                    valid_contexts_hyp = tf.boolean_mask(contexts_hyp, masks)
                    positive_logits = tf.stack(tf.split(positive_logits, num_sample, axis=0), axis=1)
                    positive_logits = tf.reshape(
                        tf.gather_nd(positive_logits, idx), [batch_size, max_word_length])
                    positive_logits = tf.boolean_mask(positive_logits, masks)
                    idxs = tf.random_shuffle(tf.range(tf.shape(valid_word_embeds_hyp)[0]))
                    negative_logits = discriminate(
                        tf.gather(valid_word_embeds_hyp, idxs), valid_contexts_hyp, reuse=True)
                    positive_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones(tf.shape(positive_logits)), logits=positive_logits)
                    negative_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.zeros(tf.shape(negative_logits)), logits=negative_logits)
                    discriminator_loss_hyp = tf.reduce_mean(positive_loss) + tf.reduce_mean(negative_loss)

                    lm_char_loss_per_sample_hyp, lm_char_weights_per_sample_hyp = get_char_loss(
                        contexts_hyp, seqs, segment_idxs_hyp)
                    lm_char_loss_hyp = tf.reduce_sum(lm_char_loss_per_sample_hyp) / \
                        tf.reduce_sum(lm_char_weights_per_sample_hyp)
                    segmented_seq_lengths_hyp = tf.reduce_sum(tf.to_int32(masks), axis=-1)
                   # pos_logits_hyp = get_pos_logits(word_encodes_hyp)
                   # viterbi_tags, viterbi_scores = tf.contrib.crf.crf_decode(
                   #     pos_logits_hyp, transition_params, segmented_seq_lengths_hyp)

                   # tf.get_variable_scope().reuse_variables()
                   # sample_seqs = sample(segmented_seqs_hyp)

                    loss = seg_loss_hyp+discriminator_loss_hyp
                    return loss, word_encodes_hyp, segmented_seqs_hyp, seg_predicts, None, seqs

                else:
                    word_embeds_gold_list, word_masks_gold_list, word_ids_gold_list = [],[],[]
                    seq_length_gold_list, pick_masks_gold_list, field_embeds_gold_list = [],[],[]
                    word_embed_loss_gold_list, word_select_loss_gold_list, word_gen_loss_gold_list = [],[],[]
                    reuse=None
                    for i, segmented_seqs_gold in enumerate(segmented_seqs_gold_list):

                        batch_size = tf.shape(segmented_seqs_gold)[0]
                        max_length = tf.shape(segmented_seqs_gold)[1]
                        seq_length_gold_list.append(max_length)

                        # embed words
                        word_embeds_gold, word_masks_gold, word_ids_gold, word_embed_loss_gold = embed_words_fn(
                            segmented_seqs_gold, reuse=reuse)
                        word_embeds_gold_list.append(word_embeds_gold)
                        word_masks_gold_list.append(word_masks_gold)
                        word_ids_gold_list.append(word_ids_gold)
                        word_embed_loss_gold_list.append(word_embed_loss_gold)

                        posit_ids = tf.tile(tf.expand_dims(tf.range(max_length), 0), [batch_size, 1])
                        posit_embeds = model_utils.embed_position(
                            posit_ids,
                            word_embeds_gold.get_shape()[-1].value)
                        field_embeds = field_embedding[i] + posit_embeds
                        field_embeds_gold_list.append(field_embeds)

                        pick_masks_gold = tf.less(tf.random_uniform([batch_size, max_length]), 0.15)
                        pick_masks_gold = tf.logical_and(pick_masks_gold, word_masks_gold)
                        pick_masks_gold_list.append(pick_masks_gold)
                        masked_word_embeds_gold = word_embeds_gold * \
                            tf.to_float(tf.expand_dims(tf.logical_not(pick_masks_gold), axis=-1))

                        reuse=True

                    # get the encodes
                    word_masks_gold = tf.concat(word_masks_gold_list, axis=1)
                    pick_masks_gold = tf.concat(pick_masks_gold_list, axis=1)
                    num_extra = 5
                    field_embeds_gold = tf.concat(
                        [tf.tile(tf.nn.embedding_lookup(task_embedding, [range(num_extra)]), [batch_size,1,1])] + \
                            field_embeds_gold_list,
                        axis=1)
                    word_embeds_gold = tf.pad(tf.concat(word_embeds_gold_list, axis=1), [[0,0],[num_extra,0],[0,0]])
                    masked_word_embeds_gold = word_embeds_gold * \
                        tf.to_float(tf.expand_dims(
                            tf.logical_not(tf.pad(pick_masks_gold, [[0,0],[num_extra,0]])), axis=-1))
                    attn_masks = tf.pad(word_masks_gold, [[0,0],[num_extra,0]], constant_values=True)
                    masked_attn_masks = tf.logical_and(word_masks_gold, tf.logical_not(pick_masks_gold))
                    masked_attn_masks = tf.pad(masked_attn_masks, [[0,0],[num_extra,0]], constant_values=True)
                    encodes_gold = encode_words(
                        field_embeds_gold, word_embeds_gold, attn_masks)
                    masked_encodes_gold = encode_words(
                        field_embeds_gold, masked_word_embeds_gold, masked_attn_masks, reuse=True)

                    # get the loss of each piece
                    encodes_gold_list = tf.split(encodes_gold, [num_extra]+seq_length_gold_list, axis=1)
                    masked_encodes_gold_list = tf.split(
                        masked_encodes_gold, [num_extra]+seq_length_gold_list, axis=1)
                    reuse = None
                    for segmented_seqs_gold, word_embeds_gold, masked_word_encodes_gold, \
                        word_masks_gold, pick_masks_gold in \
                        zip(segmented_seqs_gold_list, word_embeds_gold_list, masked_encodes_gold_list[1:],
                            word_masks_gold_list, pick_masks_gold_list):

                        valid_segmented_seqs_gold = tf.boolean_mask(segmented_seqs_gold, word_masks_gold)
                        valid_word_embeds_gold = tf.boolean_mask(word_embeds_gold, word_masks_gold)
                        pick_word_encodes_gold = tf.boolean_mask(masked_word_encodes_gold, pick_masks_gold)
                        pick_segmented_seqs_gold = tf.boolean_mask(segmented_seqs_gold, pick_masks_gold)
                        num_pick_words = tf.shape(pick_word_encodes_gold)[0]

                        unique_segmented_seqs_gold, unique_idxs = model_utils.unique_2d(valid_segmented_seqs_gold)
                        unique_word_embeds_gold = tf.gather(valid_word_embeds_gold, unique_idxs)

                        match_matrix = tf.equal(
                            tf.expand_dims(pick_segmented_seqs_gold, 1),
                            tf.expand_dims(unique_segmented_seqs_gold, 0))
                        match_matrix = tf.reduce_all(match_matrix, axis=-1)
                        match_idxs = tf.argmax(tf.to_int32(match_matrix), axis=-1)

                        cand_idxs = tf.random.multinomial(
                            tf.expand_dims(tf.log(self.word_count_array), 0),
                            512,
                            output_dtype=tf.int32)
                        cand_idxs, _ = tf.unique(tf.squeeze(cand_idxs, axis=0))
                        cand_word_ids = tf.gather(self.word_ids_array, cand_idxs)

                        word_len1 = tf.shape(segmented_seqs_gold)[-1]
                        word_len2 = tf.shape(self.word_ids_array)[-1]
                        max_word_len = tf.maximum(word_len1, word_len2)
                        unique_segmented_seqs_gold = tf.pad(
                            unique_segmented_seqs_gold, [[0,0],[0,max_word_len-word_len1]])
                        cand_word_ids = tf.pad(
                            cand_word_ids, [[0,0],[0,max_word_len-word_len2]])

                        match_matrix = tf.equal(
                            tf.expand_dims(cand_word_ids, 1),
                            tf.expand_dims(unique_segmented_seqs_gold, 0))
                        match_matrix = tf.reduce_all(match_matrix, axis=-1)
                        extra_mask = tf.logical_not(tf.reduce_any(match_matrix, axis=-1))
                        extra_word_ids = tf.boolean_mask(cand_word_ids, extra_mask)[:,:word_len2]
                        extra_word_embeds_gold = tf.squeeze(
                            embed_words_fn(tf.expand_dims(extra_word_ids, 1), reuse=True)[0], axis=1)
                        unique_word_embeds_gold = tf.concat(
                            [unique_word_embeds_gold, extra_word_embeds_gold], axis=0)

                        word_select_logits_gold = match_embeds(
                            pick_word_encodes_gold, unique_word_embeds_gold, reuse=reuse)
                        word_select_loss_gold = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=match_idxs, logits=word_select_logits_gold)
                        word_select_loss_gold = tf.reduce_mean(word_select_loss_gold)
                        word_select_loss_gold_list.append(word_select_loss_gold)

                        word_gen_loss_gold = train_generator(
                            tf.expand_dims(pick_word_encodes_gold, 1),
                            tf.ones([num_pick_words, 1], dtype=tf.bool),
                            pick_segmented_seqs_gold, reuse=reuse)
                        word_gen_loss_gold_list.append(word_gen_loss_gold)

                        reuse = True

                    # get the sentence level loss
                    sent_encodes_gold = tf.reshape(masked_encodes_gold_list[0], [batch_size*num_extra, self.size])
                    word_embeds_gold = tf.concat(word_embeds_gold_list, axis=1)
                    word_masks_gold = tf.concat(word_masks_gold_list, axis=1)
                    weights = tf.to_float(word_masks_gold)
                    select_probs = weights
                    select_probs /= (tf.reduce_sum(select_probs, axis=-1, keepdims=True) + 1e-20)
                    select_ids = tf.random.multinomial(
                        tf.log(select_probs), 1, output_dtype=tf.int32)
                    select_oneHot = tf.one_hot(
                        tf.squeeze(select_ids, axis=1), tf.shape(word_embeds_gold)[1])
                    select_embeds = tf.reduce_sum(word_embeds_gold * tf.expand_dims(select_oneHot, axis=-1), axis=1)
                    discriminate_logits = match_embeds(
                        sent_encodes_gold, select_embeds, reuse=True)
                    discriminate_logits = tf.reshape(discriminate_logits, [batch_size, num_extra, batch_size])
                    discriminate_logits = tf.reduce_sum(discriminate_logits, axis=1)
                    sent_loss_gold = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.range(batch_size, dtype=tf.int32), logits=discriminate_logits)
                    sent_loss_gold = tf.reduce_mean(sent_loss_gold)

                    loss_list = word_embed_loss_gold_list + word_select_loss_gold_list + word_gen_loss_gold_list + \
                                [sent_loss_gold]
                    loss_list = tf.stack(loss_list, axis=0)
                    return loss_list, encodes_gold_list, \
                           segmented_seqs_gold_list, word_masks_gold_list, word_ids_gold_list


            self.model = eval(model)

            self.loss_list, self.encodes_list, self.segmented_seqs_list, self.masks_list, self.word_ids_list = \
                self.model(self.pieces)
            if trainable:
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate)
                self.update = model_utils.optimize_loss(
                    tf.reduce_sum(self.loss_list),
                    self.global_step,
                    self.optimizer,
                    self.scope.trainable_variables(),
                    self.scope.name)

            total_params = 0
            for var in tf.trainable_variables():
                print(var)
                try:
                    local_params=1
                    shape = var.get_shape()  #getting shape of a variable
                    for i in shape:
                        local_params *= i.value  #mutiplying dimension values
                    total_params += local_params
                except:
                    continue
            print("total number of parameters is: {}".format(total_params))
            self.saver = tf.train.Saver()

        def wrapper(func):
            def wrapped_func(*args, **kwargs):
                with tf.variable_scope(self.scope):
                    return func(*args, **kwargs)
            return wrapped_func
        self.segment = wrapper(segment)
        self.embed_words = wrapper(embed_words_fn)
        self.encode_words = wrapper(encode_words)
        self.match_embeds = wrapper(match_embeds)
        self.train_generator = wrapper(train_generator)
        self.generate_word = wrapper(generate_word)

    def init(self, session, model_dir, kwargs):
        """Initialize the model graph.

        """

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path+'.meta'):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver.restore(session, ckpt.model_checkpoint_path)
            if kwargs.get('learning_rate') != None:
                session.run(self.max_learning_rate.assign(kwargs['learning_rate']))
            if kwargs.get('clr_period') != None:
                session.run(self.clr_period.assign(kwargs['clr_period']))
        else:
            print("Created model with fresh parameters. ")
            session.run(tf.variables_initializer(self.scope.global_variables()))

    def step(self, session, input_feed, output_feed,
             training=None, do_profiling=False):
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
        if type(self.training) != bool and training != None:
            input_feed[self.training] = training

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
