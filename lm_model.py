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
                 decoder_type="attn",
                 loss_type="sup",
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
        self.size = size
        self.num_layers = num_layers
        self.dropout = dropout


        with tf.variable_scope(scope, reuse=reuse) as global_scope:

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
            x = (tf.to_float(self.global_step) % clr_period) / clr_period
            learning_rate = tf.cond(
                tf.less(x, ratio),
                lambda: 1e-1*learning_rate + (learning_rate - 1e-1*learning_rate)*(x/ratio),
                lambda: learning_rate + (1e-1*learning_rate - learning_rate)*(x-ratio)/(1.0-ratio))
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

            """ lm components """
            def segment(seqs, num_sample=2, reuse=None):
                """
                segment seqs
                """
                with tf.variable_scope("seg", reuse=reuse):

                    lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                        num_layers=1,
                        num_units=self.size,
                        direction="bidirectional")

                    padded_seqs = tf.pad(seqs, [[0,0], [1,1]])
                    masks = tf.greater(padded_seqs, 0)
                    seg_masks_and = tf.logical_or(masks[:,:-1], masks[:,1:])
                    seg_masks_or = tf.logical_xor(masks[:,:-1], masks[:,1:])
                    seg_weights = tf.to_float(tf.logical_and(masks[:,:-1], masks[:,1:]))
                    inputs = tf.nn.embedding_lookup(self.input_embedding, padded_seqs)
                    inputs = tf.transpose(inputs, [1, 0, 2])
                    outputs, states = lstm(inputs, training=trainable)
                    padded_char_encodes = tf.transpose(outputs, [1, 0, 2])
                    padded_char_encodes *= tf.expand_dims(tf.to_float(masks), axis=-1)
                    char_encodes = padded_char_encodes[:,1:-1]
                    adjacent_encodes = tf.concat([padded_char_encodes[:,:-1], padded_char_encodes[:,1:]], axis=-1)
                    seg_logits = model_utils.MLP(
                        adjacent_encodes,
                        2,
                        self.size,
                        1,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="seg_logits")
                    seg_logits = tf.squeeze(seg_logits, -1)
                    seg_probs = tf.sigmoid(seg_logits)
                    seg_predicts = []
                    for _ in range(num_sample):
                        seg = tf.greater(seg_probs, tf.random_uniform(tf.shape(seg_probs)))
                        seg = tf.logical_or(tf.logical_and(seg, seg_masks_and), seg_masks_or)
                        seg_predicts.append(tf.to_float(seg))
                return seg_logits, seg_weights, seg_predicts

            def embed_words(segmented_seqs, reuse=None):
                """
                embed seq of words into vectors
                """
                with tf.variable_scope("word_embedder", reuse=reuse):

                    batch_size = tf.shape(segmented_seqs)[0]
                    max_word_length = tf.shape(segmented_seqs)[1]
                    max_char_length = tf.shape(segmented_seqs)[2]
                    masks = tf.greater(segmented_seqs, 0)

                    inputs = tf.nn.embedding_lookup(self.input_embedding, segmented_seqs)
                    cnn_outputs = model_utils.convolution2d(
                        inputs,
                        [self.size]*4,
                        [[1,1],[1,2],[1,3],[1,4]],
                        activation_fn=tf.nn.relu,
                        dropout=1.0-self.dropout,
                        is_training=self.training,
                        scope="convs")
                    cnn_outputs = tf.reduce_max(cnn_outputs, axis=2)
                    word_embeds = model_utils.highway(
                        cnn_outputs,
                        2,
                        activation_fn=tf.nn.relu,
                        is_training=self.training,
                        scope="highway")
                    word_embeds = model_utils.fully_connected(
                        word_embeds,
                        2*self.vocab_dim,
                        is_training=self.training,
                        scope="projs")
                    word_embeds = tf.contrib.layers.layer_norm(word_embeds, begin_norm_axis=-1)

                    weights = tf.to_float(tf.reduce_any(masks, axis=2, keepdims=True))
                    word_embeds *= weights
                return word_embeds

            def encode_words(segmented_seqs, reuse=None):
                """
                encode seq of words, include embeds and contexts
                segmented_seqs: batch_size x max_length x max_char_length
                """

                batch_size = tf.shape(segmented_seqs)[0]
                max_length = tf.shape(segmented_seqs)[1]
                masks = tf.reduce_any(tf.greater(segmented_seqs, 0), axis=-1)
                padded_seqs = tf.pad(segmented_seqs, [[0,0],[1,1],[0,0]])
                padded_word_embeds = embed_words(padded_seqs, reuse=reuse)
                word_embeds = padded_word_embeds[:,1:-1]
                if block_type == "lstm":
                    lstm_encode = tf.contrib.cudnn_rnn.CudnnLSTM(
                        num_layers=1,
                        num_units=self.vocab_dim,
                        direction="unidirectional")
                    word_encodes = []
                    inputs_fw = padded_word_embeds
                    inputs_bw = padded_word_embeds
                    with tf.variable_scope("encoder", reuse=reuse):
                        for i in range(self.num_layers):
                            with tf.variable_scope("layer_{:d}_fw".format(i)):
                                inputs_fw = tf.transpose(inputs_fw, [1,0,2])
                                outputs, states = lstm_encode(inputs_fw, training=trainable)
                                inputs_fw += outputs
                                inputs_fw = tf.transpose(inputs_fw, [1,0,2])
                            with tf.variable_scope("layer_{:d}_bw".format(i)):
                                inputs_bw = tf.transpose(tf.reverse(inputs_bw, axis=[1]), [1,0,2])
                                outputs, states = lstm_encode(inputs_bw, training=trainable)
                                inputs_bw += outputs
                                inputs_bw = tf.reverse(tf.transpose(inputs_bw, [1,0,2]), axis=[1])
                            word_encodes.append(tf.concat([inputs_fw, inputs_bw], axis=-1))
                        contexts = tf.concat([inputs_fw[:,:-2], inputs_bw[:,2:]], axis=-1)
                        contexts = model_utils.fully_connected(
                            contexts,
                            self.vocab_dim,
                            is_training=self.training,
                            scope="context_projs")
                        word_encodes = tf.stack(word_encodes, axis=2)
                        word_encodes = word_encodes[:,1:-1]
                elif block_type == "transformer":
                    word_encodes = []
                    inputs_fw = padded_word_embeds
                    inputs_bw = padded_word_embeds
                    padded_masks = tf.pad(masks, [[0,0],[1,1]])
                    attn_masks = tf.logical_and(
                        tf.tile(tf.expand_dims(padded_masks, 1), [1,tf.shape(padded_masks)[1],1]),
                        tf.expand_dims(padded_masks, -1))
                    with tf.variable_scope("encoder", reuse=reuse):
                        for i in range(self.num_layers):
                            with tf.variable_scope("layer_{:d}_fw".format(i)):
                                inputs_fw = model_utils.transformer(
                                    inputs_fw,
                                    1,
                                    attn_masks,
                                    is_training=self.training,
                                    scope="transformer")
                            with tf.variable_scope("layer_{:d}_bw".format(i)):
                                inputs_bw = model_utils.transformer(
                                    tf.reverse(inputs_bw, axis=[1]),
                                    1,
                                    tf.reverse(attn_masks, axis=[1]),
                                    is_training=self.training,
                                    scope="transformer")
                                inputs_bw = tf.reverse(inputs_bw, axis=[1])
                            word_encodes.append(tf.concat([inputs_fw, inputs_bw], axis=-1))
                        contexts = tf.concat([inputs_fw[:,:-2], inputs_bw[:,2:]], axis=-1)
                        contexts = model_utils.fully_connected(
                            contexts,
                            self.vocab_dim,
                            is_training=self.training,
                            scope="context_projs")
                        contexts = tf.contrib.layers.layer_norm(contexts, begin_norm_axis=-1)
                        word_encodes = tf.stack(word_encodes, axis=2)
                        word_encodes = word_encodes[:,1:-1]
                elif block_type == "transformer2":
                    word_encodes = []
                    attn_masks = tf.tile(tf.expand_dims(masks, 1), [1,max_length,1])
                    attn_masks = tf.logical_and(attn_masks, tf.expand_dims(masks, -1))
                    inputs = tf.zeros(tf.shape(word_embeds))
                    with tf.variable_scope("encoder", reuse=reuse):
                        for i in range(self.num_layers):
                            with tf.variable_scope("layer_{:d}".format(i)):
                                inputs = model_utils.transformer2(
                                    inputs,
                                    word_embeds,
                                    1,
                                    attn_masks,
                                    dropout=1.0-self.dropout,
                                    is_training=self.training,
                                    scope="transformer")
                                inputs *= tf.expand_dims(tf.to_float(masks), axis=-1)
                            word_encodes.append(tf.concat([word_embeds, inputs], axis=-1))
                        contexts = inputs
                        word_encodes = tf.stack(word_encodes, axis=2)
                return word_embeds, contexts, word_encodes

            """ lm model """
            def multi_lm(seqs, segs, pos_labels):

                batch_size = tf.shape(seqs)[0]

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
                    output_embedding = tf.contrib.layers.layer_norm(output_embedding, begin_norm_axis=-1)
                    self.output_embedding = output_embedding
                    pos_embedding = tf.get_variable(
                        "pos_embedding",
                        shape=[self.num_pos_tags, size],
                        dtype=tf.float32,
                        initializer=tf.initializers.truncated_normal(0.0, 0.01),
                        trainable=trainable,
                        collections=collections)
                    pos_encode_weights = tf.get_variable(
                        "pos_encode_weights",
                        shape=[self.num_layers,],
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


                num_sample = 2
                seg_logits, seg_weights, seg_predicts = segment(seqs, num_sample=num_sample)
                if segs != None:
                    seg_loss_gold = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=segs,
                        logits=seg_logits)
                    seg_loss_gold = tf.reduce_sum(seg_loss_gold*seg_weights) / tf.reduce_sum(seg_weights)
                    segmented_seqs_gold, segment_idxs_gold = model_utils.slice_words(
                        seqs, segs[:,1:-1], get_idxs=True)
                    segmented_seqs_gold = tf.stop_gradient(segmented_seqs_gold)
                    max_char_length = tf.shape(segmented_seqs_gold)[2]
                    segmented_seqs_gold = tf.cond(
                        tf.less(max_char_length, 10),
                        lambda: segmented_seqs_gold,
                        lambda: segmented_seqs_gold[:,:,:10])
                    seg_predicts.append(segs)
                    num_sample += 1
                else:
                    seg_loss_gold = 0.0
                    segmented_seqs_gold = None
                seg_predicts = tf.concat(seg_predicts, axis=0)
                segmented_seqs_hyp, segment_idxs_hyp = model_utils.slice_words(
                    tf.tile(seqs, [num_sample,1]), seg_predicts[:,1:-1], get_idxs=True)
                segmented_seqs_hyp = tf.stop_gradient(segmented_seqs_hyp)
                max_char_length = tf.shape(segmented_seqs_hyp)[2]
                segmented_seqs_hyp = tf.cond(
                    tf.less(max_char_length, 10),
                    lambda: segmented_seqs_hyp,
                    lambda: segmented_seqs_hyp[:,:,:10])
                seg_loss_hyp = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.stop_gradient(seg_predicts),
                    logits=tf.tile(seg_logits, [num_sample,1]))
                seg_loss_hyp *= tf.tile(seg_weights, [num_sample,1])
                seg_loss_hyp = tf.stack(tf.split(tf.reduce_sum(seg_loss_hyp, axis=1), num_sample, axis=0), axis=1)


                lstm_decoder = tf.contrib.cudnn_rnn.CudnnLSTM(
                    num_layers=1,
                    num_units=size,
                    direction="unidirectional")
                attn_decoder = model_utils.AttentionCell(
                    size,
                    num_layer=self.num_layers,
                    dropout=1.0-dropout,
                    is_training=self.training)

                def logit_fn(outputs):
                    logits = tf.matmul(outputs, output_embedding, transpose_b=True)
                    return logits

                def get_char_loss(contexts, seqs, segment_idxs, reuse=None):
                    with tf.variable_scope("generator", reuse=reuse):
                        batch_size = tf.shape(seqs)[0]
                        max_word_seq_length = tf.shape(contexts)[1]
                        weights = tf.pad(
                            tf.to_float(tf.greater(seqs, 0)),
                            [[0,0],[1,0]], constant_values=1.0)
                        weights_per_sample = tf.reduce_sum(weights, axis=[1])
                        seqs = tf.pad(seqs, [[0,0],[1,1]])
                        idxs = tf.pad(segment_idxs, [[0,0],[0,1]], constant_values=-1)
                        length = tf.shape(idxs)[1]

                        def not_slice():
                            inputs = tf.nn.embedding_lookup(input_embedding, seqs[:,:-1])
                            labels = seqs[:,1:]
                            encode_masks = tf.one_hot(idxs, max_word_seq_length, dtype=tf.int32)
                            encode_masks = tf.cast(encode_masks, tf.bool)
                            if decoder_type == "attn":
                                dec_inputs = tf.TensorArray(tf.float32, 0,
                                    dynamic_size=True, clear_after_read=False, infer_shape=False)
                                initial_state = (dec_inputs, contexts, encode_masks)
                                outputs, state = attn_decoder(inputs, initial_state)
                            outputs = tf.reshape(outputs, [batch_size*length, size])
                            logits = logit_fn(outputs)
                            logits = tf.reshape(
                                logits, [batch_size, length, self.vocab_size])
                            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=labels, logits=logits)
                            loss_per_sample_a = tf.reduce_sum(losses*weights, axis=1)
                            return loss_per_sample_a

                        def to_slice():
                            pad_num = tf.mod(10 - tf.mod(length, 10), 10)
                            padded_seqs = tf.pad(seqs, [[0,0],[0,pad_num]])
                            padded_idxs = tf.pad(idxs, [[0,0],[0,pad_num]], constant_values=-1)
                            padded_weights = tf.pad(weights, [[0,0],[0,pad_num]])
                            new_length = tf.shape(padded_idxs)[1]
                            num_splits = tf.div(tf.to_int32(new_length), tf.to_int32(10))

                            weights_splitted = tf.reshape(
                                padded_weights, [batch_size, num_splits, 10])
                            masks = tf.greater(
                                tf.reduce_sum(weights_splitted, axis=2), 0)
                            gathered_weights = tf.boolean_mask(weights_splitted, masks)
                            idxs_splitted = tf.reshape(
                                padded_idxs, [batch_size, num_splits, 10])
                            idxs_starts = tf.maximum(idxs_splitted[:,:,0], 0)
                            idxs_ends = tf.reduce_max(idxs_splitted, axis=2)
                            idxs_lengths = idxs_ends - idxs_starts + 1
                            contexts_splitted = model_utils.slice_fragments(
                                contexts, idxs_starts, idxs_lengths)
                            gathered_contexts = tf.boolean_mask(contexts_splitted, masks)
                            max_word_seq_length = tf.shape(gathered_contexts)[1]
                            idxs_splitted -= tf.maximum(idxs_splitted[:,:,0:1], 0)
                            gathered_idxs = tf.boolean_mask(idxs_splitted, masks)
                            encode_masks = tf.one_hot(
                                gathered_idxs, max_word_seq_length, dtype=tf.int32)
                            encode_masks = tf.cast(encode_masks, tf.bool)
                            batch_ids_splitted = tf.tile(
                                tf.expand_dims(tf.range(batch_size), axis=1), [1, num_splits])
                            batch_ids = tf.boolean_mask(batch_ids_splitted, masks)
                            labels_splitted = tf.reshape(
                                padded_seqs[:,1:], [batch_size, num_splits, 10])
                            gathered_labels = tf.boolean_mask(labels_splitted, masks)
                            inputs = tf.nn.embedding_lookup(input_embedding, padded_seqs[:,:-1])
                            inputs_splitted = tf.reshape(
                                inputs, [batch_size, num_splits, 10, self.vocab_dim])
                            gathered_inputs = tf.boolean_mask(inputs_splitted, masks)
                            prev_inputs_splitted = tf.pad(
                                inputs_splitted, [[0,0],[1,0],[0,0],[0,0]])[:,:-1]
                            prev_inputs = tf.boolean_mask(prev_inputs_splitted, masks)
                            prev_inputs = tf.pad(prev_inputs, [[0,0],[1,0],[0,0]])
                            if decoder_type == "attn":
                                dec_inputs = tf.TensorArray(tf.float32, 0,
                                    dynamic_size=True, clear_after_read=False, infer_shape=False)
                                dec_inputs = dec_inputs.write(0, prev_inputs)
                                initial_state = (dec_inputs, gathered_contexts, encode_masks)
                                outputs, state = attn_decoder(gathered_inputs, initial_state,
                                    reuse=True)
                            outputs = tf.reshape(outputs, [-1, size])
                            logits = logit_fn(outputs)
                            logits = tf.reshape(
                                logits, [-1, 10, self.vocab_size])
                            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=gathered_labels, logits=logits)
                            losses = tf.reduce_sum(losses*gathered_weights, axis=1)
                            loss_per_sample_b = tf.reshape(tf.segment_sum(losses, batch_ids), [batch_size])
                            return loss_per_sample_b

                        loss_per_sample = tf.cond(
                            tf.less(length, 30),
                            not_slice,
                            to_slice)

                    return loss_per_sample, weights_per_sample

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

                def get_word_loss(contexts, segmented_seqs):
                    batch_size = tf.shape(contexts)[0]
                    max_word_length = tf.shape(contexts)[1]
                    max_char_length = tf.shape(segmented_seqs)[2]
                    masks = tf.reduce_any(tf.greater(segmented_seqs, 0), axis=-1)
                    sample_seqs, _ = decode(contexts, max_char_length, 10, reuse=True)
                    logit_masks = tf.equal(tf.expand_dims(segmented_seqs, 2), sample_seqs)
                    logit_masks = tf.reduce_all(logit_masks, axis=-1)
                    logit_masks = tf.pad(logit_masks, [[0,0],[0,0],[1,0]])
                    sample_seqs = tf.concat(
                        [tf.expand_dims(segmented_seqs, 2), sample_seqs], axis=2)
                    sample_seqs = tf.reshape(sample_seqs, [batch_size, max_word_length*11, max_char_length])
                    sample_embeds = embed_words(sample_seqs, reuse=True)
                    with tf.variable_scope("matcher"):
                        contexts = model_utils.fully_connected(
                            contexts,
                            size,
                            is_training=self.training,
                            scope="context_projs")
                        sample_embeds = model_utils.fully_connected(
                            sample_embeds,
                            size,
                            is_training=self.training,
                            scope="embed_projs")
                        sample_embeds = tf.reshape(sample_embeds, [batch_size, max_word_length, 11, size])
                        logits = tf.matmul(sample_embeds, tf.expand_dims(contexts, axis=-1))
                        logits = tf.squeeze(logits, axis=-1)
                        logits -= 1e6 * tf.to_float(logit_masks)
                        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.zeros([batch_size, max_word_length], dtype=tf.int32), logits=logits)
                        weights = tf.to_float(masks)
                        loss = tf.reduce_sum(losses*weights) / tf.reduce_sum(weights)
                    return loss

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

                def get_pos_logits(word_encodes):
                    batch_size = tf.shape(word_encodes)[0]
                    max_length = tf.shape(word_encodes)[1]
                    word_encodes *= tf.reshape(pos_encode_weights, [1,1,self.num_layers,1])
                    word_encodes = tf.reduce_sum(word_encodes, axis=2)
                    pos_outputs = model_utils.fully_connected(
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

                if loss_type == 'unsup' or segmented_seqs_gold == None:                        
                    word_embeds_hyp, contexts_hyp, word_encodes_hyp = encode_words(
                        segmented_seqs_hyp)
                    lm_char_loss_per_sample_hyp, lm_char_weights_per_sample_hyp = get_char_loss(contexts_hyp, tf.tile(seqs, [num_sample,1]), segment_idxs_hyp)
                    lm_char_loss_per_sample_hyp = tf.stack(tf.split(lm_char_loss_per_sample_hyp, num_sample, axis=0), axis=1)
                    lm_char_weights_per_sample_hyp = tf.stack(
                        tf.split(lm_char_weights_per_sample_hyp, num_sample, axis=0), axis=1)
                    idx = tf.argmin(lm_char_loss_per_sample_hyp, axis=-1, output_type=tf.int32)
                    idx = tf.stack([tf.range(batch_size), idx], axis=1)
                    seg_loss_hyp = tf.reduce_sum(tf.reshape(tf.gather_nd(seg_loss_hyp, idx), [batch_size])) / \
                        tf.reduce_sum(seg_weights)
                    lm_char_loss_hyp = tf.reduce_sum(
                        tf.reshape(tf.gather_nd(lm_char_loss_per_sample_hyp, idx), [batch_size]))
                    lm_char_loss_hyp /= tf.reduce_sum(
                        tf.reshape(tf.gather_nd(lm_char_weights_per_sample_hyp, idx), [batch_size]))
                    segmented_seqs_hyp = tf.stack(tf.split(segmented_seqs_hyp, num_sample, axis=0), axis=1)
                    max_word_length = tf.shape(segmented_seqs_hyp)[2]
                    max_char_length = tf.shape(segmented_seqs_hyp)[3]
                    segmented_seqs_hyp = tf.reshape(
                        tf.gather_nd(segmented_seqs_hyp, idx), [batch_size, max_word_length, max_char_length])
                    contexts_hyp = tf.stack(tf.split(contexts_hyp, num_sample, axis=0), axis=1)
                    contexts_hyp = tf.reshape(
                        tf.gather_nd(contexts_hyp, idx), [batch_size, max_word_length, 2*self.vocab_dim])
                    word_encodes_hyp = tf.stack(tf.split(word_encodes_hyp, num_sample, axis=0), axis=1)
                    word_encodes_hyp = tf.reshape(
                        tf.gather_nd(word_encodes_hyp, idx),
                        [batch_size, max_word_length, self.num_layers, 4*self.vocab_dim])
                    segmented_seq_masks_hyp = tf.reduce_any(tf.greater(segmented_seqs_hyp, 0), axis=-1)
                    segmented_seq_lengths_hyp = tf.reduce_sum(tf.to_int32(segmented_seq_masks_hyp), axis=-1)
                   # pos_logits_hyp = get_pos_logits(word_encodes_hyp)
                   # viterbi_tags, viterbi_scores = tf.contrib.crf.crf_decode(
                   #     pos_logits_hyp, transition_params, segmented_seq_lengths_hyp)
                   # lm_word_loss_hyp = get_word_loss(contexts_hyp, segmented_seqs_hyp)

                    tf.get_variable_scope().reuse_variables()
                    sample_seqs = sample(segmented_seqs_hyp)

                    loss = seg_loss_hyp+lm_char_loss_hyp
                    return loss, word_encodes_hyp, segmented_seqs_hyp, seg_predicts, None, sample_seqs

                else:
                    word_embeds_gold, contexts_gold, word_encodes_gold = encode_words(
                        segmented_seqs_gold)
                    lm_char_loss_per_sample_gold, lm_char_weights_per_sample_gold = get_char_loss(
                        contexts_gold, seqs, segment_idxs_gold)
                    lm_char_loss_gold = tf.reduce_sum(lm_char_loss_per_sample_gold) / \
                        tf.reduce_sum(lm_char_weights_per_sample_gold)
                   # lm_word_loss_gold = get_word_loss(contexts_gold, segmented_seqs_gold)
                   # if pos_labels != None:
                   #     segmented_seq_masks_gold = tf.reduce_any(tf.greater(segmented_seqs_gold, 0), axis=-1)
                   #     segmented_seq_lengths_gold = tf.reduce_sum(tf.to_int32(segmented_seq_masks_gold), axis=-1)
                   #     pos_logits_gold = get_pos_logits(word_encodes_gold)
                   #     log_probs, _ = tf.contrib.crf.crf_log_likelihood(
                   #         pos_logits_gold, pos_labels, segmented_seq_lengths_gold,
                   #         transition_params=transition_params)
                   #     pos_loss = tf.reduce_mean(-log_probs)

                    loss = seg_loss_gold+lm_char_loss_gold
                    return loss, word_encodes_gold, segmented_seqs_gold, segs, pos_labels, seqs


            model = multi_lm

            with tf.variable_scope("model") as self.scope:
                self.loss, self.encodes, self.segmented_seqs, self.segs, self.pos_labels, self.sample_seqs = model(
                    self.seqs, segs, pos_labels)
            if trainable:
                self.trainable_variables = global_scope.trainable_variables()
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate)
                self.update = model_utils.optimize_loss(
                    self.loss,
                    self.global_step,
                    self.optimizer,
                    self.trainable_variables)

            self.global_variables = global_scope.global_variables()
            self.saver = tf.train.Saver(self.global_variables)

        def wrapper(func):
            def wrapped_func(*args, **kwargs):
                with tf.variable_scope(self.scope):
                    return func(*args, **kwargs)
            return wrapped_func
        self.segment = wrapper(segment)
        self.embed_words = wrapper(embed_words)
        self.encode_words = wrapper(encode_words)

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
