import math
import numpy as np
import tensorflow as tf
from utils import model_utils_py3


"""handy functions"""

def get_optimizer(
    schedule,
    global_step,
    max_lr,
    num_steps,
    pct_start):
    """
    setting parameters for scheduling
    args:
        mode: tf.estimator mode
        max_lr: maximum learning rate
        num_steps: num_steps to train
        pct_start: % of increasing lr phase
    """
    with tf.variable_scope("scheduler"):

        if schedule == '1cycle':
            """ 1cycle schedule """
            max_lr = float(max_lr)
            num_steps = float(num_steps)
            pct_start = float(pct_start)
            x = (tf.cast(global_step, tf.float32) % num_steps) / num_steps
            learning_rate = tf.cond(
                tf.less(x, pct_start),
                lambda: 4e-2*max_lr + (max_lr - 4e-2*max_lr)*(x/pct_start),
                lambda: (tf.math.cos((x-pct_start)/(1.0-pct_start)*math.pi) + 1.0) * 0.5*max_lr)
            momentum = tf.cond(
                tf.less(x, pct_start),
                lambda: 0.95 + (0.85 - 0.95)*(x/pct_start),
                lambda: -0.05*tf.math.cos((x-pct_start)/(1.0-pct_start)*math.pi) + 0.9)
            optimizer = MyAdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.95,
                beta1_t=momentum,
                beta2=0.999)
        elif schedule == 'lr_finder':
            """lr range test"""
            x = (tf.cast(global_step, tf.float32) % float(num_steps)) / float(num_steps)
            log_lr = -7.0 + x*(1.0 - (-7.0))
            learning_rate = tf.pow(10.0, log_lr)
            optimizer = MyAdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.999)
        else:
            optimizer = MyAdamOptimizer(
                learning_rate=max_lr,
                beta1=0.9,
                beta2=0.999)

    return optimizer

def get_embeddings(
    char_vocab_size,
    char_vocab_dim,
    char_vocab_emb,
    num_layers,
    layer_size,
    training=True,
    reuse=None):
    """
    get various embedding matrix
    args:
        char_vocab_size: character vocab size
        char_vocab_dim: dimension of character embedding
        char_vocab_emb: pretrained character embedding
        layer_size: size of the layer
        training: bool
    """
    with tf.variable_scope("embedding", reuse=reuse):

        collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if training:
            collections.append(tf.GraphKeys.WEIGHTS)

        if not char_vocab_emb is None:
            char_vocab_initializer = tf.initializers.constant(char_vocab_emb)
        else:
            char_vocab_initializer = tf.initializers.variance_scaling(mode='fan_out')
        char_embedding = tf.get_variable(
            "char_embedding",
            shape=[char_vocab_size, char_vocab_dim], # _pad_ in char vocab is used in speller, so one more
            dtype=tf.float32,
            initializer=char_vocab_initializer,
            trainable=training,
            collections=collections,
            aggregation=tf.VariableAggregation.MEAN)
        input_embedding = tf.concat([tf.zeros([1, char_vocab_dim]), char_embedding[1:]], axis=0)

        spellin_embedding = model_utils_py3.MLP(
            char_embedding,
            2,
            layer_size,
            int(layer_size/2),
            is_training=training,
            scope="spellin")
        spellin_embedding = model_utils_py3.layer_norm(
            spellin_embedding, begin_norm_axis=-1, is_training=training)

        field_query_embedding = tf.get_variable(
            "field_query_embedding",
            shape=[64, num_layers*layer_size],
            dtype=tf.float32,
            initializer=tf.initializers.variance_scaling(mode='fan_out'),
            trainable=training,
            collections=collections,
            aggregation=tf.VariableAggregation.MEAN)
        field_key_embedding = tf.get_variable(
            "field_key_embedding",
            shape=[64, num_layers*layer_size],
            dtype=tf.float32,
            initializer=tf.initializers.variance_scaling(mode='fan_out'),
            trainable=training,
            collections=collections,
            aggregation=tf.VariableAggregation.MEAN)
        field_value_embedding = tf.get_variable(
            "field_value_embedding",
            shape=[64, num_layers*layer_size],
            dtype=tf.float32,
            initializer=tf.initializers.variance_scaling(mode='fan_out'),
            trainable=training,
            collections=collections,
            aggregation=tf.VariableAggregation.MEAN)

    return input_embedding, spellin_embedding, field_query_embedding, field_key_embedding, field_value_embedding

def segment_words(
    seqs,
    segs):
    """
    segment seqs according to segs
    args:
        seqs: batch_size x seq_length
        segs: batch_size x (seq_length + 1)
    """
    with tf.variable_scope("segment_words"):

        batch_size = tf.shape(seqs)[0]
        length = tf.shape(seqs)[1]

        segmented_seqs_ref, segment_idxs_ref = model_utils_py3.slice_words(
            seqs, segs[:,1:-1], get_idxs=True)
        segmented_seqs_ref = tf.stop_gradient(segmented_seqs_ref)

    return segmented_seqs_ref

def encode_features(
    encoder,
    feature_list,
    attn_matrix,
    extra_feature_list=[]):
    """
    encode the features and distribute the result
    """
    with tf.variable_scope("encode_features"):

        batch_size = tf.shape(feature_list[0]['token_embeds'])[0]
        attn_list = feature_list
        to_attn_list = feature_list + extra_feature_list
        input_keys = [
            'field_query_embeds', 'field_key_embeds', 'field_value_embeds', 'posit_embeds', 'token_embeds', 'masks',
        ]
        output_keys = [
            'querys', 'keys', 'values', 'encodes',
        ]

        # prepare attn_masks
        attn_masks = []
        feature_lengths = []
        for i, fi in enumerate(attn_list):
            attn_masks_local = []
            length_i = tf.shape(fi['token_embeds'])[1]
            feature_lengths.append(length_i)
            for j, fj in enumerate(to_attn_list):
                length_j = tf.shape(fj['token_embeds'])[1]
                if attn_matrix[i][j] == 0:
                    attn_masks_local.append(
                        tf.zeros([batch_size, length_i, length_j], dtype=tf.bool))
                else:
                    attn_masks_local.append(
                        tf.tile(tf.expand_dims(fj['masks'], axis=1), [1, length_i, 1]))
            attn_masks_local = tf.concat(attn_masks_local, axis=2)
            attn_masks.append(attn_masks_local)
        attn_masks = tf.concat(attn_masks, axis=1)

        # prepare concat_features
        if len(feature_list) <= 1:
            concat_features = feature_list[0]
        else:
            concat_features = {}
            for key in input_keys:
                concat_features[key] = tf.concat(
                    [f[key] for f in feature_list], axis=1)
        if len(extra_feature_list) == 0:
            extra_concat_features = None
        elif len(extra_feature_list) == 1:
            extra_concat_features = extra_feature_list[0]
        else:
            extra_concat_features = {}
            for key in input_keys:
                extra_concat_features[key] = tf.concat(
                    [f[key] for f in extra_feature_list], axis=1)

        # encode
        concat_features = encoder(concat_features, attn_masks, extra_concat_features)

        # split and distribute results
        if len(feature_list) > 1:
            for key in output_keys:
                splitted = tf.split(concat_features[key], feature_lengths, axis=1)
                for i, fi in enumerate(feature_list):
                    fi[key] = splitted[i]

    return feature_list

def get_speller_loss(
    word_encodes,
    target_seqs,
    spellin_embedding,
    speller_cell,
    speller_matcher):
    """
    get the loss of speller
    args:
        word_encodes: batch_size x word_layer_size
        target_seqs: batch_size x dec_seq_length
        spellin_embedding: input embedding for spelling
        speller_cell: speller cell
        speller_matcher: matcher module
    """
    with tf.variable_scope("train_speller"):

        batch_size = tf.shape(target_seqs)[0]

        initial_state = (word_encodes, None, None, None, None, None)
        inputs = tf.nn.embedding_lookup(
            spellin_embedding,
            tf.pad(target_seqs, [[0,0],[1,0]]))
        outputs, state = speller_cell(inputs, initial_state)
        target_seqs = tf.pad(target_seqs, [[0,0],[0,1]])
        decMasks = tf.not_equal(target_seqs, 0)
        decMasks = tf.logical_or(decMasks, tf.pad(tf.not_equal(target_seqs, 0), [[0,0],[1,0]])[:,:-1])
        _, valid_outputs = tf.dynamic_partition(
            outputs, tf.cast(decMasks, tf.int32), 2)
        valid_target_seqs = tf.boolean_mask(target_seqs, decMasks)
        valid_logits = speller_matcher(valid_outputs, spellin_embedding)
        valid_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=valid_target_seqs,
            logits=valid_logits)
        loss = tf.cond(
            tf.greater(tf.size(valid_losses), 0),
            lambda: tf.reduce_mean(valid_losses),
            lambda: tf.zeros([]))

    return loss


"""Cells"""

class GeneralCell(object):
    """
    General Cell that defines the interface
    dummy class
    """
    def __init__(self,
                 scope,
                 **kwargs):
        """
        args:
            cell_fn takes assets, inputs, state then produce outputs, state
            kwargs takes all the args and put in self.assets
        """
        self.assets = kwargs
        self.reuse = None
        with tf.variable_scope(scope) as sc:
            self.scope=sc

    def __call__(self,
                 inputs,
                 state):
        """
        args:
            inputs: batch_size x input_dim or batch_size x length x input_dim
            state:
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):
            outputs, state = inputs, state
        self.reuse = True
        return outputs, state

class TransformerCell(GeneralCell):
    """Cell that wraps transformer"""
    def __init__(self,
                 scope,
                 posit_size,
                 encoder,
                 dropout=None,
                 training=True):
        """
        args:
            posit_size
            encoder: module
            dropout
            training
        """

        GeneralCell.__init__(self,
                             scope,
                             posit_size=posit_size,
                             encoder=encoder,
                             dropout=dropout,
                             training=training)

    def __call__(self,
                 inputs,
                 state):
        """
        fn that defines attn_cell
        inputs: batch_size x token_size or batch_size x length x token_size
        state:
            field_query_embedding: batch_size x num_layers*layer_size
            field_key_embedding: batch_size x num_layers*layer_size
            field_value_embedding: batch_size x num_layers*layer_size
            dec_features: {
                'field_query_embeds': batch_size x length x num_layers*layer_size,
                'field_key_embeds': batch_size x length x num_layers*layer_size,
                'field_value_embeds': batch_size x length x num_layers*layer_size,
                'posit_embeds': batch_size x length x posit_size,
                'token_embeds': batch_size x length x token_size,
                'masks': batch_size x length,
                'querys': batch_size x length x num_layers*layer_size,
                'keys': batch_size x length x num_layers*layer_size,
                'values': batch_size x length x num_layers*layer_size,
                'encodes': batch_size x length x layer_size
            }
            enc_features
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):

            batch_size = tf.shape(inputs)[0]
            to_squeeze = False
            if inputs.shape.ndims == 2:
                to_squeeze = True
                inputs = tf.expand_dims(inputs, axis=1)
            length = tf.shape(inputs)[1]

            field_query_embedding, field_key_embedding, field_value_embedding = state[0], state[1], state[2]
            dec_features, enc_features = state[3], state[4]
            if dec_features is None:
                current_idx = 0
            else:
                current_idx = tf.shape(dec_features['token_embeds'])[1]

            # prepare features
            features = {}
            features['field_query_embeds'] = tf.tile(
                tf.expand_dims(field_query_embedding, axis=1),
                [1, 2*length, 1])
            features['field_key_embeds'] = tf.tile(
                tf.expand_dims(field_key_embedding, axis=1),
                [1, 2*length, 1])
            features['field_value_embeds'] = tf.tile(
                tf.expand_dims(field_value_embedding, axis=1),
                [1, 2*length, 1])
            features['posit_embeds'] = tf.tile(
                model_utils_py3.embed_position(
                    tf.expand_dims(tf.concat([tf.range(length), tf.range(1, length+1)], axis=0), axis=0) + \
                        current_idx,
                    self.assets['posit_size']),
                [batch_size, 1, 1])

            features['token_embeds'] = tf.concat([inputs, tf.zeros_like(inputs)], axis=1)
            features['masks'] = tf.pad(tf.reduce_any(tf.not_equal(inputs, 0.0), axis=-1), [[0,0], [0,length]])

            # prepare extra_features
            extra_features = []
            if not dec_features is None:
                extra_features.append(dec_features)
            if not enc_features is None:
                extra_features.append(enc_features)
            if len(extra_features) > 0:
                extra_features = model_utils_py3.concat_features(extra_features)
            else:
                extra_features = None

            # prepare masks
            attn_masks = tf.sequence_mask(
                 tf.tile(tf.expand_dims(tf.range(1, length+1),0), [batch_size,1]),
                 maxlen=2*length)
            attn_masks = tf.concat([attn_masks, attn_masks], axis=1)
            if not extra_features is None:
                attn_masks = tf.concat(
                    [attn_masks, tf.tile(tf.expand_dims(extra_features['masks'], axis=1), [1, 2*length, 1])],
                    axis=2)

            features = self.assets['encoder'](features, attn_masks, extra_features=extra_features)
            input_features, output_features = model_utils_py3.split_features(features, 2)
            outputs = output_features['encodes']
            if to_squeeze:
                outputs = tf.squeeze(outputs, axis=[1])
            if dec_features is None:
                dec_features = input_features
            else:
                dec_features = model_utils_py3.concat_features([dec_features, input_features])
            state = (field_query_embedding, field_key_embedding, field_value_embedding, dec_features, enc_features)

        outputs, state = GeneralCell.__call__(self, outputs, state)

        return outputs, state

class SpellerCell(TransformerCell):
    """wraps a speller"""
    def __call__(self,
                 inputs,
                 state):
        """
        first map from word_encodes to field_value_embedding
        inputs: batch_size x token_size or batch_size x length x token_size
        state:
            word_encodes: batch_size x word_layer_size
            field_query_embedding: batch_size x num_layers*layer_size
            field_key_embedding: batch_size x num_layers*layer_size
            field_value_embedding: batch_size x num_layers*layer_size
            dec_features: {
                'field_query_embeds': batch_size x length x num_layers*layer_size,
                'field_key_embeds': batch_size x length x num_layers*layer_size,
                'field_value_embeds': batch_size x length x num_layers*layer_size,
                'posit_embeds': batch_size x length x posit_size,
                'token_embeds': batch_size x length x token_size,
                'masks': batch_size x length,
                'querys': batch_size x length x num_layers*layer_size,
                'keys': batch_size x length x num_layers*layer_size,
                'values': batch_size x length x num_layers*layer_size,
                'encodes': batch_size x length x layer_size
            }
            enc_features
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):

            word_encodes = state[0]
            field_value_embedding = state[3]
            state = state[1:]
            if field_value_embedding is None:
                num_layers = self.assets['encoder'].num_layers
                layer_size = self.assets['encoder'].layer_size
                field_value_embedding = model_utils_py3.fully_connected(
                    word_encodes,
                    num_layers*layer_size,
                    dropout=self.assets['dropout'],
                    is_training=self.assets['training'],
                    scope="enc_projs")
                field_query_embedding = tf.zeros_like(field_value_embedding)
                field_key_embedding = tf.zeros_like(field_value_embedding)
                state = (
                    field_query_embedding, field_key_embedding, field_value_embedding,
                    state[3], state[4])

        outputs, state = TransformerCell.__call__(self, inputs, state)
        state = tuple([word_encodes] + list(state))

        return outputs, state


"""Modules"""

class Module(object):
    """
    maintains module parameters, scope and reuse state
    """
    def __init__(self, scope, dropout=None, training=False):
        self.reuse = None
        self.dropout = dropout
        self.training = training
        with tf.variable_scope(scope) as sc:
            self.scope = sc

class Embedder(Module):
    """
    Word embedder
    embed seqs of chars into word embeddings
    """
    def __init__(self, input_embedding, layer_size,
                 scope, dropout=None, training=False):
        """
        args:
            input_embedding: num_embedding x embedding_dim
            layer_size: size of layer
        """
        self.input_embedding = input_embedding
        self.layer_size = layer_size
        Module.__init__(self, scope, dropout, training)

    def __call__(self, segmented_seqs):
        """
        args:
            segmented_seqs: batch_size x word_length x char_length
        returns:
            word_embeds: batch_size x word_length x layer_size
            word_masks: batch_size x word_length
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):

            batch_size = tf.shape(segmented_seqs)[0]
            max_word_length = tf.shape(segmented_seqs)[1]
            max_char_length = tf.shape(segmented_seqs)[2]
            word_masks = tf.reduce_any(tf.not_equal(segmented_seqs, 0), axis=2)

            char_embeds = tf.nn.embedding_lookup(self.input_embedding, segmented_seqs)
            l0_embeds = model_utils_py3.fully_connected(
                char_embeds,
                self.layer_size,
                activation_fn=tf.nn.relu,
                is_training=self.training,
                scope="l0_convs")
            l1_embeds = model_utils_py3.convolution2d(
                char_embeds,
                [self.layer_size]*2,
                [[1,2],[1,3]],
                activation_fn=tf.nn.relu,
                is_training=self.training,
                scope="l1_convs")
            char_embeds = tf.nn.max_pool(char_embeds, [1,1,2,1], [1,1,2,1], padding='SAME')
            l2_embeds = model_utils_py3.convolution2d(
                char_embeds,
                [self.layer_size]*2,
                [[1,2],[1,3]],
                activation_fn=tf.nn.relu,
                is_training=self.training,
                scope="l2_convs")
            concat_embeds = tf.concat(
                [tf.reduce_max(l0_embeds, axis=2),
                 tf.reduce_max(l1_embeds, axis=2),
                 tf.reduce_max(l2_embeds, axis=2)],
                axis=-1)
            concat_embeds_normed = model_utils_py3.layer_norm(
                concat_embeds, begin_norm_axis=-1, is_training=self.training)
            word_embeds = model_utils_py3.fully_connected(
                concat_embeds_normed,
                self.layer_size,
                dropout=self.dropout,
                is_training=self.training,
                scope="projs")
            word_embeds_normed = model_utils_py3.layer_norm(
                word_embeds, begin_norm_axis=-1, is_training=self.training)
            word_embeds += model_utils_py3.MLP(
                word_embeds_normed,
                2,
                2*self.layer_size,
                self.layer_size,
                dropout=self.dropout,
                is_training=self.training,
                scope="MLP")
            word_embeds = model_utils_py3.layer_norm(
                word_embeds, begin_norm_axis=-1,
                is_training=self.training)

            word_embeds *= tf.expand_dims(tf.cast(word_masks, tf.float32), axis=2)

        self.reuse = True

        return word_embeds, word_masks

class Encoder(Module):
    """
    Transformer encoder
    """
    def __init__(self, layer_size, num_layers, num_heads,
                 scope, dropout=None, training=False):
        """
        args:
            layer_size: size of layer
            num_layers: num of layers
            num_heads: num of attention heads
        """
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        Module.__init__(self, scope, dropout, training)

    def __call__(self, features, attn_masks, extra_features=None):
        """
        args:
            features: {
                'field_query_embeds': batch_size x length x num_layers*layer_size,
                'field_key_embeds': batch_size x length x num_layers*layer_size,
                'field_value_embeds': batch_size x length x num_layers*layer_size,
                'posit_embeds': batch_size x length x layer_size,
                'token_embeds': batch_size x length x layer_size,
                'masks': batch_size x length,
                'querys': batch_size x length x num_layers*layer_size,
                'keys': batch_size x length x num_layers*layer_size,
                'values': batch_size x length x num_layers*layer_size,
                'encodes': batch_size x length x layer_size
            }
            attn_masks, batch_size x length x (length+extra_length)
        returns:
            features
        """
        features = model_utils_py3.transformer(
            features,
            self.num_layers,
            self.layer_size,
            extra_features=extra_features,
            num_heads=self.num_heads,
            attn_masks=attn_masks,
            dropout=self.dropout,
            is_training=self.training,
            reuse=self.reuse,
            scope=self.scope)

        self.reuse = True

        return features

class Matcher(Module):
    """
    Matcher to get the logits of output probs
    """
    def __init__(self, layer_size,
                 scope, dropout=None, training=False):
        """
        args:
            layer_size: size of layer
            has_copy: whether has copy mechanism
        """
        self.layer_size = layer_size
        self.cached = {}
        Module.__init__(self, scope, dropout, training)

    def __call__(self, encodes, token_embeds):
        """
        args:
            encodes: batch_size x dim or batch_size x length x dim
            token_embeds: num_candidates x token_dim or batch_size x num_candidates x token_dim
        returns:
            logits: batch_size x num_candidates or batch_size x length x num_candidates
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):

            encode_projs = model_utils_py3.GLU(
                encodes,
                self.layer_size,
                dropout=self.dropout,
                is_training=self.training,
                scope="enc_projs")
            encode_projs *= tf.sqrt(1.0/float(self.layer_size))

            if self.cached.get(token_embeds) != None:
                token_projs = self.cached[token_embeds]
            else:
                token_projs = model_utils_py3.fully_connected(
                    token_embeds,
                    self.layer_size,
                    is_training=self.training,
                    scope="tok_projs")
                self.cached[token_embeds] = token_projs

            logits = tf.matmul(
                encode_projs, token_projs, transpose_b=True)

        self.reuse = True

        return logits


""" Optimizers """

class MyAdamOptimizer(tf.train.AdamOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta1_t=None, beta2=0.999, epsilon=1e-8,
                 use_locking=False, name="Adam"):
        """beta1 is the initial momentum, beta1_t is the dynamic momentum"""
        tf.train.AdamOptimizer.__init__(
            self, learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon,
            use_locking=use_locking, name=name)
        self.beta1_t = beta1_t

    def _prepare(self):
        tf.train.AdamOptimizer._prepare(self)
        if self.beta1_t != None:
            self._beta1_t = self.beta1_t
