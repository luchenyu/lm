import math
import numpy as np
import tensorflow as tf
from utils import model_utils_py3


""" lm components """

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

def training_schedule(
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
    layer_size,
    training,
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
            char_vocab_emb = np.pad(np.array(char_vocab_emb), [[1,0],[0,0]], 'constant')
            char_vocab_initializer = tf.initializers.constant(char_vocab_emb)
        else:
            char_vocab_initializer = tf.initializers.variance_scaling(mode='fan_out')
        char_embedding = tf.get_variable(
            "char_embedding",
            shape=[char_vocab_size+1, char_vocab_dim], # _pad_ in char vocab is used in speller, so one more
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

        spellout_embedding = model_utils_py3.fully_connected(
            spellin_embedding,
            int(layer_size/2),
            is_training=training,
            scope="spellout")

        field_embedding = tf.get_variable(
            "field_embedding",
            shape=[10, int(layer_size/4)],
            dtype=tf.float32,
            initializer=tf.initializers.variance_scaling(mode='fan_out'),
            trainable=training,
            collections=collections,
            aggregation=tf.VariableAggregation.MEAN)

    return input_embedding, spellin_embedding, spellout_embedding, field_embedding

def segment_words(
    seqs,
    segs,
    reuse=None):
    """
    segment seqs according to segs
    args:
        seqs: batch_size x seq_length
        segs: batch_size x (seq_length + 1)
    """
    with tf.variable_scope("segmentor", reuse=reuse):

        batch_size = tf.shape(seqs)[0]
        length = tf.shape(seqs)[1]

        segmented_seqs_ref, segment_idxs_ref = model_utils_py3.slice_words(
            seqs, segs[:,1:-1], get_idxs=True)
        segmented_seqs_ref = tf.stop_gradient(segmented_seqs_ref)

    return segmented_seqs_ref

def embed_words(
    segmented_seqs,
    input_embedding,
    layer_size,
    dropout,
    training,
    reuse=None):
    """
    embed seq of words into vectors
    args:
        segmented_seqs: batch_size x word_length x char_length
        input_embedding: num_embedding x embedding_dim
        layer_size: size of layer
        dropout: dropout ratio
        training: bool
    """
    with tf.variable_scope("embedder", reuse=reuse):

        batch_size = tf.shape(segmented_seqs)[0]
        max_word_length = tf.shape(segmented_seqs)[1]
        max_char_length = tf.shape(segmented_seqs)[2]
        masks = tf.reduce_any(tf.not_equal(segmented_seqs, 0), axis=2)

        char_embeds = tf.nn.embedding_lookup(input_embedding, tf.maximum(segmented_seqs, 0))
        l1_embeds = model_utils_py3.convolution2d(
            char_embeds,
            [layer_size]*2,
            [[1,2],[1,3]],
            activation_fn=tf.nn.relu,
            is_training=training,
            scope="l1_convs")
        char_embeds = tf.nn.max_pool(char_embeds, [1,1,2,1], [1,1,2,1], padding='SAME')
        l2_embeds = model_utils_py3.convolution2d(
            char_embeds,
            [layer_size]*2,
            [[1,2],[1,3]],
            activation_fn=tf.nn.relu,
            is_training=training,
            scope="l2_convs")
        concat_embeds = tf.concat(
            [tf.reduce_max(tf.nn.relu(char_embeds), axis=2),
             tf.reduce_max(l1_embeds, axis=2),
             tf.reduce_max(l2_embeds, axis=2)],
            axis=-1)
        concat_embeds = model_utils_py3.highway(
            concat_embeds,
            2,
            activation_fn=tf.nn.relu,
            dropout=dropout,
            is_training=training,
            scope="highway")
        concat_embeds = model_utils_py3.layer_norm(
            concat_embeds, begin_norm_axis=-1,
            is_training=training)
        word_embeds = model_utils_py3.fully_connected(
            concat_embeds,
            layer_size,
            dropout=dropout,
            is_training=training,
            scope="projs")
        word_embeds_normed = model_utils_py3.layer_norm(
            word_embeds, begin_norm_axis=-1, is_training=training)
        word_embeds += model_utils_py3.MLP(
            tf.concat([concat_embeds, word_embeds_normed], axis=-1),
            2,
            2*layer_size,
            layer_size,
            dropout=dropout,
            is_training=training,
            scope="MLP")
        word_embeds = model_utils_py3.layer_norm(
            word_embeds, begin_norm_axis=-1,
            is_training=training)

        masksLeft = tf.pad(masks, [[0,0],[1,0]])[:,:-1]
        masksRight = tf.pad(masks, [[0,0],[0,1]])[:,1:]
        word_masks = tf.logical_or(masks, tf.logical_or(masksLeft, masksRight))

    return word_embeds, word_masks

def encode_words(
    field_embeds,
    posit_embeds,
    word_embeds,
    attn_masks,
    num_layers,
    dropout,
    training,
    reuse=None):
    """
    encode seq of words, include embeds and contexts
    args:
        field_embeds: batch_size x seq_length x dim
        value_embeds: batch_size x seq_length x dim
        attn_masks: batch_size x seq_length
        num_layers: num of layers
        dropout: dropout ratio
        training: bool
    """

    with tf.variable_scope("encoder", reuse=reuse):

        layer_size = word_embeds.get_shape()[-1].value
        encodes = model_utils_py3.transformer(
            field_embeds,
            posit_embeds,
            word_embeds,
            num_layers,
            layer_size,
            masks=attn_masks,
            dropout=dropout,
            is_training=training,
            scope="transformer")

    return encodes

def match_embeds(
    encodes,
    token_embeds,
    field_embeds,
    dropout,
    training,
    reuse=None):
    """
    outputs the degree of matchness of the word embeds and contexts
    args:
        encodes: batch_size x dim
        token_embeds: batch_size x num_candidates x dim or num_candidates x dim
        field_embeds: dim
        dropout: dropout ratio
        training: bool
    """

    with tf.variable_scope("matcher", reuse=reuse):

        dim = encodes.get_shape()[-1].value
        field_dim = field_embeds.get_shape()[-1].value
        encodes = tf.concat([encodes, field_embeds+tf.zeros(tf.concat([tf.shape(encodes)[:-1],[field_dim]], axis=0))], axis=-1)
        encode_projs = model_utils_py3.GLU(
            encodes,
            dim,
            dropout=dropout,
            is_training=training,
            scope="enc_projs")
        token_embeds = tf.concat([token_embeds, field_embeds+tf.zeros(tf.concat([tf.shape(token_embeds)[:-1],[field_dim]], axis=0))], axis=-1)
        token_embed_projs = model_utils_py3.GLU(
            token_embeds,
            dim,
            is_training=training,
            scope="tok_projs")

        if len(token_embeds.get_shape()) == 2:
            logits = tf.matmul(
                encode_projs, token_embed_projs, transpose_b=True)
        else:
            logits = tf.matmul(
                token_embed_projs, tf.expand_dims(encode_projs, axis=-1))
            logits = tf.squeeze(logits, axis=-1)
        logits /= tf.sqrt(float(dim))

    return logits

def train_speller(
    encodes,
    encMasks,
    targetSeqs,
    spellin_embedding,
    spellout_embedding,
    dropout,
    training,
    reuse=None):
    """
    get the training loss of speller
    args:
        encodes: batch_size x enc_seq_length x dim
        encMasks: batch_size x enc_seq_length
        targetSeqs: batch_size x dec_seq_length
        layer_size: size of the layer
        spellin_embedding: input embedding for spelling
        spellout_embedding: output embedding for spelling
        dropout: dropout ratio
        training: bool
    """

    with tf.variable_scope("speller", reuse=reuse):

        batch_size = tf.shape(encodes)[0]
        vocab_size = spellin_embedding.get_shape()[0].value
        input_dim = spellin_embedding.get_shape()[-1].value
        output_dim = spellout_embedding.get_shape()[-1].value

        attn_cell = model_utils_py3.AttentionCell(
            output_dim,
            num_layer=2,
            dropout=dropout,
            is_training=training)

        decInputs = tf.TensorArray(tf.float32, 0,
            dynamic_size=True, clear_after_read=False, infer_shape=False)
        encode_projs = model_utils_py3.fully_connected(
            encodes,
            input_dim,
            is_training=training,
            scope="enc_projs")
        initialState = (decInputs, encode_projs, encMasks)
        inputs = tf.nn.embedding_lookup(
            spellin_embedding,
            tf.pad(targetSeqs, [[0,0],[1,0]]))
        targetIds = tf.pad(targetSeqs, [[0,0],[0,1]])
        decMasks = tf.not_equal(tf.pad(targetSeqs, [[0,0],[0,1]]), 0)
        decMasks = tf.logical_or(decMasks, tf.pad(tf.not_equal(targetIds, 0), [[0,0],[1,0]])[:,:-1])
        decLength = tf.shape(inputs)[1]
        outputs, state = attn_cell(inputs, initialState)
        state[0].mark_used()
        logits = tf.matmul(
            tf.reshape(outputs, [batch_size*decLength, outputs.get_shape()[-1].value]),
            spellout_embedding,
            transpose_b=True) / tf.sqrt(float(output_dim))
        logits = tf.reshape(logits, [batch_size, decLength, vocab_size])
        weights = tf.cast(decMasks, tf.float32)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targetIds,
            logits=logits) * weights
        weights_sum = tf.reduce_sum(weights)
        loss = tf.cond(
            tf.greater(weights_sum, 0),
            lambda: tf.reduce_sum(losses) / weights_sum,
            lambda: tf.zeros([]))

    return loss

