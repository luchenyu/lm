import math
import numpy as np
import tensorflow as tf
from collections import namedtuple
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

def encode_tfstructs(
    encoder,
    tfstruct_list,
    attn_matrix,
    extra_tfstruct_list=[]):
    """
    encode the tfstructs and distribute the result
    """
    with tf.variable_scope("encode_tfstructs"):

        batch_size = tf.shape(tfstruct_list[0].token_embeds)[0]
        attn_list = tfstruct_list
        to_attn_list = tfstruct_list + extra_tfstruct_list

        # prepare attn_masks
        attn_masks = []
        tfstruct_lengths = []
        for i, fi in enumerate(attn_list):
            attn_masks_local = []
            length_i = tf.shape(fi.token_embeds)[1]
            tfstruct_lengths.append(length_i)
            for j, fj in enumerate(to_attn_list):
                length_j = tf.shape(fj.token_embeds)[1]
                if attn_matrix[i][j] == 0:
                    attn_masks_local.append(
                        tf.zeros([batch_size, length_i, length_j], dtype=tf.bool))
                else:
                    attn_masks_local.append(
                        tf.tile(tf.expand_dims(fj.masks, axis=1), [1, length_i, 1]))
            attn_masks_local = tf.concat(attn_masks_local, axis=2)
            attn_masks.append(attn_masks_local)
        attn_masks = tf.concat(attn_masks, axis=1)

        # prepare concat_tfstruct
        concat_tfstruct = model_utils_py3.concat_tfstructs(tfstruct_list)
        extra_concat_tfstruct = model_utils_py3.concat_tfstructs(extra_tfstruct_list)

        # encode
        concat_tfstruct = encoder(concat_tfstruct, attn_masks, extra_concat_tfstruct)

        # split and distribute results
        tfstruct_list = model_utils_py3.split_tfstructs(concat_tfstruct, tfstruct_lengths)

    return tfstruct_list

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

        initial_state = SpellerState(
            word_encodes=word_encodes,
            field_query_embedding=None,
            field_key_embedding=None,
            field_value_embedding=None,
            dec_tfstruct=None,
            enc_tfstruct=None)
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


"""Generator"""

class SeqGenerator(object):
    """
    generate sequences
    """
    def __init__(self, cell, matcher, has_copy):
        """
        args:
            cell: map from inputs, state to outputs, state
            matcher:
            has_copy: bool
        """
        self.cell = cell
        self.matcher = matcher
        self.has_copy = has_copy

        self.decoder = model_utils_py3.greedy_dec

    def generate(self, initial_state, length, candidates, limited_vocab):
        """
        args:
            initial_state:
            length:
            candidates:
                when pass None, return SEP embeds and ids
                args:
                    encodes: cell outputs
                return:
                    candidate_embeds: [batch_size x ]num_candidates x input_dim
                    candidate_ids: [batch_size x ]num_candidates [x word_len]
                    candidate_masks: [batch_size x ]num_candidates
            limited_vocab: bool
        return:
            seqs: batch_size x num_candidates x length [x word_len]
            scores: batch_size x num_candidates
        """
        def candidates_callback(encodes):
            candidate_embeds, candidate_ids, candidate_masks = candidates(encodes)
            if encodes is None:
                return candidate_embeds, candidate_ids, None, None
            else:
                if self.has_copy:
                    candidate_embeds = tf.concat(
                        [candidate_embeds, tf.zeros_like(candidate_embeds)], axis=-1)
                if limited_vocab:
                    logits = self.matcher(encodes, candidate_embeds)
                else:
                    logits = self.matcher(tf.expand_dims(encodes, axis=1), candidate_embeds)
                    logits = tf.squeeze(logits, [1])
                return candidate_embeds, candidate_ids, candidate_masks, logits
        seqs, scores = self.decoder(
            length, initial_state, self.cell, candidates_callback, limited_vocab)

        return seqs, scores

class WordGenerator(SeqGenerator):
    """
    generate word
    """
    def __init__(self, speller_cell, speller_matcher, spellin_embedding, sep_id):
        """
        args:
            speller_cell: speller cell
            speller_matcher: matcher for speller
            spellin_embedding: 0 is for start/end, others are embeddings for characters
            sep_id: id for special sep token
        """
        self.sep_id = sep_id
        pad_embedding = spellin_embedding[:1]
        pad_ids = tf.zeros([1], dtype=tf.int32)
        vocab_size = tf.shape(spellin_embedding)[0]
        nosep_embedding = tf.concat(
            [spellin_embedding[:sep_id], spellin_embedding[sep_id+1:]], axis=0)
        nosep_ids = tf.concat(
            [tf.range(sep_id, dtype=tf.int32), tf.range(sep_id+1, vocab_size, dtype=tf.int32)],
            axis=0)
        def candidates(encodes):
            if encodes is None:
                return pad_embedding, pad_ids, None
            else:
                return nosep_embedding, nosep_ids, None
        self.candidates = candidates
        SeqGenerator.__init__(self, speller_cell, speller_matcher, False)

    def generate(self, initial_state, length):
        """
        args:
            initial_state:
            length:
        """
        SeqGenerator.generate(self, initial_state, length, self.candidates, True)

class SentGenerator(SeqGenerator):
    """
    generate sent
    """
    def __init__(self, word_cell, word_matcher, word_embedder, word_generator):
        """
        args:
            word_cell: word level cell
            word_matcher: word level matcher
            word_embedder: word level embedder
            word_generator:
        """
        self.word_embedder = word_embedder
        self.word_generator = word_generator
        SeqGenerator.__init__(self, word_cell, word_matcher, True)

    def generate(self, initial_state, length, max_word_len=None, sep_id=None, word_embedding=None, word_ids=None):
        """
        args:
            max_word_len: int
            sep_id: int
            word_embedding: num_words x embed_dim
            word_ids: num_words
        """
        if not (sep_id is None or word_embedding is None or word_ids is None):
            limited_vocab = True
            sep_mask = tf.equal(word_ids, sep_id)
            nosep_mask = tf.logical_not(sep_mask)
            sep_ids = tf.constant([sep_id], dtype=tf.int32)
            sep_embedding = tf.boolean_mask(word_embedding, sep_mask)
            nosep_embedding = tf.boolean_mask(word_embedding, nosep_mask)
            nosep_ids = tf.boolean_mask(word_ids, nosep_mask)
            def candidates(encodes):
                if encodes is None:
                    return sep_embedding, sep_ids, None
                else:
                    word_embeds = tf.concat([sep_embedding, nosep_embedding], axis=0)
                    word_ids = tf.concat([sep_ids, nosep_ids], axis=0)
                    return word_embeds, word_ids, None
        elif not max_word_len is None:
            limited_vocab = False
            sep_ids = tf.constant([[self.word_generator.sep_id]+[0]*(max_word_len-1)], dtype=tf.int32)
            sep_embedding = tf.squeeze(self.word_embedder(tf.expand_dims(sep_ids, axis=0)), [0])
            def candidates(encodes):
                if encodes is None:
                    return sep_embedding, sep_ids, None
                else:
                    batch_size = tf.shape(encodes)[0]
                    initial_state = SpellerState(
                        word_encodes=encodes,
                        field_query_embedding=None,
                        field_key_embedding=None,
                        field_value_embedding=None,
                        dec_tfstruct=None,
                        enc_tfstruct=None)
                    word_ids = word_generator.generate(initial_state, max_word_len)
                    word_embeds = word_embedder(word_ids)
                    word_embeds = tf.concat(
                        [tf.tile(tf.expand_dims(sep_embedding, axis=0), [batch_size, 1, 1]), word_embeds],
                        axis=1)
                    word_ids = tf.concat(
                        [tf.tile(tf.expand_dims(sep_ids, axis=0), [batch_size, 1, 1]), word_ids], axis=1)
                    return word_embeds, word_ids, None
        SeqGenerator.generate(self, initial_state, length, candidates, limited_vocab)

class ClassGenerator(object):
    """
    generate class token
    """
    def __init__(self, word_encoder, word_matcher, word_embedder, word_generator):
        """
        args:
            word_cell: word level cell
            word_matcher: word level matcher
            word_embedder: word level embedder
            word_generator:
        """
        self.encoder = word_encoder
        self.matcher = word_matcher
        self.word_embedder = word_embedder
        self.word_generator = word_generator

    def generate(self, tfstruct, extra_tfstruct, max_word_len=None, word_embedding=None, word_ids=None):
        """
        args:
            max_word_len: int
            word_embedding: num_words x embed_dim
            word_ids: num_words
        return:
            classes: batch_size x num_candidates [x word_len]
            scores: batch_size x num_candidates
        """
        batch_size = tf.shape(tfstruct.token_embeds)[0]
        attn_masks = tf.zeros([batch_size, 1, 1], tf.bool)
        if not extra_tfstruct is None:
            attn_masks = tf.concat([attn_masks, tf.expand_dims(extra_tfstruct.masks, axis=1)], axis=2)
        tfstruct = self.encoder(tfstruct, attn_masks, extra_tfstruct)
        word_encodes = tfstruct.encodes
        if not (word_embedding is None or word_ids is None):
            limited_vocab = True
            candidate_embeds, candidate_ids = word_embedding, word_ids
            candidate_embeds = tf.concat(
                [candidate_embeds, tf.zeros_like(candidate_embeds)], axis=-1)
            logits = self.matcher(word_encodes, candidate_embeds)
        elif not max_word_len is None:
            limited_vocab = False
            initial_state = SpellerState(
                word_encodes=word_encodes,
                field_query_embedding=None,
                field_key_embedding=None,
                field_value_embedding=None,
                dec_tfstruct=None,
                enc_tfstruct=None)
            candidate_ids = self.word_generator.generate(initial_state, max_word_len)
            candidate_embeds = self.word_embedder(candidate_ids)
            candidate_embeds = tf.concat(
                [candidate_embeds, tf.zeros_like(candidate_embeds)], axis=-1)
            logits = self.matcher(tf.expand_dims(word_encodes, axis=1), candidate_embeds)
            logits = tf.squeeze(logits, [1])
        log_probs = tf.nn.log_softmax(logits)
        indices = tf.argmax(log_probs, 1, output_type=tf.int32)
        batch_indices = tf.stack([tf.range(batch_size, dtype=tf.int32), indices], axis=1)
        if limited_vocab:
            classes = tf.gather(candidate_ids, indices)
        else:
            classes = tf.gather_nd(candidate_ids, batch_indices)
        scores = tf.gather_nd(log_probs, batch_indices)
        classes = tf.expand_dims(tf.expand_dims(classes, axis=1), axis=1)
        scores = tf.expand_dims(tf.expand_dims(scores, axis=1), axis=1)
        return classes, scores


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

"""
TransformerState:
    field_query_embedding: batch_size x num_layers*layer_size
    field_key_embedding: batch_size x num_layers*layer_size
    field_value_embedding: batch_size x num_layers*layer_size
    dec_tfstruct: {
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
    enc_tfstruct
"""
TransformerState = namedtuple('TransformerState', [
    'field_query_embedding',
    'field_key_embedding',
    'field_value_embedding',
    'dec_tfstruct',
    'enc_tfstruct']
)

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
        args:
            inputs: batch_size x token_size or batch_size x length x token_size
            state: TransformerState
        return:
            outputs: batch_size x layer_size or batch_size x length x layer_size
            state: TransformerState
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):

            batch_size = tf.shape(inputs)[0]
            to_squeeze = False
            if inputs.shape.ndims == 2:
                to_squeeze = True
                inputs = tf.expand_dims(inputs, axis=1)
            length = tf.shape(inputs)[1]

            if state.dec_tfstruct is None:
                current_idx = 0
            else:
                current_idx = tf.shape(state.dec_tfstruct.token_embeds)[1]

            # prepare tfstruct
            field_query_embeds = tf.tile(
                tf.expand_dims(state.field_query_embedding, axis=1),
                [1, 2*length, 1])
            field_key_embeds = tf.tile(
                tf.expand_dims(state.field_key_embedding, axis=1),
                [1, 2*length, 1])
            field_value_embeds = tf.tile(
                tf.expand_dims(state.field_value_embedding, axis=1),
                [1, 2*length, 1])
            posit_embeds = tf.tile(
                model_utils_py3.embed_position(
                    tf.expand_dims(tf.concat([tf.range(length), tf.range(1, length+1)], axis=0), axis=0) + \
                        current_idx,
                    self.assets['posit_size']),
                [batch_size, 1, 1])
            token_embeds = tf.concat([inputs, tf.zeros_like(inputs)], axis=1)
            masks = tf.pad(tf.reduce_any(tf.not_equal(inputs, 0.0), axis=-1), [[0,0], [0,length]])

            tfstruct = model_utils_py3.TransformerStruct(
                field_query_embeds=field_query_embeds,
                field_key_embeds=field_key_embeds,
                field_value_embeds=field_value_embeds,
                posit_embeds=posit_embeds,
                token_embeds=token_embeds,
                masks=masks,
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )

            # prepare extra_tfstruct
            extra_tfstruct_list = []
            if not state.dec_tfstruct is None:
                extra_tfstruct_list.append(state.dec_tfstruct)
            if not state.enc_tfstruct is None:
                extra_tfstruct_list.append(state.enc_tfstruct)
            extra_tfstruct = model_utils_py3.concat_tfstructs(extra_tfstruct_list)

            # prepare masks
            attn_masks = tf.sequence_mask(
                 tf.tile(tf.expand_dims(tf.range(1, length+1),0), [batch_size,1]),
                 maxlen=2*length)
            attn_masks = tf.concat([attn_masks, attn_masks], axis=1)
            if not extra_tfstruct is None:
                attn_masks = tf.concat(
                    [attn_masks, tf.tile(tf.expand_dims(extra_tfstruct.masks, axis=1), [1, 2*length, 1])],
                    axis=2)

            tfstruct = self.assets['encoder'](tfstruct, attn_masks, extra_tfstruct=extra_tfstruct)
            input_tfstruct, output_tfstruct = model_utils_py3.split_tfstructs(tfstruct, 2)
            outputs = output_tfstruct.encodes
            if to_squeeze:
                outputs = tf.squeeze(outputs, axis=[1])
            dec_tfstruct = model_utils_py3.concat_tfstructs([state.dec_tfstruct, input_tfstruct])
            state = TransformerState(
                field_query_embedding=state.field_query_embedding,
                field_key_embedding=state.field_key_embedding,
                field_value_embedding=state.field_value_embedding,
                dec_tfstruct=dec_tfstruct,
                enc_tfstruct=state.enc_tfstruct)

        outputs, state = GeneralCell.__call__(self, outputs, state)

        return outputs, state

"""
SpellerState:
    word_encodes: batch_size x word_layer_size
    field_query_embedding: batch_size x num_layers*layer_size
    field_key_embedding: batch_size x num_layers*layer_size
    field_value_embedding: batch_size x num_layers*layer_size
    dec_tfstruct: {
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
    enc_tfstruct
"""
SpellerState = namedtuple('SpellerState', [
    'word_encodes',
    'field_query_embedding',
    'field_key_embedding',
    'field_value_embedding',
    'dec_tfstruct',
    'enc_tfstruct']
)

class SpellerCell(TransformerCell):
    """wraps a speller"""
    def __call__(self,
                 inputs,
                 state):
        """
        first map from word_encodes to field_value_embedding
        args:
            inputs: batch_size x token_size or batch_size x length x token_size
            state: SpellerState
        return:
            outputs: batch_size x layer_size or batch_size x length x layer_size
            state: SpellerState
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):

            word_encodes = state.word_encodes
            if state.field_value_embedding is None:
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
                state = TransformerState(
                    field_query_embedding=field_query_embedding,
                    field_key_embedding=field_key_embedding,
                    field_value_embedding=field_value_embedding,
                    dec_tfstruct=state.dec_tfstruct,
                    enc_tfstruct=state.enc_tfstruct)
            else:
                state = TransformerState(
                    field_query_embedding=state.field_query_embedding,
                    field_key_embedding=state.field_key_embedding,
                    field_value_embedding=state.field_value_embedding,
                    dec_tfstruct=state.dec_tfstruct,
                    enc_tfstruct=state.enc_tfstruct)

        outputs, state = TransformerCell.__call__(self, inputs, state)
        state = SpellerState(
            word_encodes=word_encodes,
            field_query_embedding=state.field_query_embedding,
            field_key_embedding=state.field_key_embedding,
            field_value_embedding=state.field_value_embedding,
            dec_tfstruct=state.dec_tfstruct,
            enc_tfstruct=state.enc_tfstruct)

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

    def __call__(self, tfstruct, attn_masks, extra_tfstruct=None):
        """
        args:
            tfstruct: {
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
            extra_tfstruct
        returns:
            tfstruct
        """
        tfstruct = model_utils_py3.transformer(
            tfstruct,
            self.num_layers,
            self.layer_size,
            extra_tfstruct=extra_tfstruct,
            num_heads=self.num_heads,
            attn_masks=attn_masks,
            dropout=self.dropout,
            is_training=self.training,
            reuse=self.reuse,
            scope=self.scope)

        self.reuse = True

        return tfstruct

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
