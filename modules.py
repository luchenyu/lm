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
    pct_start,
    wd):
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
                lambda: (tf.math.cos((x-pct_start)/(1.0-pct_start)*math.pi) + 1.0) * 0.5*(max_lr-wd) + wd)
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
        field_context_embedding = tf.get_variable(
            "field_context_embedding",
            shape=[64, 2*layer_size],
            dtype=tf.float32,
            initializer=tf.initializers.variance_scaling(mode='fan_out'),
            trainable=training,
            collections=collections,
            aggregation=tf.VariableAggregation.MEAN)

    return input_embedding, spellin_embedding, \
        field_query_embedding, field_key_embedding, field_value_embedding, field_context_embedding

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

        # encode
        concat_tfstruct = encoder(concat_tfstruct, attn_masks, extra_tfstruct_list)

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
            dec_masks=None,
            dec_keys=None,
            dec_values=None)
        inputs = tf.nn.embedding_lookup(
            spellin_embedding,
            tf.pad(target_seqs, [[0,0],[1,0]]))
        target_seqs = tf.pad(target_seqs, [[0,0],[0,1]])
        decMasks = tf.not_equal(target_seqs, 0)
        decMasks = tf.logical_or(decMasks, tf.pad(tf.not_equal(target_seqs, 0), [[0,0],[1,0]])[:,:-1])

        def true_fn():
            outputs, state = speller_cell(inputs, initial_state)
            _, valid_outputs = tf.dynamic_partition(
                outputs, tf.cast(decMasks, tf.int32), 2)
            valid_target_seqs = tf.boolean_mask(target_seqs, decMasks)
            valid_logits = speller_matcher((valid_outputs, spellin_embedding), ('encode', 'embed'))
            valid_labels = tf.one_hot(valid_target_seqs, tf.shape(spellin_embedding)[0])
            valid_labels = tf.reshape(valid_labels, [-1])
            valid_labels /= tf.reduce_sum(valid_labels)
            valid_logits = tf.reshape(valid_logits, [-1])
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=valid_labels,
                logits=valid_logits)
            return loss

        loss = tf.cond(
            tf.reduce_any(decMasks),
            true_fn,
            lambda: tf.zeros([]))

    return loss


"""Generator"""

class SeqGenerator(object):
    """
    generate sequences
    """
    def __init__(self, cell, matcher, decoder):
        """
        args:
            cell: map from inputs, state to outputs, state
            matcher:
            has_copy: bool
        """
        self.cell = cell
        self.matcher = matcher
        self.decoder = decoder

    def generate(self, initial_state, state_si_fn, length, candidates_fn_list, start_embedding, start_id):
        """
        args:
            initial_state:
            state_si: state shape invariants
            length:
            candidates_fn:
                when pass None, return SEP embeds and ids
                args:
                    encodes: cell outputs
                return:
                    candidate_embeds: [batch_size x ]num_candidates x input_dim
                    candidate_ids: [batch_size x ]num_candidates [x word_len]
                    candidate_masks: [batch_size x ]num_candidates
                    candidate_encodes: [batch_size x ]num_candidates x encode_dim
            start_embedding: input_dim
            ids_len: 0 or int or tf.int32
        return:
            seqs: batch_size x num_candidates x length [x word_len]
            scores: batch_size x num_candidates
        """
        def candidates_callback(encodes):
            """
            for dec callback
            """
            concated_embeds, concated_ids, concated_masks, concated_logits = None, None, None, None
            for candidates_fn in candidates_fn_list:

                candidate_embeds, candidate_ids, candidate_masks, candidate_encodes \
                    = candidates_fn(encodes)

                if len(candidate_embeds.get_shape()) == 2:
                    logits = self.matcher((encodes, candidate_embeds), ('encode', 'embed'))
                    candidate_masks = tf.ones_like(logits, dtype=tf.bool)
                elif len(candidate_embeds.get_shape()) == 3:
                    logits = self.matcher((tf.expand_dims(encodes, axis=1), candidate_embeds), ('encode', 'embed'))
                if not candidate_encodes is None:
                    logits += self.matcher((tf.expand_dims(encodes, axis=1), candidate_encodes), ('encode', 'encode'))
                if len(logits.get_shape()) == 3:
                    logits = tf.squeeze(logits, [1])

                if concated_embeds is None:
                    concated_embeds, concated_ids, concated_masks, concated_logits = \
                        candidate_embeds, candidate_ids, candidate_masks, logits
                else:
                    concated_masks = tf.concat([concated_masks, candidate_masks], axis=1)
                    concated_logits = tf.concat([concated_logits, logits], axis=1)
                    if len(concated_embeds.get_shape()) == 2:
                        if len(candidate_embeds.get_shape()) == 2:
                            concated_embeds = tf.concat([concated_embeds, candidate_embeds], axis=0)
                            concated_ids = tf.concat([concated_ids, candidate_ids], axis=0)
                        elif len(candidate_embeds.get_shape()) == 3:
                            batch_size = tf.shape(candidate_embeds)[0]
                            concated_embeds = tf.tile(tf.expand_dims(concated_embeds, axis=0), [batch_size,1,1])
                            concated_ids = tf.tile(
                                tf.expand_dims(concated_ids, axis=0),
                                [batch_size,]+[1]*len(concated_ids.get_shape()))
                            concated_embeds = tf.concat([concated_embeds, candidate_embeds], axis=1)
                            concated_ids = tf.concat([concated_ids, candidate_ids], axis=1)
                    elif len(concated_embeds.get_shape()) == 3:
                        if len(candidate_embeds.get_shape()) == 2:
                            batch_size = tf.shape(concated_embeds)[0]
                            candidate_embeds = tf.tile(tf.expand_dims(candidate_embeds, axis=0), [batch_size,1,1])
                            candidate_ids = tf.tile(
                                tf.expand_dims(candidate_ids, axis=0),
                                [batch_size,]+[1]*len(candidate_ids.get_shape()))
                            concated_embeds = tf.concat([concated_embeds, candidate_embeds], axis=1)
                            concated_ids = tf.concat([concated_ids, candidate_ids], axis=1)
                        elif len(candidate_embeds.get_shape()) == 3:
                            concated_embeds = tf.concat([concated_embeds, candidate_embeds], axis=1)
                            concated_ids = tf.concat([concated_ids, candidate_ids], axis=1)

            return concated_embeds, concated_ids, concated_masks, concated_logits

        seqs, scores = self.decoder(
            length, initial_state, state_si_fn, self.cell, candidates_callback, start_embedding, start_id)

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
        self.pad_embedding = spellin_embedding[0]
        self.nosep_embedding = tf.concat(
            [spellin_embedding[:sep_id], spellin_embedding[sep_id+1:]], axis=0)
        self.nosep_ids = tf.concat(
            [tf.range(sep_id, dtype=tf.int32), 
             tf.range(sep_id+1, tf.shape(spellin_embedding)[0], dtype=tf.int32)],
            axis=0)
        speller_matcher.cache_embeds(self.nosep_embedding)
        decoder = lambda *args: model_utils_py3.beam_dec(*args, beam_size=32, num_candidates=64)

        SeqGenerator.__init__(self, speller_cell, speller_matcher, decoder)

    def generate(self, initial_state, length):
        """
        args:
            initial_state: SpellerState
            length:
        """
        def state_si_fn(state):
            state_si = SpellerState(
                word_encodes=state.word_encodes.get_shape(),
                dec_masks=tf.TensorShape([state.dec_masks.get_shape()[0], None]),
                dec_keys=tuple(
                    [tf.TensorShape([k.get_shape()[0], None, k.get_shape()[2]])
                     for k in state.dec_keys]),
                dec_values=tuple(
                    [tf.TensorShape([v.get_shape()[0], None, v.get_shape()[2]])
                     for v in state.dec_values]))
            return state_si

        self.cell.cache_encodes(initial_state.word_encodes)

        def candidates_fn(encodes):
            return self.nosep_embedding, self.nosep_ids, None, None

        start_embedding = self.pad_embedding

        seqs, scores = SeqGenerator.generate(
            self, initial_state, state_si_fn, length, [candidates_fn],
            start_embedding, tf.zeros([], dtype=tf.int32))

        return seqs[:,:,1:], scores

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
        decoder = lambda *args: model_utils_py3.stochastic_beam_dec(*args, beam_size=32, num_candidates=1)

        SeqGenerator.__init__(self, word_cell, word_matcher, decoder)

    def generate(self, initial_state, length,
                 gen_word_len=None, word_embedding=None, word_ids=None,
                 copy_embeds=None, copy_ids=None, copy_masks=None, copy_encodes=None):
        """
        args:
            initial_state: TransformerState
            length:
            gen_word_len: int
            word_embedding: num_words x embed_dim
            word_ids: num_words, no sep when in char mode, first one is sep when in word mode
            copy_embeds: batch_size x num_words x embed_dim
            copy_ids: batch_size x num_words [x word_len]
            copy_masks: batch_size x num_words
            copy_embeds_matcher: batch_size x num_words x matcher_dim
        """
        def state_si_fn(state):
            state_si = TransformerState(
                field_query_embedding=tuple(
                    [q.get_shape() for q in state.field_query_embedding]),
                field_key_embedding=tuple(
                    [k.get_shape() for k in state.field_key_embedding]),
                field_value_embedding=tuple(
                    [v.get_shape() for v in state.field_value_embedding]),
                dec_masks=tf.TensorShape([state.dec_masks.get_shape()[0], None]),
                dec_keys=tuple(
                    [tf.TensorShape([k.get_shape()[0], None, k.get_shape()[2]])
                     for k in state.dec_keys]),
                dec_values=tuple(
                    [tf.TensorShape([v.get_shape()[0], None, v.get_shape()[2]])
                     for v in state.dec_values]),
                enc_tfstruct=model_utils_py3.get_tfstruct_si(state.enc_tfstruct))
            return state_si

        candidates_fn_list = []
        max_word_len = gen_word_len if not gen_word_len is None else 0
        if not word_ids is None:
            max_word_len = tf.maximum(max_word_len, tf.shape(word_ids)[-1])
        if not copy_ids is None:
            max_word_len = tf.maximum(max_word_len, tf.shape(copy_ids)[-1])

        if not self.word_generator is None:

            # first sep token
            sep_ids = tf.constant([[self.word_generator.sep_id]], dtype=tf.int32)
            sep_embedding, _ = self.word_embedder(tf.expand_dims(sep_ids, axis=0))
            sep_embedding = tf.squeeze(sep_embedding, [0])
            sep_ids = tf.pad(sep_ids, [[0,0],[0,max_word_len-1]])
            self.matcher.cache_embeds(sep_embedding)
            def candidates_fn(encodes):
                return sep_embedding, sep_ids, None, None
            candidates_fn_list.append(candidates_fn)

            # dynamic gen
            if not gen_word_len is None:
                def candidates_fn(encodes):
                    batch_size = tf.shape(encodes)[0]
                    speller_encoder = self.word_generator.cell.assets['encoder']
                    null_tfstruct = model_utils_py3.init_tfstruct(
                        batch_size, speller_encoder.embed_size, speller_encoder.posit_size,
                        speller_encoder.layer_size, speller_encoder.num_layers)
                    speller_initial_state = SpellerState(
                        word_encodes=encodes,
                        dec_masks=tf.zeros([batch_size, 0], dtype=tf.bool),
                        dec_keys=(
                            tf.zeros([batch_size, 0, speller_encoder.layer_size]),
                        )*speller_encoder.num_layers,
                        dec_values=(
                            tf.zeros([batch_size, 0, speller_encoder.layer_size]),
                        )*speller_encoder.num_layers)
                    word_ids, _ = self.word_generator.generate(speller_initial_state, max_word_len)
                    word_embeds, word_masks = self.word_embedder(word_ids)
                    word_ids = tf.pad(word_ids, [[0,0],[0,0],[0,max_word_len-tf.shape(word_ids)[2]]])
                    return word_embeds, word_ids, word_masks, None
                candidates_fn_list.append(candidates_fn)

            start_id = tf.squeeze(sep_ids, [0])
            start_embedding = tf.squeeze(sep_embedding, [0])

        else:

            start_id = tf.cast(0, tf.int32)
            start_embedding = word_embedding[0]

        # static vocab
        if not (word_embedding is None or word_ids is None):
            word_ids = tf.pad(word_ids, [[0,0],[0,max_word_len-tf.shape(word_ids)[1]]])
            self.matcher.cache_embeds(word_embedding)
            def candidates_fn(encodes):
                return word_embedding, word_ids, None, None
            candidates_fn_list.append(candidates_fn)

        # copy
        if not (copy_embeds is None or copy_ids is None or copy_masks is None or copy_encodes is None):
            copy_ids = tf.pad(copy_ids, [[0,0],[0,0],[0,max_word_len-tf.shape(copy_ids)[2]]])
            self.matcher.cache_embeds(copy_embeds)
            self.matcher.cache_encodes(copy_encodes)
            def candidates_fn(encodes):
                return copy_embeds, copy_ids, copy_masks, copy_encodes
            candidates_fn_list.append(candidates_fn)

        seqs, scores = SeqGenerator.generate(
            self, initial_state, state_si_fn, length-1, candidates_fn_list, start_embedding, start_id)
        if len(seqs.get_shape()) == 3:
            seqs = tf.pad(seqs, [[0,0],[0,0],[0,length-tf.shape(seqs)[2]]])
        elif len(seqs.get_shape()) == 4:
            seqs = tf.pad(seqs, [[0,0],[0,0],[0,length-tf.shape(seqs)[2]],[0,0]])

        return seqs, scores

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
        word_encodes = tf.squeeze(word_encodes, [1])
        if not (word_embedding is None or word_ids is None):
            candidate_embeds, candidate_ids = word_embedding, word_ids
            logits = self.matcher((word_encodes, candidate_embeds), ('encode', 'embed'))
        elif not max_word_len is None:
            speller_encoder = self.word_generator.cell.assets['encoder']
            null_tfstruct = model_utils_py3.init_tfstruct(
                batch_size, speller_encoder.embed_size, speller_encoder.posit_size,
                speller_encoder.layer_size, speller_encoder.num_layers)
            speller_initial_state = SpellerState(
                word_encodes=encodes,
                dec_masks=tf.zeros([batch_size, 0], dtype=tf.bool),
                dec_keys=(
                    tf.zeros([batch_size, 0, speller_encoder.layer_size]),
                )*speller_encoder.num_layers,
                dec_values=(
                    tf.zeros([batch_size, 0, speller_encoder.layer_size]),
                )*speller_encoder.num_layers)
            candidate_ids, _ = self.word_generator.generate(speller_initial_state, max_word_len)
            candidate_embeds, _ = self.word_embedder(candidate_ids)
            candidate_ids = tf.pad(
                candidate_ids, [[0,0],[0,0],[0,max_word_len-tf.shape(candidate_ids)[2]]])
            logits = self.matcher((tf.expand_dims(word_encodes, axis=1), candidate_embeds), ('encode', 'embed'))
            logits = tf.squeeze(logits, [1])
        log_probs = tf.nn.log_softmax(logits)
        indices = tf.argmax(log_probs, 1, output_type=tf.int32)
        batch_indices = tf.stack([tf.range(batch_size, dtype=tf.int32), indices], axis=1)
        if len(candidate_embeds.get_shape()) == 2:
            classes = tf.gather(candidate_ids, indices)
        elif len(candidate_embeds.get_shape()) == 3:
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
    field_query_embedding: tuple (batch_size x layer_size) * num_layers
    field_key_embedding: tuple (batch_size x layer_size) * num_layers
    field_value_embedding: tuple (batch_size x layer_size) * num_layers
    dec_masks: batch_size x dec_length
    dec_keys: tuple (batch_size x dec_length x layer_size) * num_layers
    dec_values: tuple (batch_size x dec_length x layer_size) * num_layers
    enc_tfstruct: {
        'field_query_embeds': tuple (batch_size x length x layer_size) * num_layers,
        'field_key_embeds': tuple (batch_size x length x layer_size) * num_layers,
        'field_value_embeds': tuple (batch_size x length x layer_size) * num_layers,
        'posit_embeds': batch_size x length x posit_size,
        'token_embeds': batch_size x length x embed_size,
        'masks': batch_size x length,
        'querys': tuple (batch_size x length x layer_size) * num_layers,
        'keys': tuple (batch_size x length x layer_size) * num_layers,
        'values': tuple (batch_size x length x layer_size) * num_layers,
        'encodes': batch_size x length x layer_size
    }
"""
TransformerState = namedtuple('TransformerState', [
    'field_query_embedding',
    'field_key_embedding',
    'field_value_embedding',
    'dec_masks',
    'dec_keys',
    'dec_values',
    'enc_tfstruct']
)

class TransformerCell(GeneralCell):
    """Cell that wraps transformer"""
    def __init__(self,
                 scope,
                 encoder,
                 dropout=None,
                 training=True):
        """
        args:
            encoder: module
            dropout
            training
        """

        GeneralCell.__init__(self,
                             scope,
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

            if state.dec_masks is None:
                dec_length = 0
            else:
                dec_length = tf.shape(state.dec_masks)[1]

            # prepare tfstruct
            field_query_embeds = tuple(
                [tf.tile(tf.expand_dims(f, axis=1), [1, 2*length, 1]) for f in state.field_query_embedding])
            field_key_embeds = tuple(
                [tf.tile(tf.expand_dims(f, axis=1), [1, 2*length, 1]) for f in state.field_key_embedding])
            field_value_embeds = tuple(
                [tf.tile(tf.expand_dims(f, axis=1), [1, 2*length, 1]) for f in state.field_value_embedding])
            posit_embeds = tf.tile(
                model_utils_py3.embed_position(
                    tf.expand_dims(tf.concat([tf.range(length), tf.range(1, length+1)], axis=0), axis=0) + \
                        dec_length,
                    self.assets['encoder'].posit_size),
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
            if not state.dec_masks is None:
                dec_tfstruct = model_utils_py3.TransformerStruct(
                    field_query_embeds=None,
                    field_key_embeds=None,
                    field_value_embeds=None,
                    posit_embeds=None,
                    token_embeds=None,
                    masks=state.dec_masks,
                    querys=None,
                    keys=state.dec_keys,
                    values=state.dec_values,
                    encodes=None,
                )
                extra_tfstruct_list.append(dec_tfstruct)
            if not state.enc_tfstruct is None:
                extra_tfstruct_list.append(state.enc_tfstruct)

            # prepare masks
            attn_masks = tf.sequence_mask(
                 tf.tile(tf.expand_dims(tf.range(1, length+1),0), [batch_size,1]),
                 maxlen=2*length)
            attn_masks = tf.concat([attn_masks, attn_masks], axis=1)
            if not state.dec_masks is None:
                attn_masks = tf.concat(
                    [attn_masks, tf.tile(tf.expand_dims(state.dec_masks, axis=1), [1, 2*length, 1])],
                    axis=2)
            if not state.enc_tfstruct is None:
                attn_masks = tf.concat(
                    [attn_masks, tf.tile(tf.expand_dims(state.enc_tfstruct.masks, axis=1), [1, 2*length, 1])],
                    axis=2)

            tfstruct = self.assets['encoder'](tfstruct, attn_masks, extra_tfstruct_list)
            input_tfstruct, output_tfstruct = model_utils_py3.split_tfstructs(tfstruct, 2)
            outputs = output_tfstruct.encodes
            if to_squeeze:
                outputs = tf.squeeze(outputs, axis=[1])

            if state.dec_masks is None:
                dec_masks = input_tfstruct.masks
                dec_keys=input_tfstruct.keys
                dec_values=input_tfstruct.values
            else:
                dec_masks = tf.concat([state.dec_masks, input_tfstruct.masks], axis=1)
                dec_keys = tuple(
                    [tf.concat([state.dec_keys[i], input_tfstruct.keys[i]], axis=1)
                     for i in range(len(state.dec_keys))])
                dec_values = tuple(
                    [tf.concat([state.dec_values[i], input_tfstruct.values[i]], axis=1)
                     for i in range(len(state.dec_values))])
            state = TransformerState(
                field_query_embedding=state.field_query_embedding,
                field_key_embedding=state.field_key_embedding,
                field_value_embedding=state.field_value_embedding,
                dec_masks=dec_masks,
                dec_keys=dec_keys,
                dec_values=dec_values,
                enc_tfstruct=state.enc_tfstruct)

        outputs, state = GeneralCell.__call__(self, outputs, state)

        return outputs, state

"""
SpellerState:
    word_encodes: batch_size x word_layer_size
    dec_masks: batch_size x dec_length
    dec_keys: tuple (batch_size x dec_length x layer_size) * num_layers
    dec_values: tuple (batch_size x dec_length x layer_size) * num_layers
"""
SpellerState = namedtuple('SpellerState', [
    'word_encodes',
    'dec_masks',
    'dec_keys',
    'dec_values']
)

class SpellerCell(TransformerCell):
    """wraps a speller"""
    def __init__(self,
                 scope,
                 encoder,
                 dropout=None,
                 training=True):
        """
        args:
            encoder: module
            dropout
            training
        """

        self.cached = {}
        TransformerCell.__init__(self,
                                 scope,
                                 encoder=encoder,
                                 dropout=dropout,
                                 training=training)

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
            field_query_embedding, field_key_embedding, \
                field_value_embedding = self.cache_encodes(word_encodes)

            state = TransformerState(
                field_query_embedding=field_query_embedding,
                field_key_embedding=field_key_embedding,
                field_value_embedding=field_value_embedding,
                dec_masks=state.dec_masks,
                dec_keys=state.dec_keys,
                dec_values=state.dec_values,
                enc_tfstruct=None)

        outputs, state = TransformerCell.__call__(self, inputs, state)
        state = SpellerState(
            word_encodes=word_encodes,
            dec_masks=state.dec_masks,
            dec_keys=state.dec_keys,
            dec_values=state.dec_values)

        return outputs, state

    def cache_encodes(self, word_encodes):
        """
        args:
            word_encodes: batch_size x word_layer_size
        """
        with tf.variable_scope(self.scope):

            if self.cached.get(word_encodes) is None:
                num_layers = self.assets['encoder'].num_layers
                layer_size = self.assets['encoder'].layer_size
                reuse = None if len(self.cached) == 0 else True
                field_value_embedding = model_utils_py3.fully_connected(
                    word_encodes,
                    num_layers*layer_size,
                    dropout=self.assets['dropout'],
                    is_training=self.assets['training'],
                    reuse=reuse,
                    scope="enc_projs")
                field_value_embedding = tuple(
                    tf.split(field_value_embedding, num_layers, axis=-1))
                field_query_embedding = (tf.zeros_like(field_value_embedding[0]),)*num_layers
                field_key_embedding = (tf.zeros_like(field_value_embedding[0]),)*num_layers
                self.cached[word_encodes] = (
                    field_query_embedding, field_key_embedding, field_value_embedding)
            else:
                field_query_embedding, field_key_embedding, \
                    field_value_embedding = self.cached[word_encodes]

        return field_query_embedding, field_key_embedding, field_value_embedding


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
            char_masks = tf.reduce_any(tf.not_equal(char_embeds, 0.0), axis=-1, keepdims=True)
            char_masks = tf.cast(char_masks, tf.float32)
            l0_embeds = model_utils_py3.fully_connected(
                char_embeds,
                self.layer_size,
                activation_fn=tf.nn.relu,
                is_training=self.training,
                scope="l0_convs")
            l0_embeds *= char_masks
            l1_embeds = model_utils_py3.convolution2d(
                char_embeds,
                [self.layer_size]*2,
                [[1,2],[1,3]],
                activation_fn=tf.nn.relu,
                is_training=self.training,
                scope="l1_convs")
            l1_embeds *= char_masks
            char_embeds = tf.nn.max_pool(char_embeds, [1,1,2,1], [1,1,2,1], padding='SAME')
            char_masks = tf.reduce_any(tf.not_equal(char_embeds, 0.0), axis=-1, keepdims=True)
            char_masks = tf.cast(char_masks, tf.float32)
            l2_embeds = model_utils_py3.convolution2d(
                char_embeds,
                [self.layer_size]*2,
                [[1,2],[1,3]],
                activation_fn=tf.nn.relu,
                is_training=self.training,
                scope="l2_convs")
            l2_embeds *= char_masks
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
    def __init__(self, embed_size, posit_size, layer_size, num_layers, num_heads,
                 scope, dropout=None, training=False):
        """
        args:
            layer_size: size of layer
            num_layers: num of layers
            num_heads: num of attention heads
        """
        self.embed_size = embed_size
        self.posit_size = posit_size
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
        self.cached_encodes = {}
        self.cached_embeds = {}
        Module.__init__(self, scope, dropout, training)

    def __call__(self, pair, pair_type):
        """
        args:
            pair: (item1, item2), each is one of {'encode', 'embed', 'latent'} object
            pair_type: (type1, type2), str of type
        returns:
            logits: shape1 x shape2
        """
        with tf.variable_scope(self.scope, reuse=self.reuse):

            latents = []
            for item, item_type in zip(pair, pair_type):
                if item_type == 'encode':
                    latents.append(self.cache_encodes(item))
                elif item_type == 'embed':
                    latents.append(self.cache_embeds(item))
                elif item_type == 'latent':
                    latents.append(item)

            if len(latents[0].get_shape()) == len(latents[1].get_shape()):
                logits = tf.matmul(latents[0], latents[1], transpose_b=True)
            elif len(latents[0].get_shape()) == 2:
                shape = latents[1].get_shape()[:-1]
                logits = tf.matmul(
                    latents[0],
                    tf.reshape(latents[1], [-1, tf.shape(latents[1])[-1]]),
                    transpose_b=True)
                logits = tf.reshape(logits, tf.concat([tf.shape(logits)[0], shape], axis=0))
            elif len(latents[1].get_shape()) == 2:
                shape = latents[0].get_shape()[:-1]
                logits = tf.matmul(
                    tf.reshape(latents[0], [-1, tf.shape(latents[0])[-1]]),
                    latents[1],
                    transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, tf.shape(logits)[1]], axis=0))

        self.reuse = True

        return logits

    def cache_encodes(self, encodes):
        """
        args:
            encodes: num_candidates x encode_dim or batch_size x num_candidates x encode_dim
        """
        with tf.variable_scope(self.scope):

            if self.cached_encodes.get(encodes) is None:
                reuse = None if len(self.cached_encodes) == 0 else True
                encode_projs = model_utils_py3.GLU(
                    encodes,
                    self.layer_size,
                    is_training=self.training,
                    reuse=reuse,
                    scope="enc_projs")
                self.cached_encodes[encodes] = encode_projs
            else:
                encode_projs = self.cached_encodes[encodes]

        return encode_projs

    def cache_embeds(self, embeds):
        """
        args:
            embeds: num_candidates x embed_dim or batch_size x num_candidates x embed_dim
        """
        with tf.variable_scope(self.scope):

            if self.cached_embeds.get(embeds) is None:
                reuse = None if len(self.cached_embeds) == 0 else True
                embed_projs = model_utils_py3.GLU(
                    embeds,
                    self.layer_size,
                    is_training=self.training,
                    reuse=reuse,
                    scope="emb_projs")
                self.cached_embeds[embeds] = embed_projs
            else:
                embed_projs = self.cached_embeds[embeds]

        return embed_projs


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
