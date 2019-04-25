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
    match_size,
    speller_embed_size,
    speller_match_size,
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
            2*speller_embed_size,
            speller_embed_size,
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
            shape=[64, match_size],
            dtype=tf.float32,
            initializer=tf.initializers.variance_scaling(mode='fan_out'),
            trainable=training,
            collections=collections,
            aggregation=tf.VariableAggregation.MEAN)
        field_word_embedding = tf.get_variable(
            "field_word_embedding",
            shape=[64, match_size],
            dtype=tf.float32,
            initializer=tf.initializers.variance_scaling(mode='fan_out'),
            trainable=training,
            collections=collections,
            aggregation=tf.VariableAggregation.MEAN)
        speller_context_embedding = tf.get_variable(
            "speller_context_embedding",
            shape=[1, speller_match_size],
            dtype=tf.float32,
            initializer=tf.initializers.variance_scaling(mode='fan_out'),
            trainable=training,
            collections=collections,
            aggregation=tf.VariableAggregation.MEAN)

    return (input_embedding, spellin_embedding, 
            field_query_embedding, field_key_embedding, field_value_embedding,
            field_context_embedding, field_word_embedding,
            speller_context_embedding)

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

def train_masked(
    matcher,
    limited_vocab,
    field_context_embeds, field_token_embeds,
    token_ids, token_embeds, encodes,
    valid_masks, pick_masks,
    global_encodes=None, global_matcher=None,
    candidate_ids=None, candidate_embeds=None, target_seqs=None,
    copy_token_ids=None, copy_encodes=None, copy_masks=None,
    extra_loss_fn=None):
    """
    get the loss
    args:
        matcher: matcher object
        limited_vocab: bool
        field_context_embeds: 1 x match_size
        field_token_embeds: 1 x match_size
        token_ids: batch_size x seq_length x word_len
        token_embeds: batch_size x seq_length x embed_dim
        encodes: batch_size x seq_length x encode_dim
        valid_masks: batch_size x seq_length
        pick_masks: batch_size x seq_length
        global_encodes: batch_size x macro_size
        global_matcher: matcher object for global matching
        candidate_ids: num_candidates x word_len
        candidate_embeds: num_candidates x embed_dim
        target_seqs: batch_size x seq_length
        copy_token_ids: batch_size x copy_seq_length x word_len
        copy_encodes: batch_size x copy_seq_length x encode_dim
        copy_masks: batch_size x copy_seq_length
        extra_loss_fn: fn for extra_loss
    """

    batch_size = tf.shape(token_ids)[0]
    seq_length = tf.shape(token_ids)[1]
    valid_masks_int = tf.cast(valid_masks, tf.int32)
    pick_masks_int = tf.cast(pick_masks, tf.int32)

    sample_ids = tf.tile(
        tf.expand_dims(tf.range(batch_size), axis=1), [1,seq_length])

    unique_valid_masks = model_utils_py3.mask_unique_vector(
        token_ids, valid_masks)
    unique_valid_masks_int = tf.cast(unique_valid_masks, tf.int32)
    unique_valid_token_ids = tf.boolean_mask(
        token_ids, unique_valid_masks)
    _, unique_valid_token_embeds = tf.dynamic_partition(
        token_embeds, unique_valid_masks_int, 2)
    num_unique_valids = tf.shape(unique_valid_token_ids)[0]

    pick_token_ids = tf.boolean_mask(
        token_ids, pick_masks)
    pick_sample_ids = tf.boolean_mask(
        sample_ids, pick_masks)
    pick_sample_onehots = tf.one_hot(
        pick_sample_ids, batch_size)
    _, pick_encodes = tf.dynamic_partition(
        encodes, pick_masks_int, 2)
    if not target_seqs is None:
        pick_target_seqs = tf.boolean_mask(
            target_seqs, pick_masks)
    num_picks = tf.shape(pick_encodes)[0]

    if not global_matcher is None:
        # sample latent - local context distribution
        sample_context_logits = global_matcher(
            (pick_encodes, None, 'encode'),
            (global_encodes, None, 'context'))
        sample_context_labels_sum = tf.reduce_sum(
            pick_sample_onehots)
        sample_context_labels = pick_sample_onehots * (
            1.0 / sample_context_labels_sum)

    # local context prior distribution
    context_prior_logits = matcher(
        (pick_encodes, None, 'context'),
        (field_context_embeds, None, 'latent'))

    if limited_vocab:
        num_candidates = tf.shape(candidate_ids)[0]

        # local context - token distribution
        match_matrix = tf.one_hot(pick_target_seqs, num_candidates)
        context_token_labels_sum = tf.reduce_sum(match_matrix)
        context_token_labels = match_matrix * (
            1.0 / context_token_labels_sum)
        context_token_logits = matcher(
            (pick_encodes, None, 'context'),
            (candidate_embeds, None, 'embed'))

        # sample latent - token distribution
        if not global_matcher is None:
            token_prior_logits = matcher(
                (field_token_embeds, None, 'latent'),
                (candidate_embeds, None, 'embed'))
            sample_token_logits = global_matcher(
                (global_encodes, None, 'context'),
                (candidate_embeds, None, 'embed'))
            sample_token_onehots = tf.one_hot(
                target_seqs, num_candidates,
                on_value=True, off_value=False)
            sample_token_onehots = tf.logical_and(
                sample_token_onehots,
                tf.expand_dims(valid_masks, axis=2))
            sample_token_labels = tf.reduce_any(
                sample_token_onehots, axis=1)
            sample_token_labels = tf.cast(
                sample_token_labels, tf.float32)
            sample_token_labels_sum = tf.reduce_sum(
                sample_token_labels)
            sample_token_labels *= 1.0 / sample_token_labels_sum

            sample_token_logits += tf.stop_gradient(token_prior_logits)
            context_token_logits -= tf.stop_gradient(tf.reduce_sum(
                sample_context_logits * pick_sample_onehots,
                axis=1, keepdims=True))
        context_token_logits -= tf.stop_gradient(context_prior_logits)

        # copy part
        if not (copy_token_ids is None or copy_encodes is None or copy_masks is None):
            token_ids, candidate_ids, copy_token_ids = model_utils_py3.pad_vectors(
                [token_ids, candidate_ids, copy_token_ids])
            # first we get the matrix indicates whether x copy to y
            copy_score_matrix = model_utils_py3.match_vector(
                copy_token_ids, token_ids)
            copy_score_matrix = tf.logical_and(
                copy_score_matrix, tf.expand_dims(pick_masks, axis=1))
            copy_score_matrix = tf.logical_and(
                copy_score_matrix, tf.expand_dims(copy_masks, axis=2))
            # calculate the normalized prob each slot being copied
            copy_scores = tf.cast(copy_score_matrix, tf.float32)
            copy_scores /= (tf.reduce_sum(copy_scores, axis=1, keepdims=True)+1e-12)
            copy_scores = tf.reduce_sum(copy_scores, axis=2)
            # gather all valid slots which can be selected from
            copy_valid_token_ids = tf.boolean_mask(
                copy_token_ids, copy_masks)
            copy_valid_scores = tf.boolean_mask(
                copy_scores, copy_masks)
            _, copy_valid_encodes = tf.dynamic_partition(
                copy_encodes, tf.cast(copy_masks, tf.int32), 2)
            copy_valid_encodes = tf.pad(copy_valid_encodes, [[0,1],[0,0]])
            # for each candidate token, we gather the probs of all corresponding slots
            copy_valid_matrix = model_utils_py3.match_vector(
                candidate_ids, copy_valid_token_ids)
            copy_valid_matrix = tf.cast(copy_valid_matrix, tf.float32)
            copy_valid_match_scores = copy_valid_matrix * tf.expand_dims(
                copy_valid_scores+1e-12, axis=0)
            # copy / no copy is 1:1
            copy_valid_pad_score = tf.reduce_sum(
                copy_valid_match_scores, axis=1, keepdims=True)
            copy_valid_pad_score = tf.maximum(copy_valid_pad_score, 1e-12)
            copy_valid_match_scores = tf.concat(
                [copy_valid_match_scores, copy_valid_pad_score], axis=1)
            random_ids = tf.squeeze(
                tf.random.categorical(
                    tf.log(copy_valid_match_scores), 1, dtype=tf.int32),
                axis=[-1])
            num_copys = tf.shape(copy_valid_encodes)[0]
            candidate_masks = tf.not_equal(random_ids, num_copys-1)
            random_onehots = tf.one_hot(random_ids, num_copys)
            candidate_encodes = tf.matmul(random_onehots, copy_valid_encodes)
            context_token_logits += matcher(
                (pick_encodes, None, 'context'),
                (candidate_encodes, candidate_masks, 'encode'))

    else:
        num_candidates = num_unique_valids

        # token prior distribution
        token_prior_logits = matcher(
            (field_token_embeds, None, 'latent'),
            (unique_valid_token_embeds, None, 'embed'))

        # local context - token distribution
        match_matrix = model_utils_py3.match_vector(
            pick_token_ids, unique_valid_token_ids)
        match_matrix = tf.cast(match_matrix, tf.float32)
        context_token_labels_sum = tf.reduce_sum(match_matrix)
        context_token_labels = match_matrix * (
            1.0 / context_token_labels_sum)
        context_token_logits = matcher(
            (pick_encodes, None, 'context'),
            (unique_valid_token_embeds, None, 'embed'))

        # sample latent - token distribution
        if not global_matcher is None:
            sample_token_logits = global_matcher(
                (global_encodes, None, 'context'),
                (unique_valid_token_embeds, None, 'embed'))
            match_matrix = model_utils_py3.match_vector(
                tf.reshape(token_ids, [batch_size*seq_length, -1]),
                unique_valid_token_ids)
            match_matrix = tf.reshape(
                match_matrix, [batch_size, seq_length, num_unique_valids])
            match_matrix = tf.logical_and(
                match_matrix, tf.expand_dims(valid_masks, axis=2))
            sample_token_labels = tf.cast(
                tf.reduce_any(match_matrix, axis=1), tf.float32)
            sample_token_labels_sum = tf.reduce_sum(sample_token_labels)
            sample_token_labels *= 1.0 / sample_token_labels_sum

            context_token_logits -= tf.stop_gradient(tf.log(
                tf.reduce_sum(
                    tf.exp(sample_token_logits)*sample_token_labels,
                    axis=0, keepdims=True)))
            context_token_logits -= tf.stop_gradient(tf.reduce_sum(
                sample_context_logits * pick_sample_onehots,
                axis=1, keepdims=True))
        context_token_logits -= tf.stop_gradient(token_prior_logits)
        context_token_logits -= tf.stop_gradient(context_prior_logits)

        # copy part
        if not (copy_token_ids is None or copy_encodes is None or copy_masks is None):
            copy_match_matrix = model_utils_py3.match_vector(
                token_ids, copy_token_ids)
            copy_match_matrix = tf.logical_and(
                copy_match_matrix, tf.expand_dims(copy_masks, axis=1))
            copy_scores = tf.cast(copy_match_matrix, tf.float32)
            copy_pad_score = tf.reduce_sum(copy_scores, axis=2, keepdims=True)
            copy_pad_score = tf.maximum(copy_pad_score, 1e-12)
            copy_scores = tf.concat([copy_scores, copy_pad_score], axis=2)
            copy_encodes_padded = tf.pad(copy_encodes, [[0,0],[0,1],[0,0]])
            copy_length = tf.shape(copy_encodes_padded)[1]
            copy_scores = tf.reshape(
                copy_scores, [batch_size*seq_length, copy_length])
            copy_indices = tf.random.categorical(
                tf.log(copy_scores), 1, dtype=tf.int32)
            copy_indices = tf.reshape(copy_indices, [batch_size, seq_length])
            candidate_masks = tf.not_equal(copy_indices, copy_length-1)
            copy_onehots = tf.one_hot(copy_indices, copy_length)
            candidate_encodes = tf.matmul(copy_onehots, copy_encodes_padded)
            _, candidate_encodes = tf.dynamic_partition(
                candidate_encodes, unique_valid_masks_int, 2)
            candidate_masks = tf.boolean_mask(
                candidate_masks, unique_valid_masks)
            context_token_logits += matcher(
                (pick_encodes, None, 'context'),
                (candidate_encodes, candidate_masks, 'encode'))

    def get_loss():
        # local context prior loss
        context_prior_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(context_prior_logits), logits=context_prior_logits)
        context_prior_loss = tf.reduce_mean(context_prior_loss)
        # token prior loss
        if limited_vocab:
            token_prior_loss = 0.0
        else:
            token_prior_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(token_prior_logits), logits=token_prior_logits)
            token_prior_loss = tf.reduce_mean(token_prior_loss)
        if not global_matcher is None:
            # sample latent - local context loss
            labels = tf.reshape(
                sample_context_labels, [num_picks*batch_size])
            logits = tf.reshape(
                sample_context_logits, [num_picks*batch_size])
            sample_context_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(labels),
                logits=logits)
            sample_context_loss -= tf.log(sample_context_labels_sum)
            # sample latent - token loss
            labels = tf.reshape(
                sample_token_labels, [batch_size*num_candidates])
            logits = tf.reshape(
                sample_token_logits, [batch_size*num_candidates])
            sample_token_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(labels),
                logits=logits)
            sample_token_loss -= tf.log(sample_token_labels_sum)
        else:
            sample_context_loss, sample_token_loss = .0, .0
        # local context - token loss
        logits = tf.reshape(
            context_token_logits, [num_picks*num_candidates])
        labels = tf.reshape(
            context_token_labels, [num_picks*num_candidates])
        context_token_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(labels),
            logits=logits)
        context_token_loss -= tf.log(context_token_labels_sum)
        # total loss
        loss = context_token_loss + 0.1*(
            sample_context_loss+sample_token_loss+context_prior_loss+token_prior_loss)
        return loss

    loss = tf.cond(
        tf.reduce_any(pick_masks),
        get_loss,
        lambda: 0.0)

    if not extra_loss_fn is None:
        loss += extra_loss_fn(**locals())

    return loss

def train_seq(
    initial_state,
    cell, matcher,
    limited_vocab,
    field_context_embeds, field_token_embeds,
    token_ids, token_embeds, valid_masks,
    global_encodes=None, global_matcher=None,
    candidate_ids=None, candidate_embeds=None, target_seqs=None,
    copy_token_ids=None, copy_encodes=None, copy_masks=None,
    extra_loss_fn=None):
    """
    args:
        initial_state: state passed to the cell
        cell: a cell object
    """
    inputs = token_embeds[:,:-1]
    outputs, state = cell(inputs, initial_state)
    token_ids = token_ids[:,1:]
    token_embeds = token_embeds[:,1:]
    encodes = outputs
    valid_masks = tf.logical_and(
        valid_masks[:,:-1],
        valid_masks[:,1:])
    pick_masks = valid_masks
    if not target_seqs is None:
        target_seqs = target_seqs[:,1:]
    loss = train_masked(
        matcher,
        limited_vocab,
        field_context_embeds, field_token_embeds,
        token_ids, token_embeds, encodes,
        valid_masks, pick_masks,
        global_encodes, global_matcher,
        candidate_ids, candidate_embeds, target_seqs,
        copy_token_ids, copy_encodes, copy_masks,
        extra_loss_fn,
    )

    return loss

class SpellerTrainer(object):
    """get the loss of a speller"""
    def __init__(self, scope,
                 spellin_embedding, speller_context_embedding,
                 speller_cell, speller_matcher,
                 training=False):
        """
        args:
            spellin_embedding: input embedding for spelling
            speller_context_embedding: 1 x speller_match_size
            speller_cell: speller cell
            speller_matcher: matcher module
        """
        self.spellin_embedding = spellin_embedding
        self.speller_context_embedding = speller_context_embedding
        self.speller_cell = speller_cell
        self.speller_matcher = speller_matcher
        with tf.variable_scope(scope) as sc:
            self.scope = sc

    def __call__(
        self,
        word_encodes,
        target_seqs,
        ):
        """
        get the loss of speller
        args:
            word_encodes: batch_size x word_layer_size
            target_seqs: batch_size x dec_seq_length
        """
        with tf.variable_scope(self.scope, reuse=True):
            batch_size = tf.shape(target_seqs)[0]
            seq_length = tf.shape(target_seqs)[1]+2
            vocab_size = tf.shape(self.spellin_embedding)[0]
            vocab_dim = self.spellin_embedding.get_shape()[1].value

            initial_state = SpellerState(
                word_encodes=word_encodes,
                dec_masks=None,
                dec_keys=None,
                dec_values=None)
            target_seqs = tf.pad(target_seqs, [[0,0],[2,2]])
            valid_masks = tf.logical_or(
                tf.not_equal(target_seqs[:,:-2], 0),
                tf.not_equal(target_seqs[:,2:], 0))
            target_seqs = target_seqs[:,1:-1]
            token_ids = tf.expand_dims(target_seqs, axis=2)
            token_onehots = tf.one_hot(
                tf.reshape(target_seqs, [batch_size*seq_length]),
                vocab_size)
            token_embeds = tf.matmul(
                token_onehots, self.spellin_embedding)
            token_embeds = tf.reshape(
                token_embeds, [batch_size, seq_length, vocab_dim])
            loss = train_seq(
                initial_state,
                self.speller_cell, self.speller_matcher,
                True,
                self.speller_context_embedding, None,
                token_ids, token_embeds, valid_masks,
                candidate_ids=tf.expand_dims(tf.range(vocab_size), axis=1),
                candidate_embeds=self.spellin_embedding,
                target_seqs=target_seqs,
            )

        return loss

class WordTrainer(object):
    """own a speller_trainer and """
    def __init__(self, scope,
                 matcher, global_matcher, speller_trainer,
                 training=False):
        """
        args:
            speller_trainer: a SpellerTrainer object
        """
        self.matcher = matcher
        self.global_matcher = global_matcher
        self.speller_trainer = speller_trainer
        with tf.variable_scope(scope) as sc:
            self.scope = sc
        def speller_loss_fn(**kwargs):
            pick_masks = kwargs['pick_masks']
            pick_encodes = kwargs['pick_encodes']
            pick_target_seqs = kwargs['pick_token_ids']
            def get_speller_loss():
                speller_loss = self.speller_trainer(
                    pick_encodes,
                    pick_target_seqs)
                return speller_loss
            speller_loss = tf.cond(
                tf.reduce_any(pick_masks),
                get_speller_loss,
                lambda: 0.0)
            return 0.5*speller_loss
        self.speller_loss_fn = speller_loss_fn

    def __call__(
        self,
        limited_vocab,
        field_context_embeds, field_word_embeds,
        word_ids, word_embeds, encodes,
        valid_masks, pick_masks,
        global_encodes,
        candidate_ids=None, candidate_embeds=None, target_seqs=None,
        copy_word_ids=None, copy_encodes=None, copy_masks=None):
        """
        add speller loss if not limited_vocab
        """
        with tf.variable_scope(self.scope, reuse=True):
            speller_loss_fn = None if limited_vocab else self.speller_loss_fn
            loss = train_masked(
                self.matcher,
                limited_vocab,
                field_context_embeds, field_word_embeds,
                word_ids, word_embeds, encodes,
                valid_masks, pick_masks,
                global_encodes, self.global_matcher,
                candidate_ids, candidate_embeds, target_seqs,
                copy_word_ids, copy_encodes, copy_masks,
                extra_loss_fn=speller_loss_fn,
            )

        return loss

class SentTrainer(object):
    """get the loss of a sent"""
    def __init__(self, scope,
                 cell, word_trainer,
                 training=False):
        """
        args:
            word_trainer: WordTrainer object
        """
        self.cell = cell
        self.word_trainer = word_trainer
        with tf.variable_scope(scope) as sc:
            self.scope = sc

    def __call__(
        self,
        initial_state,
        limited_vocab,
        field_context_embeds, field_word_embeds,
        word_ids, word_embeds, valid_masks,
        global_encodes,
        candidate_ids=None, candidate_embeds=None, target_seqs=None,
        copy_word_ids=None, copy_encodes=None, copy_masks=None):
        """
        feed the inputs into cell, get the outputs, and then get the loss
        """
        with tf.variable_scope(self.scope, reuse=True):
            speller_loss_fn = None if limited_vocab else self.word_trainer.speller_loss_fn
            loss = train_seq(
                initial_state,
                self.cell, self.word_trainer.matcher,
                limited_vocab,
                field_context_embeds, field_word_embeds,
                word_ids, word_embeds, valid_masks,
                global_encodes,
                candidate_ids, candidate_embeds, target_seqs,
                copy_word_ids, copy_encodes, copy_masks,
                extra_loss_fn=speller_loss_fn,
            )

        return loss


"""Generator"""

def concat_candidates(encodes, candidates_fn_list):
    """
    concat candidates return by candidates_fn_list
    args:
        encodes: batch_size x encode_dim
        candidates_fn_list: 
            return:
                candidate_embeds: [batch_size x ]num_candidates x input_dim
                candidate_ids: [batch_size x ]num_candidates [x word_len]
                candidate_masks: batch_size x num_candidates
                logits: batch_size x num_candidates
    """
    concated_embeds, concated_ids, concated_masks, concated_logits = None, None, None, None
    for candidates_fn in candidates_fn_list:

        candidate_embeds, candidate_ids, candidate_masks, logits \
            = candidates_fn(encodes)

        if candidate_masks is None:
            candidate_masks = tf.ones_like(logits, dtype=tf.bool)

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

def generate_seq(
    decoder,
    cell,
    initial_state,
    state_si_fn,
    length,
    candidates_fn_list,
    start_embedding,
    start_id):
    """
    args:
        decoder: a recursive decoder
        cell: a cell object
        initial_state:
        state_si_fn: state shape invariants
        length:
        candidates_fn_list:
            a list of candidates_fn
            args:
                encodes: cell outputs
            return:
                candidate_embeds: [batch_size x ]num_candidates x input_dim
                candidate_ids: [batch_size x ]num_candidates [x word_len]
                candidate_masks: batch_size x num_candidates
                logits: batch_size x num_candidates
        start_embedding: input_dim
        start_id: id_len
    return:
        seqs: batch_size x num_candidates x length [x word_len]
        scores: batch_size x num_candidates
    """
    candidates_callback = lambda encodes: concat_candidates(encodes, candidates_fn_list)

    seqs, scores = decoder(
        length, initial_state, state_si_fn, cell, candidates_callback, start_embedding, start_id)

    return seqs, scores

class WordGenerator(object):
    """
    generate word
    """
    def __init__(self, speller_cell, speller_matcher,
                 spellin_embedding, sep_id):
        """
        args:
            speller_cell: speller cell
            speller_matcher: matcher for speller
            spellin_embedding: 0 is for start/end, others are embeddings for characters
            sep_id: id for special sep token
        """
        self.speller_cell = speller_cell
        self.speller_matcher = speller_matcher
        self.sep_id = sep_id
        self.pad_embedding = spellin_embedding[0]
        self.nosep_embedding = tf.concat(
            [spellin_embedding[:sep_id], spellin_embedding[sep_id+1:]], axis=0)
        self.nosep_ids = tf.concat(
            [tf.range(sep_id, dtype=tf.int32), 
             tf.range(sep_id+1, tf.shape(spellin_embedding)[0], dtype=tf.int32)],
            axis=0)
        speller_matcher.cache_embeds(self.nosep_embedding)
        self.decoder = lambda *args: model_utils_py3.stochastic_beam_dec(
            *args, beam_size=8, num_candidates=4, cutoff_size=4, gamma=1.0)

    def generate(
        self,
        initial_state,
        length):
        """
        args:
            initial_state: SpellerState
            length:
        """
        batch_size = tf.shape(initial_state.word_encodes)[0]
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

        self.speller_cell.cache_encodes(initial_state.word_encodes)

        def candidates_fn(encodes):
            context_token_logits = self.speller_matcher(
                (encodes, None, 'context'),
                (self.nosep_embedding, None, 'embed'))
            return self.nosep_embedding, self.nosep_ids, None, context_token_logits

        start_embedding = self.pad_embedding

        seqs, scores = generate_seq(
            self.decoder, self.speller_cell,
            initial_state, state_si_fn, length, [candidates_fn],
            start_embedding, tf.zeros([], dtype=tf.int32))

        return seqs[:,:,1:], scores

class SentGenerator(object):
    """
    generate sent
    """
    def __init__(self,
                 word_cell,
                 word_matcher, global_matcher,
                 word_embedder,
                 word_generator):
        """
        args:
            word_cell: word level cell
            word_matcher: word level matcher
            global_matcher: global level matcher
            word_embedder: word level embedder
            word_generator: WordGenerator object
        """
        self.word_cell = word_cell
        self.word_matcher = word_matcher
        self.global_matcher = global_matcher
        self.word_embedder = word_embedder
        self.word_generator = word_generator
        self.decoder = lambda *args: model_utils_py3.stochastic_beam_dec(
            *args, beam_size=4, num_candidates=1, cutoff_size=4, gamma=8.0)

    def generate(
        self,
        initial_state,
        length,
        global_encodes,
        word_embedding, word_ids,
        gen_word_len=None,
        copy_embeds=None, copy_ids=None, copy_masks=None, copy_encodes=None):
        """
        args:
            initial_state: TransformerState
            length:
            global_encodes: batch_size x encode_dim
            word_embedding: num_words x embed_dim
            word_ids: num_words, no sep when in char mode, first one is sep when in word mode
            gen_word_len: int
            copy_embeds: batch_size x num_words x embed_dim
            copy_ids: batch_size x num_words [x word_len]
            copy_masks: batch_size x num_words
            copy_embeds_matcher: batch_size x num_words x matcher_dim
        """
        batch_size = tf.shape(global_encodes)[0]
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

        global_latents = self.global_matcher.cache_contexts(global_encodes)

        start_id = word_ids[0]
        start_embedding = word_embedding[0]
        # static vocab
        if len(word_ids.get_shape()) == 2:
            word_ids = tf.pad(
                word_ids, [[0,0],[0,max_word_len-tf.shape(word_ids)[1]]])
        self.global_matcher.cache_embeds(word_embedding)
        self.word_matcher.cache_embeds(word_embedding)
        sample_token_logits = self.global_matcher(
            (global_encodes, None, 'context'),
            (word_embedding, None, 'embed'))
        def candidates_fn(encodes):
            encode_dim = encodes.get_shape()[-1].value
            encodes_reshaped = tf.reshape(
                encodes, [batch_size, -1, encode_dim])
            beam_size = tf.shape(encodes_reshaped)[1]
            sample_token_logits_tiled = tf.tile(
                tf.expand_dims(sample_token_logits, axis=1),
                [1,beam_size,1])
            sample_context_logits = self.global_matcher(
                (encodes_reshaped, None, 'encode'),
                (tf.expand_dims(global_latents, axis=1), None, 'latent'))
            context_token_logits = self.word_matcher(
                (encodes, None, 'context'),
                (word_embedding, None, 'embed'))
            sample_level_logits = tf.reshape(
                sample_token_logits_tiled + sample_context_logits,
                [batch_size*beam_size, -1])
            logits = context_token_logits + sample_level_logits
            return word_embedding, word_ids, None, logits
        candidates_fn_list.append(candidates_fn)

        # dynamic gen
        if not gen_word_len is None:
            def candidates_fn(encodes):
                encode_dim = encodes.get_shape()[-1].value
                batch_beam_size = tf.shape(encodes)[0]
                encodes_reshaped = tf.reshape(
                    encodes, [batch_size, -1, encode_dim])
                beam_size = tf.shape(encodes_reshaped)[1]
                speller_encoder = self.word_generator.speller_cell.assets['encoder']
                null_tfstruct = model_utils_py3.init_tfstruct(
                    batch_beam_size, speller_encoder.embed_size, speller_encoder.posit_size,
                    speller_encoder.layer_size, speller_encoder.num_layers)
                speller_initial_state = SpellerState(
                    word_encodes=encodes,
                    dec_masks=tf.zeros([batch_beam_size, 0], dtype=tf.bool),
                    dec_keys=(
                        tf.zeros([batch_beam_size, 0, speller_encoder.layer_size]),
                    )*speller_encoder.num_layers,
                    dec_values=(
                        tf.zeros([batch_beam_size, 0, speller_encoder.layer_size]),
                    )*speller_encoder.num_layers)
                word_ids, word_scores = self.word_generator.generate(
                    speller_initial_state, max_word_len)
                word_scores = tf.exp(word_scores)
                word_embeds, word_masks = self.word_embedder(word_ids)
                word_masks = tf.logical_and(
                    word_masks,
                    tf.not_equal(word_scores, 0.0))
                word_ids = tf.pad(word_ids, [[0,0],[0,0],[0,max_word_len-tf.shape(word_ids)[2]]])
                num_candidates = tf.shape(word_ids)[1]

                sample_token_logits = self.global_matcher(
                    (tf.reshape(
                        word_embeds,
                        [batch_size, beam_size*num_candidates, self.word_embedder.layer_size]),
                     None, 'embed'),
                    (tf.expand_dims(
                        global_latents, axis=1),
                     None, 'latent'))
                sample_token_logits = tf.reshape(
                    sample_token_logits, [batch_size*beam_size, num_candidates])
                sample_context_logits = self.global_matcher(
                    (encodes_reshaped, None, 'encode'),
                    (tf.expand_dims(
                        global_latents, axis=1),
                     None, 'latent'))
                sample_context_logits = tf.reshape(
                    sample_context_logits, [batch_size*beam_size, 1])
                context_token_logits = self.word_matcher(
                    (tf.expand_dims(encodes, axis=1), None, 'context'),
                    (word_embeds, None, 'embed'))
                context_token_logits = tf.squeeze(
                    context_token_logits, [1])
                logits = sample_context_logits + sample_token_logits + context_token_logits
                return word_embeds, word_ids, word_masks, logits
            candidates_fn_list.append(candidates_fn)

        # copy
        if not (copy_embeds is None or copy_ids is None or copy_masks is None or copy_encodes is None):
            embed_dim = copy_embeds.get_shape()[-1].value
            num_copies = tf.shape(copy_embeds)[1]
            copy_ids = tf.pad(
                copy_ids, [[0,0],[0,0],[0,max_word_len-tf.shape(copy_ids)[2]]])
            self.global_matcher.cache_embeds(copy_embeds)
            copy_embed_projs = self.word_matcher.cache_embeds(copy_embeds)
            copy_encode_projs = self.word_matcher.cache_encodes(copy_encodes)
            sample_token_logits = self.global_matcher(
                (tf.expand_dims(global_latents, axis=1), None, 'latent'),
                (copy_embeds, None, 'embed'))
            def candidates_fn(encodes):
                encode_dim = encodes.get_shape()[-1].value
                encodes_reshaped = tf.reshape(
                    encodes, [batch_size, -1, encode_dim])
                beam_size = tf.shape(encodes_reshaped)[1]
                sample_token_logits_tiled = tf.tile(
                    sample_token_logits, [1,beam_size,1])
                sample_context_logits = self.global_matcher(
                    (encodes_reshaped, None, 'encode'),
                    (tf.expand_dims(global_latents, axis=1), None, 'latent'))
                sample_level_logits = tf.reshape(
                    sample_token_logits_tiled + sample_context_logits,
                    [batch_size*beam_size, num_copies])

                copy_embeds_tiled = tf.tile(
                    tf.expand_dims(copy_embeds, axis=1),
                    [1,beam_size,1,1])
                copy_embeds_tiled = tf.reshape(
                    copy_embeds_tiled,
                    [batch_size*beam_size, num_copies, embed_dim])
                if len(copy_ids.get_shape()) == 3:
                    copy_ids_tiled = tf.tile(
                        tf.expand_dims(copy_ids, axis=1),
                        [1,beam_size,1,1])
                    copy_ids_tiled = tf.reshape(
                        copy_ids_tiled,
                        [batch_size*beam_size, num_copies, -1])
                elif len(copy_ids.get_shape()) == 2:
                    copy_ids_tiled = tf.tile(
                        tf.expand_dims(copy_ids, axis=1),
                        [1,beam_size,1])
                    copy_ids_tiled = tf.reshape(
                        copy_ids_tiled,
                        [batch_size*beam_size, num_copies])
                copy_masks_tiled = tf.tile(
                    tf.expand_dims(copy_masks, axis=1),
                    [1,beam_size,1])
                copy_masks_tiled = tf.reshape(
                    copy_masks_tiled,
                    [batch_size*beam_size, num_copies])
                copy_embed_projs_tiled = tf.tile(
                    tf.expand_dims(copy_embed_projs, axis=1),
                    [1,beam_size,1,1])
                copy_embed_projs_tiled = tf.reshape(
                    copy_embed_projs_tiled,
                    [batch_size*beam_size, num_copies, self.word_matcher.layer_size])
                copy_encode_projs_tiled = tf.tile(
                    tf.expand_dims(copy_encode_projs, axis=1),
                    [1,beam_size,1,1])
                copy_encode_projs_tiled = tf.reshape(
                    copy_encode_projs_tiled,
                    [batch_size*beam_size, num_copies, self.word_matcher.layer_size])
                encodes_expanded = tf.expand_dims(encodes, axis=1)
                context_token_logits = self.word_matcher(
                    (encodes_expanded, None, 'context')
                    (copy_embed_projs_tiled, None, 'latent'))
                copy_logits = self.word_matcher(
                    (encodes_expanded, None, 'context'),
                    (copy_encode_projs_tiled, None, 'latent'))
                context_token_logits += copy_logits
                context_token_logits = tf.squeeze(
                    context_token_logits, [1])
                logits = context_token_logits + sample_level_logits
                return copy_embeds_tiled, copy_ids_tiled, copy_masks_tiled, logits
            candidates_fn_list.append(candidates_fn)

        seqs, scores = generate_seq(
            self.decoder, self.word_cell,
            initial_state, state_si_fn, length-1,
            candidates_fn_list,
            start_embedding, start_id)
        if len(seqs.get_shape()) == 3:
            seqs = tf.pad(seqs, [[0,0],[0,0],[0,length-tf.shape(seqs)[2]]])
        elif len(seqs.get_shape()) == 4:
            seqs = tf.pad(seqs, [[0,0],[0,0],[0,length-tf.shape(seqs)[2]],[0,0]])

        return seqs, scores

class ClassGenerator(object):
    """
    generate class token
    """
    def __init__(self,
                 word_encoder,
                 word_matcher, global_matcher,
                 word_embedder,
                 word_generator):
        """
        args:
            word_cell: word level cell
            word_matcher: word level matcher
            word_embedder: word level embedder
            word_generator:
        """
        self.encoder = word_encoder
        self.word_matcher = word_matcher
        self.global_matcher = global_matcher
        self.word_embedder = word_embedder
        self.word_generator = word_generator

    def generate(self,
                 tfstruct,
                 extra_tfstruct,
                 global_encodes,
                 word_embedding=None, word_ids=None,
                 gen_word_len=None,
                 copy_embeds=None, copy_ids=None, copy_masks=None, copy_encodes=None):
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

        candidates_fn_list = []
        attn_masks = tf.zeros([batch_size, 1, 1], tf.bool)
        if not extra_tfstruct is None:
            attn_masks = tf.concat([attn_masks, tf.expand_dims(extra_tfstruct.masks, axis=1)], axis=2)
        tfstruct = self.encoder(tfstruct, attn_masks, extra_tfstruct)
        word_encodes = tfstruct.encodes
        word_encodes = tf.squeeze(word_encodes, [1])
        global_latents = self.global_matcher.cache_contexts(global_encodes)

        # static vocab
        if not (word_embedding is None or word_ids is None):
            if len(word_ids.get_shape()) == 2:
                word_ids = tf.pad(
                    word_ids, [[0,0],[0,max_word_len-tf.shape(word_ids)[1]]])
            sample_token_logits = self.global_matcher(
                (global_encodes, None, 'context'),
                (word_embedding, None, 'embed'))
            def candidates_fn(encodes):
                context_token_logits = self.word_matcher(
                    (encodes, None, 'context'),
                    (word_embedding, None, 'embed'))
                logits = sample_token_logits + context_token_logits
                return word_embedding, word_ids, None, logits
            candidates_fn_list.append(candidates_fn)

        # dynamic gen
        if not gen_word_len is None:
            def candidates_fn(encodes):
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
                word_ids, word_scores = self.word_generator.generate(
                    speller_initial_state, max_word_len)
                word_scores = tf.exp(word_scores)
                word_embeds, word_masks = self.word_embedder(word_ids)
                word_masks = tf.logical_and(
                    word_masks,
                    tf.not_equal(word_scores, 0.0))
                word_ids = tf.pad(word_ids, [[0,0],[0,0],[0,max_word_len-tf.shape(word_ids)[2]]])
                num_candidates = tf.shape(word_ids)[1]

                sample_token_logits = self.global_matcher(
                    (tf.expand_dims(global_latents, axis=1), None, 'latent'),
                    (word_embeds, None, 'embed'))
                context_token_logits = self.word_matcher(
                    (tf.expand_dims(encodes, axis=1), None, 'context'),
                    (word_embeds, None, 'embed'))
                logits = tf.squeeze(sample_token_logits+context_token_logits, [1])
                return word_embeds, word_ids, word_masks, logits
            candidates_fn_list.append(candidates_fn)

        # copy
        if not (copy_embeds is None or copy_ids is None or copy_masks is None or copy_encodes is None):
            sample_token_logits = self.global_matcher(
                (tf.expand_dims(global_latents, axis=1), None, 'latent'),
                (copy_embeds, None, 'embed'))
            def candidates_fn(encodes):
                encodes_expanded = tf.expand_dims(encodes, axis=1)
                context_token_logits = self.word_matcher(
                    (encodes_expanded, None, 'context')
                    (copy_embeds, None, 'embed'))
                copy_logits = self.word_matcher(
                    (encodes_expanded, None, 'context'),
                    (copy_encodes, None, 'encode'))
                context_token_logits += copy_logits
                context_token_logits = tf.squeeze(
                    context_token_logits, [1])
                logits = context_token_logits + sample_token_logits
                logits += tf.log(tf.cast(copy_masks, tf.float32))
                return copy_embeds, copy_ids, copy_masks, logits
            candidates_fn_list.append(candidates_fn)

        concat_embeds, concat_ids, concat_masks, concat_logits = concat_candidates(
            word_encodes, candidates_fn_list)

        indices = tf.argmax(concat_logits, 1, output_type=tf.int32)
        batch_indices = tf.stack([tf.range(batch_size, dtype=tf.int32), indices], axis=1)
        if len(concat_embeds.get_shape()) == 2:
            classes = tf.gather(concat_ids, indices)
        elif len(concat_embeds.get_shape()) == 3:
            classes = tf.gather_nd(concat_ids, batch_indices)
        scores = tf.gather_nd(concat_logits, batch_indices)
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
            field_query_embedding, \
            field_key_embedding, \
            field_value_embedding  = self.cache_encodes(word_encodes)

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
                    field_query_embedding,
                    field_key_embedding,
                    field_value_embedding)
            else:
                field_query_embedding, \
                field_key_embedding, \
                field_value_embedding = self.cached[word_encodes]

        return (field_query_embedding,
                field_key_embedding,
                field_value_embedding)


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
        """
        self.layer_size = layer_size
        self.cached_contexts, self.cached_encodes, self.cached_embeds = {},{},{}
        self.context_reuse, self.encode_reuse, self.embed_reuse = None, None, None
        Module.__init__(self, scope, dropout, training)

    def __call__(self, *pair):
        """
        args:
            pair: (item1, item2), each item is a tuple or list of tuples,
                each tuple is (tensor_to_match, tensor_masks, type),
                type is one of {'macro', 'context', 'token', 'latent'} object
        returns:
            logits: shape1 x shape2
        """
        assert(len(pair) == 2)
        with tf.variable_scope(self.scope, reuse=self.reuse):

            def project(inputs, masks, input_type):
                if input_type == 'context':
                    projs = self.cache_contexts(inputs)
                elif input_type == 'encode':
                    projs = self.cache_encodes(inputs)
                elif input_type == 'embed':
                    projs = self.cache_embeds(inputs)
                elif input_type == 'latent':
                    projs = inputs
                else:
                    projs = None
                if not masks is None:
                    projs *= tf.expand_dims(
                        tf.cast(masks, tf.float32), axis=-1)
                return projs

            latents = []
            for item in pair:
                if isinstance(item, list):
                    projs_list = [project(*i) for i in item]
                    projs = projs_list[0]
                    for p in projs_list[1:]:
                        projs += p
                else:
                    projs = project(*item)
                latents.append(projs)

            latents = tf.cond(
                tf.greater(tf.size(latents[0]), tf.size(latents[1])),
                lambda: (latents[0], latents[1]*(float(self.layer_size)**-0.5)),
                lambda: (latents[0]*(float(self.layer_size)**-0.5), latents[1]))

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

    def cache_contexts(self, contexts):
        """
        args:
            contexts: num_candidates x encode_dim or batch_size x num_candidates x encode_dim
        """
        with tf.variable_scope(self.scope, reuse=self.context_reuse):

            if self.cached_contexts.get(contexts) is None:
                context_projs = model_utils_py3.GLU(
                    contexts,
                    self.layer_size,
                    is_training=self.training,
                    scope="context_projs")
                self.cached_contexts[contexts] = context_projs
            else:
                context_projs = self.cached_contexts[contexts]

        self.context_reuse = True

        return context_projs

    def cache_encodes(self, encodes):
        """
        args:
            encodes: num_candidates x encode_dim or batch_size x num_candidates x encode_dim
        """
        with tf.variable_scope(self.scope, reuse=self.encode_reuse):

            if self.cached_encodes.get(encodes) is None:
                encode_projs = model_utils_py3.fully_connected(
                    encodes,
                    self.layer_size,
                    is_training=self.training,
                    scope="encode_projs")
                self.cached_encodes[encodes] = encode_projs
            else:
                encode_projs = self.cached_encodes[encodes]

        self.encode_reuse = True

        return encode_projs

    def cache_embeds(self, embeds):
        """
        args:
            embeds: num_candidates x embed_dim or batch_size x num_candidates x embed_dim
        """
        with tf.variable_scope(self.scope, reuse=self.embed_reuse):

            if self.cached_embeds.get(embeds) is None:
                embed_projs = model_utils_py3.fully_connected(
                    embeds,
                    self.layer_size,
                    is_training=self.training,
                    scope="embed_projs")
                self.cached_embeds[embeds] = embed_projs
            else:
                embed_projs = self.cached_embeds[embeds]

        self.embed_reuse = True

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
