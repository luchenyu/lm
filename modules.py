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

class GeneralTrainer(object):
    """get the loss of masked slots"""
    def __init__(self, scope,
                 matcher,
                 training=False):
        """
        args:
            scope: str
            matcher: a matcher object
        """
        self.matcher = matcher
        with tf.variable_scope(scope) as sc:
            self.scope = sc
        self.training = training

    def __call__(
        self,
        limited_vocab,
        sample_latent_embeds,
        field_context_embeds, field_token_embeds,
        token_ids, token_embeds, encodes,
        valid_masks, pick_masks,
        candidate_ids=None, candidate_embeds=None, target_seqs=None,
        copy_token_ids=None, copy_encodes=None, copy_masks=None,
        extra_loss_fn=None):
        """
        get the loss
        args:
            limited_vocab: bool
            sample_latent_embeds: batch_size x macro_size
            field_context_embeds: 1 x match_size
            token_ids: batch_size x seq_length x word_len
            token_embeds: batch_size x seq_length x embed_dim
            encodes: batch_size x seq_length x encode_dim
            valid_masks: batch_size x seq_length
            pick_masks: batch_size x seq_length
            field_token_embeds: 1 x match_size
            candidate_ids: num_candidates x word_len
            candidate_embeds: num_candidates x embed_dim
            target_seqs: batch_size x seq_length
            copy_token_ids: batch_size x copy_seq_length x word_len
            copy_encodes: batch_size x copy_seq_length x encode_dim
            copy_masks: batch_size x copy_seq_length
            hook_fn: fn to get the local vars here
        """
        with tf.variable_scope(self.scope, reuse=True):

            batch_size = tf.shape(token_ids)[0]
            seq_length = tf.shape(token_ids)[1]
            valid_masks_int = tf.cast(valid_masks, tf.int32)
            pick_masks_int = tf.cast(pick_masks, tf.int32)

            sample_ids = tf.tile(
                tf.expand_dims(tf.range(batch_size), axis=1), [1,seq_length])
            grid_x, grid_y = tf.meshgrid(tf.range(seq_length), tf.range(batch_size))
            grid_ids = tf.stack([grid_y, grid_x], axis=2)

            valid_token_ids = tf.boolean_mask(
                token_ids, valid_masks)
            valid_sample_ids = tf.boolean_mask(
                sample_ids, valid_masks)
            valid_grid_ids = tf.boolean_mask(
                grid_ids, valid_masks)
            _, valid_token_embeds = tf.dynamic_partition(
                token_embeds, valid_masks_int, 2)
            num_valids = tf.shape(valid_token_ids)[0]

            pick_token_ids = tf.boolean_mask(
                token_ids, pick_masks)
            pick_sample_ids = tf.boolean_mask(
                sample_ids, pick_masks)
            pick_grid_ids = tf.boolean_mask(
                grid_ids, pick_masks)
            _, pick_encodes = tf.dynamic_partition(
                encodes, pick_masks_int, 2)
            _, pick_token_embeds = tf.dynamic_partition(
                token_embeds, pick_masks_int, 2)
            if not target_seqs is None:
                pick_target_seqs = tf.boolean_mask(
                    target_seqs, pick_masks)
            num_picks = tf.shape(pick_encodes)[0]

            pick_sample_onehots = tf.one_hot(
                pick_sample_ids, batch_size, axis=0)
            pick_sample_labels_sum = tf.reduce_sum(
                pick_sample_onehots, axis=1, keepdims=True)
            pick_sample_masks = tf.squeeze(
                tf.greater(pick_sample_labels_sum, 0), [1])
            pick_sample_masks_int = tf.cast(
                pick_sample_masks, tf.int32)
            pick_sample_labels = pick_sample_onehots / tf.maximum(
                pick_sample_labels_sum, 1e-12)

            # sample latent - local context distribution
            sample_context_logits = self.matcher(
                (sample_latent_embeds, None, 'macro'),
                (pick_encodes, None, 'context'))

            # sample latent - token distribution
            sample_token_logits = self.matcher(
                (sample_latent_embeds, None, 'macro'),
                (pick_token_embeds, None, 'token'))

            # local context - token distribution
            context_prior_logits = self.matcher(
                (pick_encodes, None, 'context'),
                (field_context_embeds, None, 'latent'))
            if limited_vocab:
                num_candidates = tf.shape(candidate_ids)[0]
                match_matrix = tf.one_hot(pick_target_seqs, num_candidates)
                context_token_logits = self.matcher(
                    (pick_encodes, None, 'context'),
                    (candidate_embeds, None, 'token'))
                context_token_logits -= context_prior_logits

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
                    context_token_logits += self.matcher(
                        (pick_encodes, None, 'context'),
                        (candidate_encodes, candidate_masks, 'context'))

            else:
                token_prior_logits = self.matcher(
                    (field_token_embeds, None, 'latent'),
                    (valid_token_embeds, None, 'token'))

                num_candidates = num_valids
                context_token_logits = self.matcher(
                    (pick_encodes, None, 'context'),
                    (valid_token_embeds, None, 'token'))
                context_token_logits -= token_prior_logits
                context_token_logits -= context_prior_logits
                match_matrix = model_utils_py3.match_vector(
                    pick_grid_ids, valid_grid_ids)
                match_matrix = tf.cast(match_matrix, tf.float32)

                # copy part
                if not (copy_token_ids is None or copy_encodes is None or copy_masks is None):
                    copy_length = tf.shape(copy_token_ids)[1]
                    token_ids, copy_token_ids = model_utils_py3.pad_vectors(
                        [token_ids, copy_token_ids])
                    # random choose valid copy encodes
                    copy_match_matrix = model_utils_py3.match_vector(
                        token_ids, copy_token_ids)
                    copy_match_matrix = tf.logical_and(
                        copy_match_matrix,
                        tf.expand_dims(copy_masks, axis=1))
                    copy_scores = tf.cast(copy_match_matrix, tf.float32)
                    copy_scores_pad = tf.reduce_sum(
                        copy_scores, axis=2, keepdims=True)
                    copy_scores_pad = tf.maximum(copy_scores_pad, 1e-12)
                    copy_scores = tf.concat(
                        [copy_scores, copy_scores_pad], axis=2)
                    copy_scores = tf.log(copy_scores)
                    copy_random_ids = tf.random.categorical(
                        tf.reshape(copy_scores, [batch_size*seq_length, copy_length+1]),
                        1, dtype=tf.int32)
                    copy_random_ids = tf.reshape(
                        copy_random_ids, [batch_size, seq_length])
                    copy_random_onehots = tf.one_hot(
                        copy_random_ids, copy_length+1)
                    copy_encodes = tf.pad(copy_encodes, [[0,0],[0,1],[0,0]])
                    random_copy_encodes = tf.matmul(
                        copy_random_onehots, copy_encodes)
                    _, valid_copy_encodes = tf.dynamic_partition(
                        random_copy_encodes, valid_masks_int, 2)
                    context_token_logits += self.matcher(
                        (pick_encodes, None, 'context'),
                        (valid_copy_encodes, None, 'context'))

            context_token_labels_sum = tf.reduce_sum(match_matrix)
            context_token_labels = match_matrix * (1.0/context_token_labels_sum)
            context_token_logits = tf.reshape(
                context_token_logits, [num_picks*num_candidates])
            context_token_labels = tf.reshape(
                context_token_labels, [num_picks*num_candidates])

            def get_loss():
                # local context prior loss
                context_prior_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(context_prior_logits), logits=context_prior_logits)
                context_prior_loss = tf.reduce_mean(context_prior_loss)
                # sample latent - local context loss
                sample_context_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.stop_gradient(pick_sample_labels),
                    logits=sample_context_logits)
                # sample latent - token loss
                sample_token_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.stop_gradient(pick_sample_labels),
                    logits=sample_token_logits)
                sample_context_loss, sample_token_loss, pick_sample_labels_sum_local = \
                tf.cond(
                    tf.reduce_all(pick_sample_masks),
                    lambda: (sample_context_loss, sample_token_loss, pick_sample_labels_sum),
                    lambda: (
                        tf.dynamic_partition(
                            sample_context_loss, pick_sample_masks_int, 2)[1],
                        tf.dynamic_partition(
                            sample_token_loss, pick_sample_masks_int, 2)[1],
                        tf.boolean_mask(
                            pick_sample_labels_sum, pick_sample_masks)),
                )
                log_pick_sample_labels_sum_local = tf.log(
                    tf.squeeze(pick_sample_labels_sum_local, [1]))
                sample_context_loss -= log_pick_sample_labels_sum_local
                sample_context_loss = tf.reduce_mean(sample_context_loss)
                sample_token_loss -= log_pick_sample_labels_sum_local
                sample_token_loss = tf.reduce_mean(sample_token_loss)
                # token prior loss
                if limited_vocab:
                    token_prior_loss = 0.0
                else:
                    token_prior_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones_like(token_prior_logits), logits=token_prior_logits)
                    token_prior_loss = tf.reduce_mean(token_prior_loss)
                # local context - token loss
                context_token_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=tf.stop_gradient(context_token_labels),
                    logits=context_token_logits)
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

class SeqTrainer(object):
    """own a cell and do the training"""
    def __init__(self, scope,
                 cell, trainer,
                 training=False):
        """
        args:
            cell: a cell object
            trainer: a GeneralTrainer object
        """
        self.cell = cell
        self.trainer = trainer
        with tf.variable_scope(scope) as sc:
            self.scope = sc
        self.training = training

    def __call__(
        self,
        limited_vocab,
        initial_state,
        sample_latent_embeds,
        field_context_embeds, field_token_embeds,
        token_ids, token_embeds, valid_masks,
        candidate_ids=None, candidate_embeds=None, target_seqs=None,
        copy_token_ids=None, copy_encodes=None, copy_masks=None,
        extra_loss_fn=None):
        """
        args:
            initial_state: state passed to the cell
        """
        with tf.variable_scope(self.scope, reuse=True):
            inputs = token_embeds[:,:-1]
            outputs, state = self.cell(inputs, initial_state)
            token_ids = token_ids[:,1:]
            token_embeds = token_embeds[:,1:]
            encodes = outputs
            valid_masks = tf.logical_and(
                valid_masks[:,:-1],
                valid_masks[:,1:])
            pick_masks = valid_masks
            if not target_seqs is None:
                target_seqs = target_seqs[:,1:]
        loss = self.trainer(
            limited_vocab,
            sample_latent_embeds,
            field_context_embeds, field_token_embeds,
            token_ids, token_embeds, encodes,
            valid_masks, pick_masks,
            candidate_ids, candidate_embeds, target_seqs,
            copy_token_ids, copy_encodes, copy_masks,
            extra_loss_fn,
        )

        return loss

class SpellerTrainer(SeqTrainer):
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
        trainer = GeneralTrainer(scope, speller_matcher, training)
        SeqTrainer.__init__(self, scope, speller_cell, trainer, training)

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
        loss = SeqTrainer.__call__(
            self,
            True,
            initial_state,
            word_encodes,
            self.speller_context_embedding, None,
            token_ids, token_embeds, valid_masks,
            candidate_ids=tf.expand_dims(tf.range(vocab_size), axis=1),
            candidate_embeds=self.spellin_embedding,
            target_seqs=target_seqs,
        )

        return loss

class WordTrainer(GeneralTrainer):
    """own a speller_trainer and """
    def __init__(self, scope,
                 matcher, speller_trainer,
                 training=False):
        """
        args:
            speller_trainer: a SpellerTrainer object
        """
        self.speller_trainer = speller_trainer
        GeneralTrainer.__init__(self, scope, matcher, training)

    def __call__(
        self,
        limited_vocab,
        sample_encodes,
        field_context_embeds, field_word_embeds,
        word_ids, word_embeds, encodes,
        valid_masks, pick_masks,
        candidate_ids=None, candidate_embeds=None, target_seqs=None,
        copy_word_ids=None, copy_encodes=None, copy_masks=None,
        extra_loss_fn=None):
        """
        add speller loss if not limited_vocab
        """
        if not limited_vocab:
            def speller_loss_fn(**kwargs):
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
                return speller_loss
        else:
            speller_loss_fn = None

        loss = GeneralTrainer.__call__(
            self,
            limited_vocab,
            sample_encodes,
            field_context_embeds, field_word_embeds,
            word_ids, word_embeds, encodes,
            valid_masks, pick_masks,
            candidate_ids, candidate_embeds, target_seqs,
            copy_word_ids, copy_encodes, copy_masks,
            extra_loss_fn=speller_loss_fn,
        )

        return loss

class SentTrainer(SeqTrainer):
    """get the loss of a sent"""
    def __init__(self, scope,
                 cell, word_trainer,
                 training=False):
        """
        args:
            word_trainer: WordTrainer object
        """
        SeqTrainer.__init__(self, scope, cell, word_trainer, training)

    def __call__(
        self,
        limited_vocab,
        initial_state,
        sample_encodes,
        field_context_embeds, field_word_embeds,
        word_ids, word_embeds, valid_masks,
        candidate_ids=None, candidate_embeds=None, target_seqs=None,
        copy_word_ids=None, copy_encodes=None, copy_masks=None,
        extra_loss_fn=None):
        """
        feed the inputs into cell, get the outputs, and then get the loss
        """
        loss = SeqTrainer.__call__(
            self,
            limited_vocab,
            initial_state,
            sample_encodes,
            field_context_embeds, field_word_embeds,
            word_ids, word_embeds, valid_masks,
            candidate_ids, candidate_embeds, target_seqs,
            copy_word_ids, copy_encodes, copy_masks,
            extra_loss_fn,
        )

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

                if candidate_encodes is None:
                    if len(candidate_embeds.get_shape()) == 2:
                        logits = self.matcher(
                            (encodes, None, 'context'),
                            (candidate_embeds, None, 'token'))
                    elif len(candidate_embeds.get_shape()) == 3:
                        logits = self.matcher(
                            (tf.expand_dims(encodes, axis=1), None, 'context'),
                            (candidate_embeds, None, 'token'))
                else:
                    logits = self.matcher(
                        (tf.expand_dims(encodes, axis=1), None, 'context'),
                        [(candidate_embeds, None, 'token'), (candidate_encodes, None, 'context')])
                if len(logits.get_shape()) == 3:
                    logits = tf.squeeze(logits, [1])
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
        speller_matcher.cache_tokens(self.nosep_embedding)
        decoder = lambda *args: model_utils_py3.stochastic_beam_dec(
            *args, beam_size=16, num_candidates=16, cutoff_size=8, gamma=1.0)

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
        decoder = lambda *args: model_utils_py3.stochastic_beam_dec(
            *args, beam_size=4, num_candidates=1, cutoff_size=16, gamma=4.0)

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
            self.matcher.cache_tokens(sep_embedding)
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
                    word_ids, word_scores = self.word_generator.generate(speller_initial_state, max_word_len)
                    word_scores = tf.exp(word_scores)
                    word_embeds, word_masks = self.word_embedder(word_ids)
                    word_masks = tf.logical_and(
                        word_masks,
                        tf.not_equal(word_scores, 0.0))
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
            self.matcher.cache_tokens(word_embedding)
            def candidates_fn(encodes):
                return word_embedding, word_ids, None, None
            candidates_fn_list.append(candidates_fn)

        # copy
        if not (copy_embeds is None or copy_ids is None or copy_masks is None or copy_encodes is None):
            copy_ids = tf.pad(copy_ids, [[0,0],[0,0],[0,max_word_len-tf.shape(copy_ids)[2]]])
            self.matcher.cache_tokens(copy_embeds)
            self.matcher.cache_contexts(copy_encodes)
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
            logits = self.matcher(
                (word_encodes, None, 'context'), (candidate_embeds, None, 'token'))
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
            logits = self.matcher(
                (tf.expand_dims(word_encodes, axis=1), None, 'context'), (candidate_embeds, None, 'token'))
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
        self.cached_macros, self.cached_contexts, self.cached_tokens = {},{},{}
        self.macro_reuse, self.context_reuse, self.token_reuse = None, None, None
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
                if input_type == 'macro':
                    projs = self.cache_macros(inputs)
                elif input_type == 'context':
                    projs = self.cache_contexts(inputs)
                elif input_type == 'token':
                    projs = self.cache_tokens(inputs)
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

    def cache_macros(self, macros):
        """
        args:
            macros: num_candidates x encode_dim or batch_size x num_candidates x encode_dim
        """
        with tf.variable_scope(self.scope, reuse=self.macro_reuse):

            if self.cached_macros.get(macros) is None:
                macro_projs = model_utils_py3.GLU(
                    macros,
                    self.layer_size,
                    is_training=self.training,
                    scope="macro_projs")
                self.cached_macros[macros] = macro_projs
            else:
                macro_projs = self.cached_macros[macros]

        self.macro_reuse = True

        return macro_projs

    def cache_contexts(self, encodes):
        """
        args:
            encodes: num_candidates x encode_dim or batch_size x num_candidates x encode_dim
        """
        with tf.variable_scope(self.scope, reuse=self.context_reuse):

            if self.cached_contexts.get(encodes) is None:
                encode_projs = model_utils_py3.GLU(
                    encodes,
                    self.layer_size,
                    is_training=self.training,
                    scope="enc_projs")
                self.cached_contexts[encodes] = encode_projs
            else:
                encode_projs = self.cached_contexts[encodes]

        self.context_reuse = True

        return encode_projs

    def cache_tokens(self, embeds):
        """
        args:
            embeds: num_candidates x embed_dim or batch_size x num_candidates x embed_dim
        """
        with tf.variable_scope(self.scope, reuse=self.token_reuse):

            if self.cached_tokens.get(embeds) is None:
                embed_projs = model_utils_py3.GLU(
                    embeds,
                    self.layer_size,
                    is_training=self.training,
                    scope="emb_projs")
                self.cached_tokens[embeds] = embed_projs
            else:
                embed_projs = self.cached_tokens[embeds]

        self.token_reuse = True

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
