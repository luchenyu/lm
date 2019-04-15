import json, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import model_utils_py3
from modules import *


"""training hook"""

class LRFinderHook(tf.train.SessionRunHook):
    def __init__(
        self,
        fetches,
        num_steps):
        tf.logging.info('Create LRFinderHook.')
        self.fetches = fetches
        self.num_steps = num_steps
        self.learning_rates = []
        self.losses_smoothed = []
        self.losses = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def begin(self):
        pass

    def after_create_session(self, session, coord):
        pass

    def before_run(self, run_context):        
        return tf.train.SessionRunArgs(self.fetches)

    def after_run(self, run_context, run_values):
        learning_rate = run_values.results['learning_rate']
        loss = run_values.results['loss']
        self.learning_rates.append(learning_rate)
        self.losses.append(loss)
        if len(self.losses_smoothed) == 0:
            self.losses_smoothed.append(loss)
        else:
            self.losses_smoothed.append(0.9*self.losses_smoothed[-1] + 0.1*loss)
        window_size = int(0.05*self.num_steps)
        if len(self.losses_smoothed) > 2*window_size and \
            (min(self.losses_smoothed[-window_size:]) > 1.2*min(self.losses_smoothed) \
            or self.losses_smoothed[-1] >= 1.2*max(self.losses_smoothed[-2*window_size:-window_size])):
            run_context.request_stop()
        elif len(self.learning_rates) % int(self.num_steps / 100) == 0:
            self.ax.clear()
            self.ax.semilogx(self.learning_rates, self.losses_smoothed)
            self.ax.semilogx(self.learning_rates, self.losses)

    def end(self, session):
        self.fig.show()
        self.fig.canvas.draw()


""" lm model """

class Model(object):
    def __init__(self, model_config, char_vocab, train_dir,
                 pretrain_dir=None, vars_to_warm_start='.*'):
        """
        args:
            model_config: {'char_vocab_size': int, 'char_vocab_dim': int, 'char_vocab_emb': np.array,
                'layer_size': int, 'num_layers': int, 'num_heads': int}
        """
        self.model_config = model_config
        self.char_vocab = char_vocab
        self.train_dir = train_dir
        self.warm_start_from = None
        self.isFreeze = False
        if pretrain_dir != None:
            self.warm_start_from = tf.estimator.WarmStartSettings(
                        ckpt_to_initialize_from=pretrain_dir,
                        vars_to_warm_start=vars_to_warm_start)
            if os.path.exists(os.path.join(pretrain_dir, 'model_config')):
                with open(os.path.join(pretrain_dir, 'model_config'), 'r') as json_file:
                    self.model_config = json.load(json_file)
            self.isFreeze = True
        os.makedirs(train_dir, exist_ok=True)
        if os.path.exists(os.path.join(train_dir, 'model_config')):
            with open(os.path.join(train_dir, 'model_config'), 'r') as json_file:
                self.model_config = json.load(json_file)
        else:
            with open(os.path.join(train_dir, 'model_config'), 'w') as json_file:
                json.dump(self.model_config, json_file, indent=4)

    def freeze(self):
        """freeze most parameters"""
        self.isFreeze = True
        return self

    def unfreeze(self):
        """unfreeze all parameters"""
        self.isFreeze = False
        return self

    def get_global_step(self):
        """load global step from ckpt"""
        ckpt = tf.train.get_checkpoint_state(self.train_dir)
        if ckpt != None:
            return int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        else:
            return 0

    def lm_model_fn(
        self,
        features, # This is batch_features from input_fn
        labels,   # This is batch_labels from input_fn
        mode,     # An instance of tf.estimator.ModeKeys
        params):  # Additional configuration
        """
        lm model function for tf.estimator
        features:
            {0:{'seqs':seqs, 'segs':segs},...}
        params:
            schema: [{
                'field_id': int,
                'group_id': int,
                'item_id': int,
                'type': 'sequence'|'class',
                'limited_vocab': bool,
                'copy_from': [field_ids],
                'token_vocab': Vocab,},]
            run_config: {'batch_size': int, 'max_train_steps': int,
                'max_lr': float, 'pct_start': [0,1], 'dropout': [0,1], 'wd': float,
                'data': [{'target_level': int >= 0, 'max_token_length': int, 'min_seq_length': int, 'max_seq_length': int},],}
            schedule: '1cycle'|'lr_finder'
            num_steps: int
            distributed: bool
        """
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_spec = self.train_fn(features, labels, params)
            return train_spec
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_spec = self.eval_fn(features, labels, params)
            return eval_spec
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predict_spec = self.predict_fn(features, labels, params)
            return predict_spec

    def train_fn(
        self,
        features, # This is batch_features from input_fn
        labels,   # This is batch_labels from input_fn
        params):  # Additional configuration
        """
        lm model function for tf.estimator
        features:
            {0:{'seqs':seqs, 'segs':segs},...}
        params:
            schema: [{
                'field_id': int,
                'group_id': int,
                'item_id': int,
                'type': 'sequence'|'class',
                'limited_vocab': bool,
                'copy_from': [field_ids],
                'token_vocab': Vocab,},]
            run_config: {'batch_size': int, 'max_train_steps': int,
                'max_lr': float, 'pct_start': [0,1], 'dropout': [0,1], 'wd': float,
                'data': [{'target_level': int >= 0, 'max_token_length': int, 'min_seq_length': int, 'max_seq_length': int},],}
            schedule: '1cycle'|'lr_finder'
            num_steps: int
            distributed: bool
        """

        schema = params['schema']
        run_config = params['run_config']
        model_config = self.model_config
        input_embedding, spellin_embedding, field_query_embedding, \
            field_key_embedding, field_value_embedding, field_context_embedding = get_embeddings(
            model_config['char_vocab_size'], model_config['char_vocab_dim'], self.char_vocab.embedding_init,
            model_config['num_layers'], model_config['layer_size'], training=True)
        if run_config.get('dropout') != None:
            embedder_dropout = 0.5*run_config.get('dropout')
        else:
            embedder_dropout = None
        word_embedder = Embedder(
            input_embedding, model_config['layer_size'],
            "Word_Embedder", dropout=embedder_dropout, training=True)
        word_encoder = Encoder(
            model_config['layer_size'], int(model_config['layer_size']/4),
            model_config['layer_size'], model_config['num_layers'], model_config['num_heads'],
            "Word_Encoder", dropout=run_config.get('dropout'), training=True)
        word_matcher = Matcher(
            2*model_config['layer_size'],
            "Word_Matcher", dropout=run_config.get('dropout'), training=True)
        speller_encoder = Encoder(
            int(model_config['layer_size']/2), int(model_config['layer_size']/8),
            int(model_config['layer_size']/2), 2, 4,
            "Speller_Encoder", dropout=run_config.get('dropout'), training=True)
        speller_matcher = Matcher(
            int(model_config['layer_size']/2),
            "Speller_Matcher", dropout=run_config.get('dropout'), training=True)
        speller_cell = SpellerCell("Speller_Cell",
                                   speller_encoder,
                                   dropout=run_config.get('dropout'),
                                   training=True)
        posit_size = int(model_config['layer_size']/4)
        batch_size = tf.shape(features[0]['seqs'])[0]

        for i, piece in features.items():

            seqs = piece['seqs']
            segs = piece['segs']

            # segment
            token_vocab = schema[i].get('token_vocab')
            if not token_vocab is None:
                token_char_ids = tf.constant(token_vocab.decompose_table, tf.int32)
                piece['token_char_ids'] = token_char_ids
            if schema[i]['limited_vocab']:
                segmented_seqs_ref = tf.gather(piece['token_char_ids'], seqs)
            elif schema[i]['type'] == 'class':
                segmented_seqs_ref = tf.expand_dims(seqs, axis=1)
            else:
                segmented_seqs_ref = segment_words(
                    seqs, segs)
            max_token_length = run_config['data'][schema[i]['field_id']]['max_token_length']
            if max_token_length != None:
                segmented_seqs_ref = tf.cond(
                    tf.less(tf.shape(segmented_seqs_ref)[2], max_token_length),
                    lambda: segmented_seqs_ref,
                    lambda: segmented_seqs_ref[:,:,:max_token_length])
            piece['segmented_seqs'] = segmented_seqs_ref

            seq_length = tf.shape(segmented_seqs_ref)[1]
            piece['seq_length'] = seq_length
            if token_vocab is None:
                piece['token_length'] = tf.shape(segmented_seqs_ref)[2]
            else:
                piece['token_length'] = tf.maximum(
                    tf.shape(segmented_seqs_ref)[2], tf.shape(piece['token_char_ids'])[1])

            # embed words
            if not token_vocab is None:
                candidate_word_embeds, _ = word_embedder(
                    tf.expand_dims(piece['token_char_ids'][1:], 0))
                candidate_word_embeds = tf.squeeze(candidate_word_embeds, [0])
                piece['candidate_word_embeds'] = candidate_word_embeds
            if schema[i]['limited_vocab']:
                word_embeds_ref = tf.gather(tf.pad(candidate_word_embeds, [[1,0],[0,0]]), seqs)
                word_masks_ref = tf.greater(seqs, 0)
            else:
                word_embeds_ref, word_masks_ref = word_embedder(
                    segmented_seqs_ref)
            piece['word_embeds'] = word_embeds_ref
            piece['word_masks'] = word_masks_ref

            # field_encodes and posit_embeds
            field_query_embeds = tf.tile(
                tf.nn.embedding_lookup(field_query_embedding, [[schema[i]['field_id']+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            piece['field_query_embeds'] = tuple(tf.split(field_query_embeds, model_config['num_layers'], axis=2))
            field_key_embeds = tf.tile(
                tf.nn.embedding_lookup(field_key_embedding, [[schema[i]['field_id']+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            piece['field_key_embeds'] = tuple(tf.split(field_key_embeds, model_config['num_layers'], axis=2))
            field_value_embeds = tf.tile(
                tf.nn.embedding_lookup(field_value_embedding, [[schema[i]['field_id']+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            piece['field_value_embeds'] = tuple(tf.split(field_value_embeds, model_config['num_layers'], axis=2))
            if schema[i]['type'] == 'sequence':
                posit_ids = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])
                posit_embeds = model_utils_py3.embed_position(
                    posit_ids,
                    posit_size)
            else:
                posit_embeds = tf.zeros([batch_size, seq_length, posit_size])
            piece['posit_embeds'] = posit_embeds

            # picking tokens
            if run_config['data'][schema[i]['field_id']]['target_level'] > 0:
                if schema[i]['type'] == 'sequence':
                    pick_prob = 0.2
                elif schema[i]['type'] == 'class':
                    pick_prob = 0.3
            else:
                if schema[i]['type'] == 'sequence':
                    pick_prob = 0.1
                elif schema[i]['type'] == 'class':
                    pick_prob = 0.1
            pick_masks_ref = tf.less(tf.random_uniform([batch_size, seq_length]), pick_prob)
            pick_masks_ref = tf.logical_and(pick_masks_ref, word_masks_ref)
            piece['pick_masks'] = pick_masks_ref

            # tfstruct
            piece['tfstruct'] = model_utils_py3.TransformerStruct(
                field_query_embeds=piece['field_query_embeds'],
                field_key_embeds=piece['field_key_embeds'],
                field_value_embeds=piece['field_value_embeds'],
                posit_embeds=piece['posit_embeds'],
                token_embeds=piece['word_embeds'],
                masks=piece['word_masks'],
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )
            valid_masks = tf.logical_and(piece['word_masks'], tf.logical_not(piece['pick_masks']))
            piece['masked_tfstruct'] = model_utils_py3.TransformerStruct(
                field_query_embeds=piece['field_query_embeds'],
                field_key_embeds=piece['field_key_embeds'],
                field_value_embeds=piece['field_value_embeds'],
                posit_embeds=piece['posit_embeds'],
                token_embeds=piece['word_embeds'] * tf.cast(tf.expand_dims(valid_masks, axis=2), tf.float32),
                masks=valid_masks,
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )

        # pad segmented_seqs and token_ids to same token_length
        max_token_length = tf.reduce_max(tf.stack([features[i]['token_length'] for i in features], axis=0))
        for i in features:
            piece['token_length'] = max_token_length
            features[i]['segmented_seqs'] = tf.pad(
                features[i]['segmented_seqs'],
                [[0,0],[0,0],[0,max_token_length-tf.shape(features[i]['segmented_seqs'])[2]]])
            if not features[i].get('token_char_ids') is None:
                features[i]['token_char_ids'] = tf.pad(
                    features[i]['token_char_ids'],
                    [[0,0],[0,max_token_length-tf.shape(features[i]['token_char_ids'])[1]]])

        # add extra sample-level tfstruct
        global_tfstruct = model_utils_py3.TransformerStruct(
            field_query_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_query_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                model_config['num_layers'],
                axis=-1)),
            field_key_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_key_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                model_config['num_layers'],
                axis=-1)),
            field_value_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_value_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                model_config['num_layers'],
                axis=-1)),
            posit_embeds=tf.zeros([batch_size, 1, posit_size]),
            token_embeds=tf.zeros([batch_size, 1, model_config['layer_size']]),
            masks=tf.zeros([batch_size, 1], dtype=tf.bool),
            querys=None,
            keys=None,
            values=None,
            encodes=None,
        )

        # prepare attn_matrix
        attn_matrix = []
        attn_matrix.append([1]*(len(features)+1))
        for i in features:
            attn_matrix_local = [0]
            for j in features:
                if i == j:
                    attn_matrix_local.append(1)
                elif schema[i]['group_id'] == schema[j]['group_id'] and schema[i]['item_id'] != schema[j]['item_id']:
                    attn_matrix_local.append(0)
                elif run_config['data'][schema[j]['field_id']]['target_level'] == 0:
                    attn_matrix_local.append(1)
                elif run_config['data'][schema[i]['field_id']]['target_level'] > \
                    run_config['data'][schema[j]['field_id']]['target_level']:
                    attn_matrix_local.append(1)
                else:
                    attn_matrix_local.append(0)
            attn_matrix.append(attn_matrix_local)

        # get encodes
        tfstruct_list = [global_tfstruct] + [features[i]['masked_tfstruct'] for i in features]
        tfstruct_list = encode_tfstructs(word_encoder, tfstruct_list, attn_matrix)
        global_tfstruct = tfstruct_list[0]
        for i in features:
            features[i]['masked_tfstruct'] = tfstruct_list[i+1]

        # get the loss of each piece
        metrics = {}
        for i in features:

            token_char_ids = features[i].get('token_char_ids')

            # picked tokens
            pick_segmented_seqs_ref = tf.boolean_mask(
                features[i]['segmented_seqs'],
                features[i]['pick_masks'])
            _, pick_word_encodes_ref = tf.dynamic_partition(
                features[i]['masked_tfstruct'].encodes, tf.cast(features[i]['pick_masks'], tf.int32), 2)
            num_pick_words = tf.shape(pick_word_encodes_ref)[0]
            item1 = (pick_word_encodes_ref, None, 'encode')

            def true_fn():
                # form candidates
                if schema[i]['limited_vocab']:
                    candidate_segmented_seqs = token_char_ids[1:]
                    candidate_word_embeds = features[i]['candidate_word_embeds']
                    match_idxs = tf.boolean_mask(features[i]['seqs'], features[i]['pick_masks']) - 1 # seqs is 1-based
                    match_matrix = tf.one_hot(match_idxs, tf.shape(candidate_word_embeds)[0])
                else:
                    # all features to select token from
                    select_from_segmented_seqs, select_from_word_embeds, select_from_word_masks = [], [], []
                    for j in features:
                        if j == i or schema[j]['field_id'] in schema[i]['copy_from']:
                            select_from_segmented_seqs.append(features[j]['segmented_seqs'])
                            select_from_word_embeds.append(features[j]['word_embeds'])
                            select_from_word_masks.append(features[j]['word_masks'])
                    select_from_segmented_seqs = tf.concat(select_from_segmented_seqs, axis=1)
                    select_from_word_embeds = tf.concat(select_from_word_embeds, axis=1)
                    select_from_word_masks = tf.concat(select_from_word_masks, axis=1)

                    valid_segmented_seqs_ref = tf.boolean_mask(select_from_segmented_seqs, select_from_word_masks)
                    _, valid_word_embeds_ref = tf.dynamic_partition(
                        select_from_word_embeds, tf.cast(select_from_word_masks, tf.int32), 2)

                    unique_segmented_seqs_ref, unique_idxs = model_utils_py3.unique_vector(valid_segmented_seqs_ref)
                    # use tf.dynamic_partition to replace tf.gather
                    unique_1hots = tf.reduce_sum(
                        tf.one_hot(unique_idxs, tf.shape(valid_word_embeds_ref)[0], dtype=tf.int32),
                        axis=0)
                    _, unique_word_embeds_ref = tf.dynamic_partition(
                        valid_word_embeds_ref,
                        unique_1hots, 2)
                    if token_char_ids is None:
                        candidate_segmented_seqs = unique_segmented_seqs_ref
                        candidate_word_embeds = unique_word_embeds_ref
                    else:
                        extra_cand_ids = token_char_ids[1:]
                        extra_cand_match_matrix = model_utils_py3.match_vector(
                            extra_cand_ids,
                            unique_segmented_seqs_ref)
                        extra_cand_masks = tf.logical_not(tf.reduce_any(extra_cand_match_matrix, axis=-1))
                        candidate_segmented_seqs = tf.concat(
                            [unique_segmented_seqs_ref,
                             tf.boolean_mask(extra_cand_ids, extra_cand_masks)],
                            axis=0)
                        candidate_word_embeds = tf.concat(
                            [unique_word_embeds_ref,
                             tf.boolean_mask(features[i]['candidate_word_embeds'], extra_cand_masks)],
                            axis=0)

                    match_matrix = model_utils_py3.match_vector(
                        pick_segmented_seqs_ref,
                        candidate_segmented_seqs)
                    match_matrix = tf.cast(match_matrix, tf.float32)
                    match_idxs = tf.argmax(match_matrix, axis=-1, output_type=tf.int32)
                num_candidates = tf.shape(candidate_segmented_seqs)[0]
                item2 = (candidate_word_embeds, None, 'embed')

                # retrieve copy encodes and add copy logits
                if len(schema[i]['copy_from']) > 0:
                    copy_from_segmented_seqs, copy_from_encodes, \
                        copy_from_valid_masks, copy_from_pick_masks = [],[],[],[]
                    for j in features:
                        if schema[j]['field_id'] in schema[i]['copy_from']:
                            copy_from_segmented_seqs.append(features[j]['segmented_seqs'])
                            copy_from_encodes.append(features[j]['masked_tfstruct'].encodes)
                            copy_from_valid_masks.append(features[j]['masked_tfstruct'].masks)
                            copy_from_pick_masks.append(features[j]['pick_masks'])
                    # first we get the matrix indicates whether x copy to y
                    copy_from_segmented_seqs = tf.concat(copy_from_segmented_seqs, axis=1)
                    copy_from_encodes = tf.concat(copy_from_encodes, axis=1)
                    copy_from_valid_masks = tf.concat(copy_from_valid_masks, axis=1)
                    copy_from_pick_masks = tf.concat(copy_from_pick_masks, axis=1)
                    copy_from_match_matrix = model_utils_py3.match_vector(
                        copy_from_segmented_seqs, copy_from_segmented_seqs)
                    copy_from_match_matrix = tf.logical_and(copy_from_match_matrix, tf.expand_dims(copy_from_pick_masks, 1))
                    copy_from_match_matrix = tf.logical_and(copy_from_match_matrix, tf.expand_dims(copy_from_valid_masks, 2))
                    # calculate the normalized prob each slot being copied
                    copy_from_scores = tf.cast(copy_from_match_matrix, tf.float32)
                    copy_from_scores /= (tf.reduce_sum(copy_from_scores, axis=1, keepdims=True)+1e-12)
                    copy_from_scores = tf.reduce_sum(copy_from_scores, axis=2)
                    # gather all valid slots which can be selected from
                    valid_segmented_seqs = tf.boolean_mask(copy_from_segmented_seqs, copy_from_valid_masks)
                    valid_scores = tf.boolean_mask(copy_from_scores, copy_from_valid_masks)
                    _, valid_encodes = tf.dynamic_partition(
                        copy_from_encodes, tf.cast(copy_from_valid_masks, tf.int32), 2)
                    valid_encodes = tf.pad(valid_encodes, [[0,1],[0,0]])
                    # for each candidate token, we gather the probs of all corresponding slots
                    valid_match_matrix = model_utils_py3.match_vector(
                        candidate_segmented_seqs,
                        valid_segmented_seqs)
                    valid_match_matrix = tf.cast(valid_match_matrix, tf.float32)
                    valid_match_scores = valid_match_matrix * tf.expand_dims(valid_scores+1e-12, axis=0)
                    # copy / no copy is 1:1
                    valid_pad_score = tf.reduce_sum(valid_match_scores, axis=1, keepdims=True)
                    valid_pad_score = tf.maximum(valid_pad_score, 1e-12)
                    valid_match_scores = tf.concat([valid_match_scores, valid_pad_score], axis=1)
                    sample_ids = tf.squeeze(tf.random.categorical(tf.log(valid_match_scores), 1, dtype=tf.int32), axis=[-1])
                    copy_masks = tf.not_equal(sample_ids, tf.shape(valid_encodes)[0]-1)
                    sample_onehots = tf.one_hot(sample_ids, tf.shape(valid_encodes)[0])
                    candidate_encodes = tf.matmul(sample_onehots, valid_encodes)
                    item2 = [item2, (candidate_encodes, copy_masks, 'encode')]

                # prior logits
                field_context_embeds = tf.nn.embedding_lookup(
                    field_context_embedding, [schema[i]['field_id']+1,])
                context_prior_logits = word_matcher(
                    (pick_word_encodes_ref, None, 'encode'), (field_context_embeds, None, 'latent'))
                word_prior_logits = word_matcher(
                    (field_context_embeds, None, 'latent'), (candidate_word_embeds, None, 'embed'))
                word_select_logits = word_matcher(item1, item2)
                word_select_logits = word_select_logits - context_prior_logits - word_prior_logits

                # word_select_loss
                word_select_logits = tf.reshape(word_select_logits, [num_pick_words*num_candidates])
                labels = tf.reshape(match_matrix, [num_pick_words*num_candidates])
                labels /= tf.reduce_sum(labels)
                word_select_loss_post = tf.nn.softmax_cross_entropy_with_logits(
                    labels=labels, logits=word_select_logits)
                word_select_loss_prior1 = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(context_prior_logits), logits=context_prior_logits)
                word_select_loss_prior1 = tf.reduce_mean(word_select_loss_prior1)
                word_select_loss_prior2 = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(word_prior_logits), logits=word_prior_logits)
                word_select_loss_prior2 = tf.reduce_mean(word_select_loss_prior2)
                word_select_loss = word_select_loss_post \
                    + 0.1*word_select_loss_prior1 \
                    + 0.1*word_select_loss_prior2

                return word_select_loss

            word_select_loss_ref = tf.cond(
                tf.greater(num_pick_words, 0),
                true_fn,
                lambda: tf.zeros([]))

            # word_gen_loss
            if not schema[i]['limited_vocab']:
                word_gen_loss_ref = get_speller_loss(
                    pick_word_encodes_ref,
                    pick_segmented_seqs_ref,
                    spellin_embedding,
                    speller_cell,
                    speller_matcher)
            else:
                word_gen_loss_ref = tf.zeros([])

            features[i]['word_select_loss'] = word_select_loss_ref
            features[i]['word_gen_loss'] = word_gen_loss_ref

        # gather losses
        target_loss_list = []
        regulation_loss_list = []
        for i in features:
            if run_config['data'][schema[i]['field_id']]['target_level'] > 0:
                target_loss_list.append(features[i]['word_select_loss'] + features[i]['word_gen_loss'])
            else:
                regulation_loss_list.append(features[i]['word_select_loss'] + features[i]['word_gen_loss'])
        loss_train = sum(target_loss_list) if len(regulation_loss_list) == 0 else \
            sum(target_loss_list) + sum(regulation_loss_list) / float(len(regulation_loss_list))
        loss_show = sum(target_loss_list)

        # print total num of parameters
        total_params = 0
        for var in tf.global_variables():
#             print(var)
            local_params=1
            shape = var.get_shape()  #getting shape of a variable
            for i in shape:
                local_params *= i.value  #mutiplying dimension values
            total_params += local_params
        tf.logging.info('total number of parameters is: {}'.format(total_params))

        # return EstimatorSpec
        global_step = tf.train.get_global_step()
        optimizer = get_optimizer(
            params['schedule'],
            global_step,
            run_config.get('max_lr'),
            params['num_steps'],
            run_config.get('pct_start'),
            run_config.get('wd'))
        if self.isFreeze:
            var_list = [field_query_embedding, field_key_embedding, field_value_embedding]
        else:
            var_list = None
        train_op = model_utils_py3.optimize_loss(
            loss_train,
            global_step,
            optimizer,
            wd=run_config.get('wd'),
            var_list=var_list,
            scope=None)
        hooks = []
        if params.get('schedule') == 'lr_finder':
            fetches = {'global_step': global_step, 'learning_rate': optimizer._lr_t, 'loss': loss_show}
            hooks.append(LRFinderHook(fetches, params['num_steps']))
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.TRAIN, loss=loss_show, train_op=train_op, training_hooks=hooks)

    def eval_fn(
        self,
        features, # This is batch_features from input_fn
        labels,   # This is batch_labels from input_fn
        params):  # Additional configuration
        """
        lm model function for tf.estimator
        features:
            {0:{'seqs':seqs, 'segs':segs},...}
        params:
            schema: [{
                'field_id': int,
                'group_id': int,
                'item_id': int,
                'type': 'sequence'|'class',
                'limited_vocab': bool,
                'copy_from': [field_ids],
                'token_vocab': Vocab,},]
            run_config: {'batch_size': int, 'max_train_steps': int,
                'max_lr': float, 'pct_start': [0,1], 'dropout': [0,1], 'wd': float,
                'data': [{'target_level': int >= 0, 'max_token_length': int, 'min_seq_length': int, 'max_seq_length': int},],}
            schedule: '1cycle'|'lr_finder'
            num_steps: int
            distributed: bool
        """

        schema = params['schema']
        run_config = params['run_config']
        model_config = self.model_config
        input_embedding, spellin_embedding, field_query_embedding, \
            field_key_embedding, field_value_embedding, field_context_embedding = get_embeddings(
            model_config['char_vocab_size'], model_config['char_vocab_dim'], self.char_vocab.embedding_init,
            model_config['num_layers'], model_config['layer_size'], training=False)
        word_embedder = Embedder(
            input_embedding, model_config['layer_size'],
            "Word_Embedder", dropout=None, training=False)
        word_encoder = Encoder(
            model_config['layer_size'], int(model_config['layer_size']/4),
            model_config['layer_size'], model_config['num_layers'], model_config['num_heads'],
            "Word_Encoder", dropout=None, training=False)
        word_matcher = Matcher(
            2*model_config['layer_size'],
            "Word_Matcher", dropout=None, training=False)
        speller_encoder = Encoder(
            int(model_config['layer_size']/2), int(model_config['layer_size']/8),
            int(model_config['layer_size']/2), 2, 4,
            "Speller_Encoder", dropout=None, training=False)
        speller_matcher = Matcher(
            int(model_config['layer_size']/2),
            "Speller_Matcher", dropout=None, training=False)
        speller_cell = SpellerCell("Speller_Cell",
                                   speller_encoder,
                                   dropout=None,
                                   training=False)
        posit_size = int(model_config['layer_size']/4)
        batch_size = tf.shape(features[0]['seqs'])[0]
        max_target_level = max([item['target_level'] for item in run_config['data']])

        for i, piece in features.items():

            seqs = piece['seqs']
            segs = piece['segs']

            # segment
            token_vocab = schema[i].get('token_vocab')
            if not token_vocab is None:
                token_char_ids = tf.constant(token_vocab.decompose_table, tf.int32)
                piece['token_char_ids'] = token_char_ids
            if schema[i]['limited_vocab']:
                segmented_seqs_ref = tf.gather(piece['token_char_ids'], seqs)
            elif schema[i]['type'] == 'class':
                segmented_seqs_ref = tf.expand_dims(seqs, axis=1)
            else:
                segmented_seqs_ref = segment_words(
                    seqs, segs)
            max_token_length = run_config['data'][schema[i]['field_id']]['max_token_length']
            if max_token_length != None:
                segmented_seqs_ref = tf.cond(
                    tf.less(tf.shape(segmented_seqs_ref)[2], max_token_length),
                    lambda: segmented_seqs_ref,
                    lambda: segmented_seqs_ref[:,:,:max_token_length])
            piece['segmented_seqs'] = segmented_seqs_ref

            seq_length = tf.shape(segmented_seqs_ref)[1]
            piece['seq_length'] = seq_length
            if token_vocab is None:
                piece['token_length'] = tf.shape(segmented_seqs_ref)[2]
            else:
                piece['token_length'] = tf.maximum(tf.shape(segmented_seqs_ref)[2], tf.shape(piece['token_char_ids'])[1])

            # embed words
            if not token_vocab is None:
                candidate_word_embeds, _ = word_embedder(
                    tf.expand_dims(piece['token_char_ids'][1:], 0))
                candidate_word_embeds = tf.squeeze(candidate_word_embeds, [0])
                piece['candidate_word_embeds'] = candidate_word_embeds
            if schema[i]['limited_vocab']:
                word_embeds_ref = tf.gather(tf.pad(candidate_word_embeds, [[1,0],[0,0]]), seqs)
                word_masks_ref = tf.greater(seqs, 0)
            else:
                word_embeds_ref, word_masks_ref = word_embedder(
                    segmented_seqs_ref)
            piece['word_embeds'] = word_embeds_ref
            piece['word_masks'] = word_masks_ref

            # field_encodes and posit_embeds
            field_query_embeds = tf.tile(
                tf.nn.embedding_lookup(field_query_embedding, [[schema[i]['field_id']+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            piece['field_query_embeds'] = tuple(tf.split(field_query_embeds, model_config['num_layers'], axis=2))
            field_key_embeds = tf.tile(
                tf.nn.embedding_lookup(field_key_embedding, [[schema[i]['field_id']+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            piece['field_key_embeds'] = tuple(tf.split(field_key_embeds, model_config['num_layers'], axis=2))
            field_value_embeds = tf.tile(
                tf.nn.embedding_lookup(field_value_embedding, [[schema[i]['field_id']+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            piece['field_value_embeds'] = tuple(tf.split(field_value_embeds, model_config['num_layers'], axis=2))
            if schema[i]['type'] == 'sequence':
                posit_ids = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])
                posit_embeds = model_utils_py3.embed_position(
                    posit_ids,
                    posit_size)
            else:
                posit_embeds = tf.zeros([batch_size, seq_length, posit_size])
            piece['posit_embeds'] = posit_embeds

            # picking tokens
            if schema[i]['type'] == 'sequence':
                pick_prob = 0.2
            elif schema[i]['type'] == 'class':
                pick_prob = 1.0
            pick_masks_ref = tf.less(tf.random_uniform([batch_size, seq_length]), pick_prob)
            pick_masks_ref = tf.logical_and(pick_masks_ref, word_masks_ref)
            piece['pick_masks'] = pick_masks_ref

            # tfstruct
            piece['tfstruct'] = model_utils_py3.TransformerStruct(
                field_query_embeds=piece['field_query_embeds'],
                field_key_embeds=piece['field_key_embeds'],
                field_value_embeds=piece['field_value_embeds'],
                posit_embeds=piece['posit_embeds'],
                token_embeds=piece['word_embeds'],
                masks=piece['word_masks'],
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )
            valid_masks = tf.logical_and(piece['word_masks'], tf.logical_not(piece['pick_masks']))
            piece['masked_tfstruct'] = model_utils_py3.TransformerStruct(
                field_query_embeds=piece['field_query_embeds'],
                field_key_embeds=piece['field_key_embeds'],
                field_value_embeds=piece['field_value_embeds'],
                posit_embeds=piece['posit_embeds'],
                token_embeds=piece['word_embeds'] * tf.cast(tf.expand_dims(valid_masks, axis=2), tf.float32),
                masks=valid_masks,
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )

        # pad segmented_seqs and token_ids to same token_length
        max_token_length = tf.reduce_max(tf.stack([features[i]['token_length'] for i in features], axis=0))
        for i in features:
            piece['token_length'] = max_token_length
            features[i]['segmented_seqs'] = tf.pad(
                features[i]['segmented_seqs'],
                [[0,0],[0,0],[0,max_token_length-tf.shape(features[i]['segmented_seqs'])[2]]])
            if not features[i].get('token_char_ids') is None:
                features[i]['token_char_ids'] = tf.pad(
                    features[i]['token_char_ids'],
                    [[0,0],[0,max_token_length-tf.shape(features[i]['token_char_ids'])[1]]])

        # add extra sample-level tfstruct
        global_tfstruct = model_utils_py3.TransformerStruct(
            field_query_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_query_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                model_config['num_layers'],
                axis=-1)),
            field_key_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_key_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                model_config['num_layers'],
                axis=-1)),
            field_value_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_value_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                model_config['num_layers'],
                axis=-1)),
            posit_embeds=tf.zeros([batch_size, 1, posit_size]),
            token_embeds=tf.zeros([batch_size, 1, model_config['layer_size']]),
            masks=tf.zeros([batch_size, 1], dtype=tf.bool),
            querys=None,
            keys=None,
            values=None,
            encodes=None,
        )

        # loop for target_level
        extra_tfstruct_list = []
        extra_feature_id_list = []
        for tlevel in range(max_target_level+1):

            # gather tfstructs
            tfstruct_list, masked_tfstruct_list, feature_id_list = [], [], []
            for i in features:
                if run_config['data'][schema[i]['field_id']]['target_level'] == tlevel:
                    feature_id_list.append(i)
                    tfstruct_list.append(features[i]['tfstruct'])
                    masked_tfstruct_list.append(features[i]['masked_tfstruct'])

            # prepare attn_matrix
            attn_matrix = []
            for i in feature_id_list:
                attn_matrix_local = []
                for j in feature_id_list:
                    if i == j:
                        attn_matrix_local.append(1)
                    elif schema[i]['group_id'] == schema[j]['group_id'] and schema[i]['item_id'] != schema[j]['item_id']:
                        attn_matrix_local.append(0)
                    elif tlevel == 0:
                        attn_matrix_local.append(1)
                    else:
                        attn_matrix_local.append(0)
                for j in extra_feature_id_list:
                    if schema[i]['group_id'] == schema[j]['group_id'] and schema[i]['item_id'] != schema[j]['item_id']:
                        attn_matrix_local.append(0)
                    else:
                        attn_matrix_local.append(1)
                attn_matrix.append(attn_matrix_local)

            # get encodes
            if len(feature_id_list) > 0:
                masked_tfstruct_list = encode_tfstructs(
                    word_encoder, masked_tfstruct_list, attn_matrix, extra_tfstruct_list)
                for i, feature_id in enumerate(feature_id_list):
                    features[feature_id]['masked_tfstruct'] = masked_tfstruct_list[i]
                if tlevel < max_target_level:
                    tfstruct_list = encode_tfstructs(
                        word_encoder, tfstruct_list, attn_matrix, extra_tfstruct_list)
                    for i, feature_id in enumerate(feature_id_list):
                        features[feature_id]['tfstruct'] = tfstruct_list[i]
                    extra_feature_id_list.extend(feature_id_list)
                    extra_tfstruct_list.extend(tfstruct_list)

        # get the loss of each piece
        metrics = {}
        loss_list = []
        for i in features:

            token_char_ids = features[i].get('token_char_ids')

            # picked tokens
            pick_segmented_seqs_ref = tf.boolean_mask(
                features[i]['segmented_seqs'],
                features[i]['pick_masks'])
            _, pick_word_encodes_ref = tf.dynamic_partition(
                features[i]['masked_tfstruct'].encodes, tf.cast(features[i]['pick_masks'], tf.int32), 2)
            num_pick_words = tf.shape(pick_word_encodes_ref)[0]
            item1 = (pick_word_encodes_ref, None, 'encode')

            def true_fn():
                # form candidates
                if schema[i]['limited_vocab']:
                    candidate_segmented_seqs = token_char_ids[1:]
                    candidate_word_embeds = features[i]['candidate_word_embeds']
                    match_idxs = tf.boolean_mask(features[i]['seqs'], features[i]['pick_masks']) - 1 # seqs is 1-based
                    match_matrix = tf.one_hot(match_idxs, tf.shape(candidate_word_embeds)[0])
                else:
                    # all features to select token from
                    select_from_segmented_seqs, select_from_word_embeds, select_from_word_masks = [], [], []
                    for j in features:
                        if j == i or schema[j]['field_id'] in schema[i]['copy_from']:
                            select_from_segmented_seqs.append(features[j]['segmented_seqs'])
                            select_from_word_embeds.append(features[j]['word_embeds'])
                            select_from_word_masks.append(features[j]['word_masks'])
                    select_from_segmented_seqs = tf.concat(select_from_segmented_seqs, axis=1)
                    select_from_word_embeds = tf.concat(select_from_word_embeds, axis=1)
                    select_from_word_masks = tf.concat(select_from_word_masks, axis=1)

                    valid_segmented_seqs_ref = tf.boolean_mask(select_from_segmented_seqs, select_from_word_masks)
                    _, valid_word_embeds_ref = tf.dynamic_partition(
                        select_from_word_embeds, tf.cast(select_from_word_masks, tf.int32), 2)

                    unique_segmented_seqs_ref, unique_idxs = model_utils_py3.unique_vector(valid_segmented_seqs_ref)
                    # use tf.dynamic_partition to replace tf.gather
                    unique_1hots = tf.reduce_sum(
                        tf.one_hot(unique_idxs, tf.shape(valid_word_embeds_ref)[0], dtype=tf.int32),
                        axis=0)
                    _, unique_word_embeds_ref = tf.dynamic_partition(
                        valid_word_embeds_ref,
                        unique_1hots, 2)
                    if token_char_ids is None:
                        candidate_segmented_seqs = unique_segmented_seqs_ref
                        candidate_word_embeds = unique_word_embeds_ref
                    else:
                        extra_cand_ids = token_char_ids[1:]
                        extra_cand_match_matrix = model_utils_py3.match_vector(
                            extra_cand_ids,
                            unique_segmented_seqs_ref)
                        extra_cand_masks = tf.logical_not(tf.reduce_any(extra_cand_match_matrix, axis=-1))
                        candidate_segmented_seqs = tf.concat(
                            [unique_segmented_seqs_ref,
                             tf.boolean_mask(extra_cand_ids, extra_cand_masks)],
                            axis=0)
                        candidate_word_embeds = tf.concat(
                            [unique_word_embeds_ref,
                             tf.gather(features[i]['candidate_word_embeds'], extra_cand_masks)],
                            axis=0)

                    match_matrix = model_utils_py3.match_vector(
                        pick_segmented_seqs_ref,
                        candidate_segmented_seqs)
                    match_matrix = tf.cast(match_matrix, tf.float32)
                    match_idxs = tf.argmax(match_matrix, axis=-1, output_type=tf.int32)
                num_candidates = tf.shape(candidate_segmented_seqs)[0]
                item2 = (candidate_word_embeds, None, 'embed')

                # retrieve copy encodes and add copy logits
                if len(schema[i]['copy_from']) > 0:
                    copy_from_segmented_seqs, copy_from_encodes, \
                        copy_from_valid_masks, copy_from_pick_masks = [],[],[],[]
                    for j in features:
                        if schema[j]['field_id'] in schema[i]['copy_from']:
                            copy_from_segmented_seqs.append(features[j]['segmented_seqs'])
                            copy_from_encodes.append(features[j]['masked_tfstruct'].encodes)
                            copy_from_valid_masks.append(features[j]['masked_tfstruct'].masks)
                            copy_from_pick_masks.append(features[j]['pick_masks'])
                    # first we get the matrix indicates whether x copy to y
                    copy_from_segmented_seqs = tf.concat(copy_from_segmented_seqs, axis=1)
                    copy_from_encodes = tf.concat(copy_from_encodes, axis=1)
                    copy_from_valid_masks = tf.concat(copy_from_valid_masks, axis=1)
                    copy_from_pick_masks = tf.concat(copy_from_pick_masks, axis=1)
                    copy_from_match_matrix = model_utils_py3.match_vector(
                        copy_from_segmented_seqs, copy_from_segmented_seqs)
                    copy_from_match_matrix = tf.logical_and(copy_from_match_matrix, tf.expand_dims(copy_from_pick_masks, 1))
                    copy_from_match_matrix = tf.logical_and(copy_from_match_matrix, tf.expand_dims(copy_from_valid_masks, 2))
                    # calculate the normalized prob each slot being copied
                    copy_from_scores = tf.cast(copy_from_match_matrix, tf.float32)
                    copy_from_scores /= (tf.reduce_sum(copy_from_scores, axis=1, keepdims=True)+1e-12)
                    copy_from_scores = tf.reduce_sum(copy_from_scores, axis=2)
                    # gather all valid slots which can be selected from
                    valid_segmented_seqs = tf.boolean_mask(copy_from_segmented_seqs, copy_from_valid_masks)
                    valid_scores = tf.boolean_mask(copy_from_scores, copy_from_valid_masks)
                    _, valid_encodes = tf.dynamic_partition(
                        copy_from_encodes, tf.cast(copy_from_valid_masks, tf.int32), 2)
                    valid_encodes = tf.pad(valid_encodes, [[0,1],[0,0]])
                    # for each candidate token, we gather the probs of all corresponding slots
                    valid_match_matrix = model_utils_py3.match_vector(
                        candidate_segmented_seqs,
                        valid_segmented_seqs)
                    valid_match_matrix = tf.cast(valid_match_matrix, tf.float32)
                    valid_match_scores = valid_match_matrix * tf.expand_dims(valid_scores+1e-12, axis=0)
                    # copy / no copy is 1:1
                    valid_pad_score = tf.reduce_sum(valid_match_scores, axis=1, keepdims=True)
                    valid_pad_score = tf.maximum(valid_pad_score, 1e-12)
                    valid_match_scores = tf.concat([valid_match_scores, valid_pad_score], axis=1)
                    sample_ids = tf.squeeze(tf.random.categorical(tf.log(valid_match_scores), 1, dtype=tf.int32), axis=[-1])
                    copy_masks = tf.not_equal(sample_ids, tf.shape(valid_encodes)[0]-1)
                    sample_onehots = tf.one_hot(sample_ids, tf.shape(valid_encodes)[0])
                    candidate_encodes = tf.matmul(sample_onehots, valid_encodes)
                    item2 = [item2, (candidate_encodes, copy_masks, 'encode')]

                # word_select_loss
                word_select_logits = word_matcher(item1, item2)
                word_select_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=match_idxs, logits=word_select_logits)
                word_select_loss = tf.reduce_mean(word_select_loss)

                # predictions
                predictions = tf.argmax(word_select_logits, axis=-1, output_type=tf.int32)
                labels = match_idxs

                return word_select_loss, predictions, labels

            word_select_loss, predictions, labels = tf.cond(
                tf.greater(num_pick_words, 0),
                true_fn,
                lambda: (tf.zeros([]), tf.zeros([0], tf.int32), tf.zeros([0], tf.int32)))

            # eval metrics
            accuracy = tf.metrics.accuracy(
                labels=labels,
                predictions=predictions)
            metrics[str(i)] = accuracy

            if run_config['data'][schema[i]['field_id']]['target_level'] > 0 or max_target_level == 0:
                loss_list.append(word_select_loss)

        # gather losses
        loss = sum(loss_list)

        # print total num of parameters
        total_params = 0
        for var in tf.global_variables():
#             print(var)
            local_params=1
            shape = var.get_shape()  #getting shape of a variable
            for i in shape:
                local_params *= i.value  #mutiplying dimension values
            total_params += local_params
        tf.logging.info('total number of parameters is: {}'.format(total_params))

        # return EstimatorSpec
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=metrics)

    def predict_fn(
        self,
        features, # This is batch_features from input_fn
        labels,   # This is batch_labels from input_fn
        params):  # Additional configuration
        """
        lm model function for tf.estimator
        features:
            {0:{'seqs':seqs, 'segs':segs},...}
        params:
            schema: [{
                'field_id': int,
                'group_id': int,
                'item_id': int,
                'type': 'sequence'|'class',
                'limited_vocab': bool,
                'copy_from': [field_ids],
                'token_vocab': Vocab,},]
            run_config: {'batch_size': int, 'max_train_steps': int,
                'max_lr': float, 'pct_start': [0,1], 'dropout': [0,1], 'wd': float,
                'data': [{'target_level': int >= 0, 'max_token_length': int, 'min_seq_length': int, 'max_seq_length': int},],}
            schedule: '1cycle'|'lr_finder'
            num_steps: int
            distributed: bool
        """

        schema = params['schema']
        run_config = params['run_config']
        model_config = self.model_config
        input_embedding, spellin_embedding, field_query_embedding, \
            field_key_embedding, field_value_embedding, field_context_embedding = get_embeddings(
            model_config['char_vocab_size'], model_config['char_vocab_dim'], self.char_vocab.embedding_init,
            model_config['num_layers'], model_config['layer_size'], training=False)
        word_embedder = Embedder(
            input_embedding, model_config['layer_size'],
            "Word_Embedder", dropout=None, training=False)
        word_encoder = Encoder(
            model_config['layer_size'], int(model_config['layer_size']/4),
            model_config['layer_size'], model_config['num_layers'], model_config['num_heads'],
            "Word_Encoder", dropout=None, training=False)
        word_matcher = Matcher(
            2*model_config['layer_size'],
            "Word_Matcher", dropout=None, training=False)
        word_cell = TransformerCell("Word_Cell",
                                    word_encoder,
                                    dropout=None,
                                    training=False)
        speller_encoder = Encoder(
            int(model_config['layer_size']/2), int(model_config['layer_size']/8),
            int(model_config['layer_size']/2), 2, 4,
            "Speller_Encoder", dropout=None, training=False)
        speller_matcher = Matcher(
            int(model_config['layer_size']/2),
            "Speller_Matcher", dropout=None, training=False)
        speller_cell = SpellerCell("Speller_Cell",
                                   speller_encoder,
                                   dropout=None,
                                   training=False)
        word_generator = WordGenerator(speller_cell, speller_matcher, spellin_embedding,
                                       self.char_vocab.token2id[self.char_vocab.sep])
        sent_generator = SentGenerator(word_cell, word_matcher, word_embedder, word_generator)
        class_generator = ClassGenerator(word_encoder, word_matcher, word_embedder, word_generator)
        posit_size = int(model_config['layer_size']/4)
        batch_size = tf.shape(features[0]['seqs'])[0]
        max_target_level = max([item['target_level'] for item in run_config['data']])

        for i, piece in features.items():

            seqs = piece['seqs']
            segs = piece['segs']

            # segment
            token_vocab = schema[i].get('token_vocab')
            if not token_vocab is None:
                token_char_ids = tf.constant(token_vocab.decompose_table, tf.int32)
                piece['token_char_ids'] = token_char_ids
            if schema[i]['limited_vocab']:
                segmented_seqs_ref = tf.gather(piece['token_char_ids'], seqs)
            elif schema[i]['type'] == 'class':
                segmented_seqs_ref = tf.expand_dims(seqs, axis=1)
            else:
                segmented_seqs_ref = segment_words(
                    seqs, segs)
            max_token_length = run_config['data'][schema[i]['field_id']]['max_token_length']
            if max_token_length != None:
                segmented_seqs_ref = tf.cond(
                    tf.less(tf.shape(segmented_seqs_ref)[2], max_token_length),
                    lambda: segmented_seqs_ref,
                    lambda: segmented_seqs_ref[:,:,:max_token_length])
            piece['segmented_seqs'] = segmented_seqs_ref

            # seq_length and token_length
            if run_config['data'][schema[i]['field_id']]['target_level'] > 0:
                if schema[i]['type'] == 'sequence':
                    seq_length = run_config['data'][schema[i]['field_id']]['max_seq_length']
                elif schema[i]['type'] == 'class':
                    seq_length = 1
                if token_vocab is None:
                    token_length = run_config['data'][schema[i]['field_id']]['max_token_length']
                else:
                    token_length = tf.shape(piece['token_char_ids'])[1]
            else:
                seq_length = tf.shape(segmented_seqs_ref)[1]
                if token_vocab is None:
                    token_length = tf.shape(segmented_seqs_ref)[2]
                else:
                    token_length = tf.maximum(tf.shape(segmented_seqs_ref)[2],
                                                       tf.shape(piece['token_char_ids'])[1])
            piece['seq_length'] = seq_length
            piece['token_length'] = token_length

            # embed words
            if not token_vocab is None:
                candidate_word_embeds, _ = word_embedder(
                    tf.expand_dims(piece['token_char_ids'][1:], 0))
                candidate_word_embeds = tf.squeeze(candidate_word_embeds, [0])
                piece['candidate_word_embeds'] = candidate_word_embeds
            if run_config['data'][schema[i]['field_id']]['target_level'] > 0:
                word_embeds_ref = tf.zeros([batch_size, seq_length, model_config['layer_size']])
                word_masks_ref = tf.zeros([batch_size, seq_length], dtype=tf.bool)
            else:
                if schema[i]['limited_vocab']:
                    word_embeds_ref = tf.gather(tf.pad(candidate_word_embeds, [[1,0],[0,0]]), seqs)
                    word_masks_ref = tf.greater(seqs, 0)
                else:
                    word_embeds_ref, word_masks_ref = word_embedder(
                        segmented_seqs_ref)
            piece['word_embeds'] = word_embeds_ref
            piece['word_masks'] = word_masks_ref

            # field_encodes and posit_embeds
            field_query_embeds = tf.tile(
                tf.nn.embedding_lookup(field_query_embedding, [[schema[i]['field_id']+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            piece['field_query_embeds'] = tuple(tf.split(field_query_embeds, model_config['num_layers'], axis=2))
            field_key_embeds = tf.tile(
                tf.nn.embedding_lookup(field_key_embedding, [[schema[i]['field_id']+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            piece['field_key_embeds'] = tuple(tf.split(field_key_embeds, model_config['num_layers'], axis=2))
            field_value_embeds = tf.tile(
                tf.nn.embedding_lookup(field_value_embedding, [[schema[i]['field_id']+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            piece['field_value_embeds'] = tuple(tf.split(field_value_embeds, model_config['num_layers'], axis=2))
            if schema[i]['type'] == 'sequence':
                posit_ids = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])
                posit_embeds = model_utils_py3.embed_position(
                    posit_ids,
                    posit_size)
            elif schema[i]['type'] == 'class':
                posit_embeds = tf.zeros([batch_size, seq_length, posit_size])
            piece['posit_embeds'] = posit_embeds

            # tfstruct
            piece['tfstruct'] = model_utils_py3.TransformerStruct(
                field_query_embeds=piece['field_query_embeds'],
                field_key_embeds=piece['field_key_embeds'],
                field_value_embeds=piece['field_value_embeds'],
                posit_embeds=piece['posit_embeds'],
                token_embeds=piece['word_embeds'],
                masks=piece['word_masks'],
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )

        # pad segmented_seqs and token_ids to same token_length
        max_token_length = tf.reduce_max(tf.stack([features[i]['token_length'] for i in features], axis=0))
        for i in features:
            piece['token_length'] = max_token_length
            features[i]['segmented_seqs'] = tf.pad(
                features[i]['segmented_seqs'],
                [[0,0],[0,0],[0,max_token_length-tf.shape(features[i]['segmented_seqs'])[2]]])
            if not features[i].get('token_char_ids') is None:
                features[i]['token_char_ids'] = tf.pad(
                    features[i]['token_char_ids'],
                    [[0,0],[0,max_token_length-tf.shape(features[i]['token_char_ids'])[1]]])


        # loop for non targets
        tfstruct_list, feature_id_list = [], []
        for i in features:
            if run_config['data'][schema[i]['field_id']]['target_level'] == 0:
                feature_id_list.append(i)
                tfstruct_list.append(features[i]['tfstruct'])

        # prepare attn_matrix
        attn_matrix = []
        for i in feature_id_list:
            attn_matrix_local = []
            for j in feature_id_list:
                if i == j:
                    attn_matrix_local.append(1)
                elif schema[i]['group_id'] == schema[j]['group_id'] and schema[i]['item_id'] != schema[j]['item_id']:
                    attn_matrix_local.append(0)
                else:
                    attn_matrix_local.append(1)
            attn_matrix.append(attn_matrix_local)

        # get encodes
        if len(feature_id_list) > 0:
            tfstruct_list = encode_tfstructs(
                word_encoder, tfstruct_list, attn_matrix, [])
            for i, feature_id in enumerate(feature_id_list):
                features[feature_id]['tfstruct'] = tfstruct_list[i]

        # null tfstruct
        null_tfstruct = model_utils_py3.init_tfstruct(
            batch_size, word_encoder.embed_size, word_encoder.posit_size,
            word_encoder.layer_size, word_encoder.num_layers)

        # loop for target_level
        predictions = {}
        for tlevel in range(1, max_target_level+1):

            # generate one-by-one
            for i in features:
                if run_config['data'][schema[i]['field_id']]['target_level'] == tlevel:
                    extra_tfstruct_list, extra_feature_id_list = [], []
                    for j in features:
                        if run_config['data'][schema[j]['field_id']]['target_level'] >= tlevel:
                            continue
                        elif schema[i]['group_id'] == schema[j]['group_id'] and \
                            schema[i]['item_id'] != schema[j]['item_id']:
                            continue
                        else:
                            extra_tfstruct_list.append(features[j]['tfstruct'])
                            extra_feature_id_list.append(j)
                    if len(extra_tfstruct_list) > 0:
                        extra_tfstruct = model_utils_py3.concat_tfstructs(extra_tfstruct_list)
                    else:
                        extra_tfstruct = null_tfstruct

                    # generate
                    if schema[i]['type'] == 'sequence':
                        seq_length = run_config['data'][schema[i]['field_id']]['max_seq_length']
                        initial_state = TransformerState(
                            field_query_embedding=tuple(tf.split(
                                tf.tile(
                                    tf.nn.embedding_lookup(field_query_embedding, [schema[i]['field_id']+1,]),
                                    [batch_size, 1]),
                                model_config['num_layers'],
                                axis=-1)),
                            field_key_embedding=tuple(tf.split(
                                tf.tile(
                                    tf.nn.embedding_lookup(field_key_embedding, [schema[i]['field_id']+1,]),
                                    [batch_size, 1]),
                                model_config['num_layers'],
                                axis=-1)),
                            field_value_embedding=tuple(tf.split(
                                tf.tile(
                                    tf.nn.embedding_lookup(field_value_embedding, [schema[i]['field_id']+1,]),
                                    [batch_size, 1]),
                                model_config['num_layers'],
                                axis=-1)),
                            dec_masks=tf.zeros([batch_size, 0], dtype=tf.bool),
                            dec_keys=(
                                tf.zeros([batch_size, 0, model_config['layer_size']]),
                            )*model_config['num_layers'],
                            dec_values=(
                                tf.zeros([batch_size, 0, model_config['layer_size']]),
                            )*model_config['num_layers'],
                            enc_tfstruct=extra_tfstruct,
                        )
                        if schema[i]['limited_vocab']:
                            token_vocab = scheme[i]['token_vocab']
                            sep_id = token_vocab.token2id[token_vocab.sep]
                            word_embedding = features[i]['candidate_word_embeds']
                            word_embedding = tf.concat(
                                [word_embedding[sep_id-1:sep_id], word_embedding[:sep_id-1], word_embedding[sep_id:]],
                                axis=0)
                            word_ids = tf.range(1, tf.shape(word_embedding)[0]+1, dtype=tf.int32)
                            word_ids = tf.concat(
                                [word_ids[sep_id-1:sep_id], word_ids[:sep_id-1], word_ids[sep_id:]],
                                axis=0)
                            seqs, scores = sent_generator.generate(
                                initial_state, seq_length,
                                word_embedding=word_embedding, word_ids=word_ids)
                            features[i]['seqs'] = seqs[:,0]
                            features[i]['segs'] = tf.zeros([batch_size, 0])
                            features[i]['segmented_seqs'] = tf.gather(
                                features[i]['token_char_ids'],features[i]['seqs'])
                        else:
                            candidate_word_embeds = features[i].get('candidate_word_embeds')
                            if candidate_word_embeds is None:
                                max_word_len = run_config['data'][schema[i]['field_id']]['max_token_length']
                                seqs, scores = sent_generator.generate(
                                    initial_state, seq_length,
                                    max_word_len=max_word_len)
                            else:
                                candidate_word_ids = features[i]['token_char_ids'][1:]
                                unk_ids = tf.constant([[self.char_vocab.token2id[self.char_vocab.unk]]], dtype=tf.int32)
                                unk_ids = tf.pad(unk_ids, [[0,0],[0,tf.shape(candidate_word_ids)[1]-1]])
                                unk_masks = tf.reduce_all(tf.equal(candidate_word_ids, unk_ids), axis=1)
                                valid_masks = tf.logical_not(unk_masks)
                                candidate_word_ids = tf.boolean_mask(candidate_word_ids, valid_masks)
                                candidate_word_embeds = tf.boolean_mask(candidate_word_embeds, valid_masks)
                                seqs, scores = sent_generator.generate(
                                    initial_state, seq_length,
                                    word_embedding=candidate_word_embeds,
                                    word_ids=candidate_word_ids)
                            features[i]['segmented_seqs'] = seqs[:,0]
                            features[i]['seqs'], features[i]['segs'] = model_utils_py3.stitch_chars(
                                features[i]['segmented_seqs'])
                            features[i]['segs'] = tf.pad(
                                features[i]['segs'], [[0,0],[1,1]], constant_values=1.0)
                    elif schema[i]['type'] == 'class':
                        tfstruct = features[i]['tfstruct']
                        if schema[i]['limited_vocab']:
                            token_vocab = schema[i]['token_vocab']
                            word_embedding = features[i]['candidate_word_embeds']
                            word_ids = tf.range(1, tf.shape(word_embedding)[0]+1, dtype=tf.int32)
                            classes, scores = class_generator.generate(
                                tfstruct, extra_tfstruct,
                                word_embedding=word_embedding, word_ids=word_ids)
                            features[i]['seqs'] = classes[:,0]
                            features[i]['segs'] = tf.zeros([batch_size, 0])
                            features[i]['segmented_seqs'] = tf.gather(
                                features[i]['token_char_ids'],features[i]['seqs'])
                        else:
                            max_word_len = run_config['data'][schema[i]['field_id']]['max_token_length']
                            classes, scores = class_generator.generate(
                                tfstruct, extra_tfstruct,
                                max_word_len=max_word_len)
                            features[i]['segmented_seqs'] = seqs[:,0]
                            features[i]['seqs'], features[i]['segs'] = model_utils_py3.stitch_chars(
                                features[i]['segmented_seqs'])
                            features[i]['segs'] = tf.pad(
                                features[i]['segs'], [[0,0],[1,1]], constant_values=1.0)

                    # add to predictions
                    predictions[str(i)+'-seqs'] = features[i]['seqs']
                    predictions[str(i)+'-segs'] = features[i]['segs']

                    if tlevel < max_target_level:
                        # embed words
                        if schema[i]['limited_vocab']:
                            word_embeds_ref = tf.gather(
                                tf.pad(features[i]['candidate_word_embeds'], [[1,0],[0,0]]), features[i]['seqs'])
                            word_masks_ref = tf.greater(features[i]['seqs'], 0)
                        else:
                            word_embeds_ref, word_masks_ref = word_embedder(
                                features[i]['segmented_seqs'])
                        piece['word_embeds'] = word_embeds_ref
                        piece['word_masks'] = word_masks_ref

                        # encode
                        tfstruct = model_utils_py3.TransformerStruct(
                            field_query_embeds=features[i]['field_query_embeds'],
                            field_key_embeds=features[i]['field_key_embeds'],
                            field_value_embeds=features[i]['field_value_embeds'],
                            posit_embeds=features[i]['posit_embeds'],
                            token_embeds=features[i]['word_embeds'],
                            masks=features[i]['word_masks'],
                            querys=None,
                            keys=None,
                            values=None,
                            encodes=None,
                        )
                        attn_masks = tfstruct.masks
                        seq_length = tf.shape(attn_masks)[1]
                        if not extra_tfstruct is None:
                            attn_masks = tf.concat([attn_masks, extra_tfstruct.masks], axis=1)
                        attn_masks = tf.tile(tf.expand_dims(attn_masks, axis=1), [1, seq_length, 1])
                        features[i]['tfstruct'] = word_encoder(tfstruct, attn_masks, extra_tfstruct)


        # print total num of parameters
        total_params = 0
        for var in tf.global_variables():
#             print(var)
            local_params=1
            shape = var.get_shape()  #getting shape of a variable
            for i in shape:
                local_params *= i.value  #mutiplying dimension values
            total_params += local_params
        tf.logging.info('total number of parameters is: {}'.format(total_params))

        # return EstimatorSpec
        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT, predictions=predictions)
