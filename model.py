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
    def __init__(self, model_config, train_dir, pretrain_dir=None):
        """
        args:
            model_config: {'char_vocab_size': int, 'char_vocab_dim': int, 'char_vocab_emb': np.array,
                'layer_size': int, 'num_layers': int, 'num_heads': int}
        """
        self.model_config = model_config
        self.train_dir = train_dir
        self.warm_start_from = None
        self.isFreeze = False
        if pretrain_dir != None:
            self.warm_start_from = tf.estimator.WarmStartSettings(
                        ckpt_to_initialize_from=pretrain_dir,
                        vars_to_warm_start=".*")
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
                'data': [{'is_target': true|false, 'max_token_length': int, 'min_seq_length': int, 'max_seq_length': int},],}
            schedule: '1cycle'|'lr_finder'
            num_steps: int
            distributed: bool
        """

        schema = params['schema']
        run_config = params['run_config']
        model_config = self.model_config
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        input_embedding, spellin_embedding, spellout_embedding, field_embedding = get_embeddings(
            model_config['char_vocab_size'], model_config['char_vocab_dim'], model_config.get('char_vocab_emb'),
            model_config['num_layers'], model_config['layer_size'], training)

        reuse=None
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
                    seqs, segs, reuse=reuse)
            max_token_length = run_config['data'][i]['max_token_length']
            if max_token_length != None:
                segmented_seqs_ref = tf.cond(
                    tf.less(tf.shape(segmented_seqs_ref)[2], max_token_length),
                    lambda: segmented_seqs_ref,
                    lambda: segmented_seqs_ref[:,:,:max_token_length])
            piece['segmented_seqs'] = segmented_seqs_ref

            batch_size = tf.shape(segmented_seqs_ref)[0]
            seq_length = tf.shape(segmented_seqs_ref)[1]
            piece['seq_length'] = seq_length
            if token_vocab is None:
                piece['token_length'] = tf.shape(segmented_seqs_ref)[2]
            else:
                piece['token_length'] = tf.maximum(tf.shape(segmented_seqs_ref)[2], tf.shape(piece['token_char_ids'])[1])

            # embed words
            reuse_local = False
            if not token_vocab is None:
                candidate_word_embeds, _ = embed_words(
                    tf.expand_dims(piece['token_char_ids'][1:], 0), input_embedding, model_config['layer_size'],
                    run_config.get('dropout'), training, reuse=reuse)
                reuse_local = True
                candidate_word_embeds = tf.squeeze(candidate_word_embeds, [0])
                piece['candidate_word_embeds'] = candidate_word_embeds
            if schema[i]['limited_vocab']:
                word_embeds_ref = tf.gather(tf.pad(candidate_word_embeds, [[1,0],[0,0]]), seqs)
                word_masks_ref = tf.greater(seqs, 0)
            else:
                word_embeds_ref, word_masks_ref = embed_words(
                    segmented_seqs_ref, input_embedding, model_config['layer_size'],
                    run_config.get('dropout'), training, reuse=(reuse or reuse_local))
            piece['word_embeds'] = word_embeds_ref
            piece['word_masks'] = word_masks_ref

            # field_encodes and posit_embeds
            field_embeds = tf.tile(
                tf.nn.embedding_lookup(field_embedding, [[schema[i]['field_id']+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1, 1])
            piece['field_embeds'] = field_embeds
            posit_dim = int(model_config['layer_size']/4)
            if schema[i]['type'] == 'sequence':
                posit_ids = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])
                posit_embeds = model_utils_py3.embed_position(
                    posit_ids,
                    posit_dim)
            else:
                posit_embeds = tf.zeros([batch_size, seq_length, posit_dim])
            piece['posit_embeds'] = posit_embeds

            # picking tokens
            if training:
                if run_config['data'][i]['is_target']:
                    if schema[i]['type'] == 'sequence':
                        pick_prob = 0.2
                    elif schema[i]['type'] == 'class':
                        pick_prob = 1.0
                else:
                    if schema[i]['type'] == 'sequence':
                        pick_prob = 0.1
                    elif schema[i]['type'] == 'class':
                        pick_prob = 0.1
            else:
                if run_config['data'][i]['is_target']:
                    if schema[i]['type'] == 'sequence':
                        pick_prob = 0.2
                    elif schema[i]['type'] == 'class':
                        pick_prob = 1.0
                else:
                    pick_prob = 0.0
            pick_masks_ref = tf.less(tf.random_uniform([batch_size, seq_length]), pick_prob)
            pick_masks_ref = tf.logical_and(pick_masks_ref, word_masks_ref)
            piece['pick_masks'] = pick_masks_ref

            reuse=True

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
            
        # prepare attn_masks
        attn_masks = []
        for i in features:
            attn_masks_local = []
            for j in features:
                if schema[i]['group_id'] == schema[j]['group_id'] and schema[i]['item_id'] != schema[j]['item_id']:
                    attn_masks_local.append(
                        tf.zeros([batch_size, features[i]['seq_length'], features[j]['seq_length']], dtype=tf.bool))
                elif (not run_config['data'][i]['is_target']) and run_config['data'][j]['is_target']:
                    attn_masks_local.append(
                        tf.zeros([batch_size, features[i]['seq_length'], features[j]['seq_length']], dtype=tf.bool))
                else:
                    attn_masks_local.append(
                        tf.ones([batch_size, features[i]['seq_length'], features[j]['seq_length']], dtype=tf.bool))
            attn_masks_local = tf.concat(attn_masks_local, axis=-1)
            attn_masks.append(attn_masks_local)
        attn_masks = tf.concat(attn_masks, axis=1)
        word_masks_ref = tf.concat([features[i]['word_masks'] for i in features], axis=1)
        pick_masks_ref = tf.concat([features[i]['pick_masks'] for i in features], axis=1)
        valid_masks = tf.expand_dims(tf.logical_and(word_masks_ref, tf.logical_not(pick_masks_ref)), axis=1)
        attn_masks = tf.logical_and(attn_masks, valid_masks)
        attn_masks = tf.concat([valid_masks, attn_masks], axis=1)
        attn_masks = tf.pad(attn_masks, [[0,0],[0,0],[1,0]], constant_values=True)

        # prepare field and posit embeds
        field_embeds_ref = tf.concat([features[i]['field_embeds'] for i in features], axis=1)
        field_embeds_ref = tf.concat(
            [tf.tile(tf.nn.embedding_lookup(field_embedding, [[0,]]), [tf.shape(field_embeds_ref)[0],1,1,1]),
            field_embeds_ref],
            axis=1)
        posit_embeds_ref = tf.concat([features[i]['posit_embeds'] for i in features], axis=1)
        posit_embeds_ref = tf.pad(posit_embeds_ref, [[0,0],[1,0],[0,0]])

        # mask word_embeds
        word_embeds_ref = tf.pad(tf.concat([features[i]['word_embeds'] for i in features], axis=1), [[0,0],[1,0],[0,0]])
        masked_word_embeds_ref = word_embeds_ref * \
            tf.cast(tf.expand_dims(
                tf.logical_not(tf.pad(pick_masks_ref, [[0,0],[1,0]])), axis=-1), tf.float32)

        # get encodes
        masked_encodes_ref = encode_words(
            field_embeds_ref, posit_embeds_ref, masked_word_embeds_ref, attn_masks,
            model_config['num_layers'], model_config['num_heads'], run_config.get('dropout'), training)

        # get the loss of each piece
        masked_encodes_ref_list = tf.split(
            masked_encodes_ref, [1,]+[features[i]['seq_length'] for i in features], axis=1)
        for i, piece in features.items():
            piece['encodes'] = masked_encodes_ref_list[i+1]
            piece['final_encodes'] = masked_encodes_ref_list[i+1][:,:,-1]
        metrics = {}
        reuse = None
        for i in features:

            if training or (mode == tf.estimator.ModeKeys.EVAL and run_config['data'][i]['is_target']):

                token_char_ids = features[i].get('token_char_ids')

                # picked tokens
                pick_segmented_seqs_ref = tf.boolean_mask(
                    features[i]['segmented_seqs'],
                    features[i]['pick_masks'])
                _, pick_word_encodes_ref = tf.dynamic_partition(
                    features[i]['final_encodes'], tf.cast(features[i]['pick_masks'], tf.int32), 2)
                num_pick_words = tf.shape(pick_word_encodes_ref)[0]
                
                # form candidates
                if schema[i]['limited_vocab']:
                    candidate_segmented_seqs = token_char_ids[1:]
                    candidate_word_embeds = features[i]['candidate_word_embeds']
                    match_idxs = tf.boolean_mask(features[i]['seqs'], features[i]['pick_masks']) - 1 # seqs is 1-based
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
                    match_idxs = tf.argmax(tf.cast(match_matrix, tf.int32), axis=-1, output_type=tf.int32)
                num_candidates = tf.shape(candidate_segmented_seqs)[0]

                # retrieve final encodes of candidates
                if len(schema[i]['copy_from']) > 0:
                    copy_from_segmented_seqs, copy_from_final_encodes, \
                        copy_from_word_masks, copy_from_pick_masks = [],[],[],[]
                    for j in features:
                        if schema[j]['field_id'] in schema[i]['copy_from']:
                            copy_from_segmented_seqs.append(features[j]['segmented_seqs'])
                            copy_from_final_encodes.append(features[j]['final_encodes'])
                            copy_from_word_masks.append(features[j]['word_masks'])
                            copy_from_pick_masks.append(features[j]['pick_masks'])
                    # first we get the matrix indicates whether x copy to y
                    copy_from_segmented_seqs = tf.concat(copy_from_segmented_seqs, axis=1)
                    copy_from_final_encodes = tf.concat(copy_from_final_encodes, axis=1)
                    copy_from_word_masks = tf.concat(copy_from_word_masks, axis=1)
                    copy_from_pick_masks = tf.concat(copy_from_pick_masks, axis=1)
                    copy_from_valid_masks = tf.logical_and(copy_from_word_masks, tf.logical_not(copy_from_pick_masks))
                    copy_from_match_matrix = model_utils_py3.match_vector(copy_from_segmented_seqs, copy_from_segmented_seqs)
                    copy_from_match_matrix = tf.logical_and(copy_from_match_matrix, tf.expand_dims(copy_from_pick_masks, 1))
                    copy_from_match_matrix = tf.logical_and(copy_from_match_matrix, tf.expand_dims(copy_from_valid_masks, 2))
                    # calculate the normalized prob each slot being copied
                    copy_from_scores = tf.cast(copy_from_match_matrix, tf.float32)
                    copy_from_scores /= (tf.reduce_sum(copy_from_scores, axis=1, keepdims=True)+1e-12)
                    copy_from_scores = tf.reduce_sum(copy_from_scores, axis=2)
                    # gather all valid slots which can be selected from
                    valid_segmented_seqs = tf.boolean_mask(copy_from_segmented_seqs, copy_from_valid_masks)
                    valid_scores = tf.boolean_mask(copy_from_scores, copy_from_valid_masks)
                    valid_final_encodes = tf.boolean_mask(copy_from_final_encodes, copy_from_valid_masks)
                    valid_final_encodes = tf.pad(valid_final_encodes, [[1,0],[0,0]])
                    # for each candidate token, we gather the probs of all corresponding slots
                    valid_match_matrix = model_utils_py3.match_vector(
                        candidate_segmented_seqs,
                        valid_segmented_seqs)
                    valid_match_matrix = tf.cast(valid_match_matrix, tf.float32)
                    valid_match_scores = valid_match_matrix * (tf.expand_dims(valid_scores, axis=0)+1e-12)
                    # scale the probs of all slots of one candidate so that high-freq candidate has low prob to be copied
                    scale = 64.0 / tf.cast(num_pick_words, tf.float32)
                    valid_pad_score = tf.reduce_sum(valid_match_scores, axis=1, keepdims=True) * \
                        (tf.maximum(tf.reduce_sum(valid_match_scores, axis=1, keepdims=True) * scale, 1.0)-1.0)+1e-12
                    valid_match_scores = tf.concat([valid_pad_score, valid_match_scores], axis=1)
                    sample_ids = tf.squeeze(tf.random.categorical(tf.log(valid_match_scores), 1, dtype=tf.int32), axis=[-1])
                    candidate_final_encodes = tf.gather(valid_final_encodes, sample_ids)
                else:
                    candidate_final_encodes = tf.zeros_like(candidate_word_embeds)

                # word_select_loss
                word_select_logits_ref = match_embeds(
                    pick_word_encodes_ref, candidate_word_embeds, candidate_final_encodes,
                    run_config.get('dropout'), training, reuse=reuse)
                word_select_loss_ref = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=match_idxs, logits=word_select_logits_ref)
                word_select_loss_ref = tf.cond(
                    tf.greater(tf.size(word_select_loss_ref), 0),
                    lambda: tf.reduce_mean(word_select_loss_ref),
                    lambda: tf.zeros([]))

                # eval metrics
                if mode == tf.estimator.ModeKeys.EVAL and run_config['data'][i]['is_target']:
                    predicted_classes = tf.argmax(word_select_logits_ref, axis=-1, output_type=tf.int32)
                    accuracy = tf.metrics.accuracy(
                        labels=match_idxs,
                        predictions=predicted_classes)
                    metrics[str(i)] = accuracy

                # word_gen_loss
                if not schema[i]['limited_vocab']:
                    word_gen_loss_ref = train_speller(
                        pick_word_encodes_ref,
                        pick_segmented_seqs_ref,
                        spellin_embedding, spellout_embedding,
                        run_config.get('dropout'), training, reuse=reuse)
                else:
                    word_gen_loss_ref = tf.zeros([])

                reuse = True

            else:
                word_select_loss_ref = tf.zeros([])
                word_gen_loss_ref = tf.zeros([])

            features[i]['word_select_loss'] = word_select_loss_ref
            features[i]['word_gen_loss'] = word_gen_loss_ref

        target_loss_list = []
        regulation_loss_list = []
        for i in features:
            if run_config['data'][i]['is_target']:
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
        print('total number of parameters is: {}'.format(total_params))

        # return EstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()
            optimizer = training_schedule(
                params['schedule'], global_step, run_config.get('max_lr'), params['num_steps'], run_config.get('pct_start'))
            if self.isFreeze:
                var_list = [field_embedding]
#                 var_list += tf.trainable_variables(scope='matcher')
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
                mode, loss=loss_show, train_op=train_op, training_hooks=hooks)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss_show, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'features': features,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
