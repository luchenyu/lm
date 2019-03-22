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
        window_size = int(0.1*self.num_steps)
        if len(self.losses_smoothed) > window_size and \
            (min(self.losses_smoothed[-window_size:]) > min(self.losses_smoothed) \
            or self.losses_smoothed[-1] >= max(self.losses_smoothed[-2*window_size:-window_size])):
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
            model_config: {'char_vocab_size': int, 'char_vocab_dim': int, 'char_vocab_emb': np.array, 'layer_size': int, 'num_layers': int}
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
            schema: [{'type': 'sequence'|'class', 'token_vocab_file': none|path,
                'token_vocab': Vocab, 'token_char_ids': np.array([num_token, token_len])}, ]
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
        input_embedding, spellin_embedding, spellout_embedding, field_posit_embedding, field_encode_embedding = get_embeddings(
            model_config['char_vocab_size'], model_config['char_vocab_dim'], model_config.get('char_vocab_emb'),
            model_config['layer_size'], training)

        reuse=None
        for i, piece in features.items():

            seqs = piece['seqs']
            segs = piece['segs']

            # segment
            token_char_ids = schema[i].get('token_char_ids')
            if not token_char_ids is None:
                token_char_ids = tf.constant(token_char_ids, tf.int32)
                schema[i]['token_char_ids'] = token_char_ids
                segmented_seqs_ref = tf.gather(token_char_ids, seqs)
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
            max_length = tf.shape(segmented_seqs_ref)[1]
            piece['seq_length'] = max_length

            # embed words
            if token_char_ids is None:
                word_embeds_ref, word_masks_ref = embed_words(
                    segmented_seqs_ref, input_embedding, model_config['layer_size'],
                    run_config.get('dropout'), training, reuse=reuse)
            else:
                candidate_word_embeds, _ = embed_words(
                    tf.expand_dims(token_char_ids[1:], 0), input_embedding, model_config['layer_size'],
                    run_config.get('dropout'), training, reuse=reuse)
                candidate_word_embeds = tf.squeeze(candidate_word_embeds, [0])
                piece['candidate_word_embeds'] = candidate_word_embeds
                word_embeds_ref = tf.gather(tf.pad(candidate_word_embeds, [[1,0],[0,0]]), seqs)
                word_masks_ref = tf.greater(seqs, 0)
            piece['word_embeds'] = word_embeds_ref
            piece['word_masks'] = word_masks_ref

            # field_encodes and posit_embeds
            field_encodes = field_encode_embedding[i+1] + tf.zeros([batch_size, max_length, model_config['layer_size']])
            piece['field_encodes'] = field_encodes
            posit_dim = field_posit_embedding.get_shape()[-1].value
            if schema[i]['type'] == 'sequence':
                posit_ids = tf.tile(tf.expand_dims(tf.range(max_length), 0), [batch_size, 1])
                posit_embeds = model_utils_py3.embed_position(
                    posit_ids,
                    posit_dim)
            else:
                posit_embeds = tf.zeros([batch_size, max_length, posit_dim])
            posit_embeds += field_posit_embedding[i+1]
            piece['posit_embeds'] = posit_embeds

            # picking tokens
            if training:
                if run_config['data'][i]['is_target']:
                    if schema[i]['type'] == 'sequence':
                        pick_prob = 0.2
                    elif schema[i]['type'] == 'class':
                        pick_prob = 0.8
                else:
                    if schema[i]['type'] == 'sequence':
                        pick_prob = 0.1
                    elif schema[i]['type'] == 'class':
                        pick_prob = 0.2
            else:
                if run_config['data'][i]['is_target']:
                    if schema[i]['type'] == 'sequence':
                        pick_prob = 0.2
                    elif schema[i]['type'] == 'class':
                        pick_prob = 1.0
                else:
                    pick_prob = 0.0
            pick_masks_ref = tf.less(tf.random_uniform([batch_size, max_length]), pick_prob)
            pick_masks_ref = tf.logical_and(pick_masks_ref, word_masks_ref)
            piece['pick_masks'] = pick_masks_ref

            reuse=True

        # get the encodes
        word_masks_ref = tf.concat([features[i]['word_masks'] for i in features], axis=1)
        pick_masks_ref = tf.concat([features[i]['pick_masks'] for i in features], axis=1)
        field_encodes_ref = tf.concat([features[i]['field_encodes'] for i in features], axis=1)
        field_encodes_ref = tf.concat(
            [tf.tile(tf.nn.embedding_lookup(field_encode_embedding, [[0,]]), [tf.shape(field_encodes_ref)[0],1,1]),
            field_encodes_ref],
            axis=1)
        posit_embeds_ref = tf.concat([features[i]['posit_embeds'] for i in features], axis=1)
        posit_embeds_ref = tf.concat(
            [tf.tile(tf.nn.embedding_lookup(field_posit_embedding, [[0,]]), [tf.shape(posit_embeds_ref)[0],1,1]),
             posit_embeds_ref],
            axis=1)
        word_embeds_ref = tf.pad(tf.concat([features[i]['word_embeds'] for i in features], axis=1), [[0,0],[1,0],[0,0]])
        masked_word_embeds_ref = word_embeds_ref * \
            tf.cast(tf.expand_dims(
                tf.logical_not(tf.pad(pick_masks_ref, [[0,0],[1,0]])), axis=-1), tf.float32)
        masked_attn_masks = tf.logical_and(word_masks_ref, tf.logical_not(pick_masks_ref))
        masked_attn_masks = tf.pad(masked_attn_masks, [[0,0],[1,0]], constant_values=True)
        masked_encodes_ref = encode_words(
            field_encodes_ref, posit_embeds_ref, masked_word_embeds_ref, masked_attn_masks,
            model_config['num_layers'], run_config.get('dropout'), training)

        # get the loss of each piece
        masked_encodes_ref_list = tf.split(
            masked_encodes_ref, [1,]+[features[i]['seq_length'] for i in features], axis=1)
        for i, piece in features.items():
            piece['encodes'] = masked_encodes_ref_list[i+1]
        metrics = {}
        reuse = None
        for i in features:

            if training or (mode == tf.estimator.ModeKeys.EVAL and run_config['data'][i]['is_target']):

                _, pick_word_encodes_ref = tf.dynamic_partition(
                    features[i]['encodes'], tf.cast(features[i]['pick_masks'], tf.int32), 2)
                token_char_ids = schema[i].get('token_char_ids')

                # word_select_loss
                if not token_char_ids is None:
                    candidate_word_embeds = features[i]['candidate_word_embeds']
                    match_idxs = tf.boolean_mask(features[i]['seqs'], features[i]['pick_masks']) - 1 # seqs is 1-based
                else:
                    valid_segmented_seqs_ref = tf.boolean_mask(features[i]['segmented_seqs'], features[i]['word_masks'])
                    pick_segmented_seqs_ref = tf.boolean_mask(features[i]['segmented_seqs'], features[i]['pick_masks'])
                    _, valid_word_embeds_ref = tf.dynamic_partition(
                        features[i]['word_embeds'], tf.cast(features[i]['word_masks'], tf.int32), 2)
                    num_pick_words = tf.shape(pick_word_encodes_ref)[0]

                    unique_segmented_seqs_ref, unique_idxs = model_utils_py3.unique_2d(valid_segmented_seqs_ref)
                    # use tf.dynamic_partition to replace th.gather
                    unique_1hots = tf.reduce_sum(
                        tf.one_hot(unique_idxs, tf.shape(valid_word_embeds_ref)[0], dtype=tf.int32),
                        axis=0)
                    _, unique_word_embeds_ref = tf.dynamic_partition(
                        valid_word_embeds_ref,
                        unique_1hots, 2)
                    candidate_word_embeds = unique_word_embeds_ref

                    match_matrix = tf.equal(
                        tf.expand_dims(pick_segmented_seqs_ref, 1),
                        tf.expand_dims(unique_segmented_seqs_ref, 0))
                    match_matrix = tf.reduce_all(match_matrix, axis=-1)
                    match_idxs = tf.argmax(tf.cast(match_matrix, tf.int32), axis=-1, output_type=tf.int32)
                word_select_logits_ref = match_embeds(
                    pick_word_encodes_ref, candidate_word_embeds,
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
                if token_char_ids is None:
                    word_gen_loss_ref = train_speller(
                        tf.expand_dims(pick_word_encodes_ref, 1),
                        tf.ones([num_pick_words, 1], dtype=tf.bool),
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

        loss_train = sum(
            [features[i]['word_select_loss'] + features[i]['word_gen_loss'] if run_config['data'][i]['is_target'] else 
             0.1*(features[i]['word_select_loss'] + features[i]['word_gen_loss']) for i in features])
        loss_show = sum(
            [features[i]['word_select_loss'] + features[i]['word_gen_loss'] if run_config['data'][i]['is_target'] else 
             0.0 for i in features])

        # print total num of parameters
        total_params = 0
        for var in tf.global_variables():
            #print(var)
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
                var_list = [field_posit_embedding, field_encode_embedding]
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
