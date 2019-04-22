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
    def __init__(self, model_config, task_config,
                 char_vocab, train_dir,
                 pretrain_dir=None, vars_to_warm_start='.*'):
        """
        args:
            model_config: {
                'char_embed_dim': int,
                'layer_size': int,
                'num_layers': int,
                'num_heads': int,
            }
            task_config: {
                'task_spec': [
                    {
                        'type': 'sequence'|'class',
                        'copy_from': [field_ids],
                        'target_level': int >= 0,
                        'group_id': int,
                    },
                ],
            }
            char_vocab: character vocab
            train_dir: dir for checkpoints
            pretrain_dir: dir to warmstart from
            vars_to_warm_start: pass to ws object
        """
        self.model_config = model_config
        self.task_config = task_config
        self.char_vocab = char_vocab
        self.train_dir = train_dir
        self.warm_start_from = None
        self.isFreeze = False
        if pretrain_dir != None:
            self.warm_start_from = tf.estimator.WarmStartSettings(
                        ckpt_to_initialize_from=pretrain_dir,
                        vars_to_warm_start=vars_to_warm_start)
            self.isFreeze = True
            # check model_config
            if os.path.exists(os.path.join(pretrain_dir, 'model_config')):
                with open(os.path.join(pretrain_dir, 'model_config'), 'r') as json_file:
                    model_config = json.load(json_file)
                    assert(self.model_config == model_config)

        os.makedirs(train_dir, exist_ok=True)
        # check model_config
        if os.path.exists(os.path.join(train_dir, 'model_config')):
            with open(os.path.join(train_dir, 'model_config'), 'r') as json_file:
                model_config = json.load(json_file)
                assert(self.model_config == model_config)
        else:
            with open(os.path.join(train_dir, 'model_config'), 'w') as json_file:
                json.dump(self.model_config, json_file, indent=4)
        # check task_config
        if os.path.exists(os.path.join(train_dir, 'task_config')):
            with open(os.path.join(train_dir, 'task_config'), 'r') as json_file:
                task_config = json.load(json_file)
                assert(self.task_config == task_config)
        else:
            with open(os.path.join(train_dir, 'task_config'), 'w') as json_file:
                json.dump(self.task_config, json_file, indent=4)

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

    def model_fn(
        self,
        features, # This is batch_features from input_fn
        labels,   # This is batch_labels from input_fn
        mode,     # An instance of tf.estimator.ModeKeys
        params):  # Additional configuration
        """
        lm model function for tf.estimator
        features:
            {0:{'seqs':seqs, 'segs':segs},...}
        params: {
            'data_index': [ ## list by feature_id
                {
                    'field_id': int,
                    'item_id': int,
                },
            ],
            'data_schema': [ ## list by field_id
                {
                    'type': 'sequence'|'class',
                    'limited_vocab': bool,
                    'token_vocab': MoleculeVocab,
                    'max_token_length': int,
                    'min_seq_length': int,
                    'max_seq_length': int,
                    'group_id': int,
                },
            ],
            'hyper_params': {
                'batch_size': int,
                'max_train_steps': int,
                'max_lr': float,
                'pct_start': 0~1,
                'dropout': 0~1,
                'wd': float,
            },
            schedule: '1cycle'|'lr_finder'
            num_steps: int
            distributed: bool
        }
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
        params: {
            'data_index': [ ## list by feature_id
                {
                    'field_id': int,
                    'item_id': int,
                },
            ],
            'data_schema': [ ## list by field_id
                {
                    'type': 'sequence'|'class',
                    'limited_vocab': bool,
                    'token_vocab': MoleculeVocab,
                    'max_token_length': int,
                    'min_seq_length': int,
                    'max_seq_length': int,
                    'group_id': int,
                },
            ],
            'hyper_params': {
                'batch_size': int,
                'max_train_steps': int,
                'max_lr': float,
                'pct_start': 0~1,
                'dropout': 0~1,
                'wd': float,
            },
            schedule: '1cycle'|'lr_finder'
            num_steps: int
            distributed: bool
        }
        """

        data_index = params['data_index']
        data_schema = params['data_schema']
        hyper_params = params['hyper_params']

        task_config = self.task_config
        task_spec = task_config['task_spec']

        model_config = self.model_config
        layer_size = model_config['layer_size']
        posit_size = int(layer_size/4)
        embed_size = layer_size
        num_layers = model_config['num_layers']
        num_heads = model_config['num_heads']
        word_match_size = 2*layer_size
        global_match_size = layer_size
        speller_layer_size = int(layer_size/2)
        speller_posit_size = int(speller_layer_size/4)
        speller_embed_size = speller_layer_size
        speller_num_layers = 2
        speller_num_heads = 4
        speller_match_size = speller_layer_size

        dropout = hyper_params.get('dropout')
        if not dropout is None:
            embedder_dropout = 0.5*dropout
        else:
            embedder_dropout = None

        input_embedding, spellin_embedding, \
        field_query_embedding, \
        field_key_embedding, \
        field_value_embedding, \
        field_context_embedding, \
        field_word_embedding, \
        speller_context_embedding = get_embeddings(
            len(self.char_vocab), model_config['char_embed_dim'],
            self.char_vocab.embedding_init, num_layers, layer_size,
            word_match_size, speller_embed_size, speller_match_size, training=True)

        word_embedder = Embedder(
            input_embedding, embed_size,
            "Word_Embedder",
            dropout=embedder_dropout, training=True)
        word_encoder = Encoder(
            embed_size, posit_size, layer_size, num_layers, num_heads,
            "Word_Encoder",
            dropout=dropout, training=True)
        word_matcher = Matcher(
            word_match_size,
            "Word_Matcher",
            dropout=dropout, training=True)
        global_matcher = Matcher(
            global_match_size,"Global_Matcher",
            dropout=dropout, training=True)
        word_cell = TransformerCell(
            "Word_Cell",
            word_encoder,
            dropout=dropout,
            training=True)
        speller_encoder = Encoder(
            speller_embed_size, speller_posit_size,
            speller_layer_size, speller_num_layers, speller_num_heads,
            "Speller_Encoder",
            dropout=dropout, training=True)
        speller_matcher = Matcher(
            speller_match_size,
            "Speller_Matcher",
            dropout=dropout, training=True)
        speller_cell = SpellerCell(
            "Speller_Cell",
            speller_encoder,
            dropout=dropout,
            training=True)
        speller_trainer = SpellerTrainer(
            "Speller_Trainer",
            spellin_embedding,
            speller_context_embedding,
            speller_cell,
            speller_matcher,
            training=True)
        word_trainer = WordTrainer(
            "Word_Trainer",
            word_matcher,
            global_matcher,
            speller_trainer,
            training=True)
        sent_trainer = SentTrainer(
            "Sent_Trainer",
            word_cell,
            word_trainer,
            training=True)

        batch_size = tf.shape(features[0]['seqs'])[0]

        for i, feature in features.items():

            field_id = data_index[i]['field_id']
            feature_type = data_schema[field_id]['type']
            limited_vocab = data_schema[field_id]['limited_vocab']
            token_vocab = data_schema[field_id].get('token_vocab')
            max_token_length = data_schema[field_id].get('max_token_length')
            target_level = task_spec[field_id]['target_level']
            seqs = feature['seqs']
            segs = feature['segs']

            # segment
            if not token_vocab is None:
                token_ids = data_schema[field_id].get('token_ids')
                candidate_ids = data_schema[field_id].get('candidate_ids')
                if token_ids is None:
                    token_ids = tf.constant(token_vocab.decompose_table, tf.int32)
                    if not max_token_length is None:
                        token_ids = tf.cond(
                            tf.less(tf.shape(token_ids)[1], max_token_length),
                            lambda: token_ids,
                            lambda: token_ids[:,:max_token_length])
                    candidate_ids = token_ids[1:]
                    data_schema[field_id]['token_ids'] = token_ids
                    data_schema[field_id]['candidate_ids'] = candidate_ids
            if limited_vocab:
                segmented_seqs = tf.gather(token_ids, seqs)
            else:
                if feature_type == 'class':
                    segmented_seqs = tf.expand_dims(seqs, axis=1)
                elif feature_type == 'sequence':
                    segmented_seqs = segment_words(seqs, segs)
                else:
                    segmented_seqs = None
                if not max_token_length is None:
                    segmented_seqs = tf.cond(
                        tf.less(tf.shape(segmented_seqs)[2], max_token_length),
                        lambda: segmented_seqs,
                        lambda: segmented_seqs[:,:,:max_token_length])
            feature['segmented_seqs'] = segmented_seqs

            seq_length = tf.shape(segmented_seqs)[1]
            feature['seq_length'] = seq_length
            if token_vocab is None:
                feature['token_length'] = tf.shape(segmented_seqs)[2]
            else:
                feature['token_length'] = tf.maximum(
                    tf.shape(segmented_seqs)[2], tf.shape(token_ids)[1])

            # embed words
            if not token_vocab is None:
                token_embeds = data_schema[field_id].get('token_embeds')
                candidate_embeds = data_schema[field_id].get('candidate_embeds')
                if token_embeds is None:
                    candidate_embeds, _ = word_embedder(
                        tf.expand_dims(candidate_ids, 0))
                    candidate_embeds = tf.squeeze(candidate_embeds, [0])
                    token_embeds = tf.pad(candidate_embeds, [[1,0],[0,0]])
                    data_schema[field_id]['token_embeds'] = token_embeds
                    data_schema[field_id]['candidate_embeds'] = candidate_embeds
            if limited_vocab:
                onehots = tf.one_hot(
                    tf.reshape(seqs, [batch_size*seq_length]),
                    tf.shape(token_embeds)[0])
                word_embeds = tf.matmul(onehots, token_embeds)
                word_embeds = tf.reshape(
                    word_embeds, [batch_size, seq_length, embed_size])
                word_masks = tf.greater(seqs, 0)
            else:
                word_embeds, word_masks = word_embedder(
                    segmented_seqs)
            feature['word_embeds'] = word_embeds
            feature['word_masks'] = word_masks

            # field_encodes and posit_embeds
            field_query_embeds, field_key_embeds, field_value_embeds = \
                data_schema[field_id].get('field_query_embedding'), \
                data_schema[field_id].get('field_key_embedding'), \
                data_schema[field_id].get('field_value_embedding')
            if field_query_embeds is None or field_key_embeds is None or field_value_embeds is None:
                field_query_embeds = tf.tile(
                    tf.nn.embedding_lookup(field_query_embedding, [field_id+1,]),# field embeds 0 is reserved
                    [batch_size, 1])
                field_query_embeds = tuple(tf.split(field_query_embeds, num_layers, axis=1))
                field_key_embeds = tf.tile(
                    tf.nn.embedding_lookup(field_key_embedding, [field_id+1,]),# field embeds 0 is reserved
                    [batch_size, 1])
                field_key_embeds = tuple(tf.split(field_key_embeds, num_layers, axis=1))
                field_value_embeds = tf.tile(
                    tf.nn.embedding_lookup(field_value_embedding, [field_id+1,]),# field embeds 0 is reserved
                    [batch_size, 1])
                field_value_embeds = tuple(tf.split(field_value_embeds, num_layers, axis=1))
                data_schema[field_id]['field_query_embedding'] = field_query_embeds
                data_schema[field_id]['field_key_embedding'] = field_key_embeds
                data_schema[field_id]['field_value_embedding'] = field_value_embeds
            feature['field_query_embeds'] = tuple(
                [tf.tile(tf.expand_dims(e, axis=1), [1,seq_length,1]) for e in field_query_embeds])
            feature['field_key_embeds'] = tuple(
                [tf.tile(tf.expand_dims(e, axis=1), [1,seq_length,1]) for e in field_key_embeds])
            feature['field_value_embeds'] = tuple(
                [tf.tile(tf.expand_dims(e, axis=1), [1,seq_length,1]) for e in field_value_embeds])
            if feature_type == 'sequence':
                posit_ids = tf.tile(
                    tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])
                posit_embeds = model_utils_py3.embed_position(
                    posit_ids,
                    posit_size)
            elif feature_type == 'class':
                posit_embeds = tf.zeros([batch_size, seq_length, posit_size])
            else:
                posit_embeds = None
            feature['posit_embeds'] = posit_embeds

            # picking tokens
            if feature_type == 'sequence':
                pick_prob = 0.2
            elif feature_type == 'class':
                pick_prob = 0.1
            else:
                pick_prob = None
            pick_masks = tf.less(tf.random_uniform([batch_size, seq_length]), pick_prob)
            pick_masks = tf.logical_and(pick_masks, word_masks)
            feature['pick_masks'] = pick_masks

            # tfstruct
            feature['tfstruct'] = model_utils_py3.TransformerStruct(
                field_query_embeds=feature['field_query_embeds'],
                field_key_embeds=feature['field_key_embeds'],
                field_value_embeds=feature['field_value_embeds'],
                posit_embeds=feature['posit_embeds'],
                token_embeds=feature['word_embeds'],
                masks=feature['word_masks'],
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )
            valid_masks = tf.logical_and(
                feature['word_masks'], tf.logical_not(feature['pick_masks']))
            feature['masked_tfstruct'] = model_utils_py3.TransformerStruct(
                field_query_embeds=feature['field_query_embeds'],
                field_key_embeds=feature['field_key_embeds'],
                field_value_embeds=feature['field_value_embeds'],
                posit_embeds=feature['posit_embeds'],
                token_embeds=feature['word_embeds'] * \
                    tf.expand_dims(tf.cast(valid_masks, tf.float32), axis=2),
                masks=valid_masks,
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )

        # add extra sample-level tfstruct
        global_tfstruct = model_utils_py3.TransformerStruct(
            field_query_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_query_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                num_layers,
                axis=-1)),
            field_key_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_key_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                num_layers,
                axis=-1)),
            field_value_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_value_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                num_layers,
                axis=-1)),
            posit_embeds=tf.zeros([batch_size, 1, posit_size]),
            token_embeds=tf.zeros([batch_size, 1, embed_size]),
            masks=tf.zeros([batch_size, 1], dtype=tf.bool),
            querys=None,
            keys=None,
            values=None,
            encodes=None,
        )

        # prepare attn_matrix
        attn_matrix = []
        attn_matrix_macro = [0]
        for i in features:
            field_id_i = data_index[i]['field_id']
            item_id_i = data_index[i]['item_id']
            group_id_i = data_schema[field_id_i]['group_id']
            target_level_i = task_spec[field_id_i]['target_level']
            if target_level_i == 0:
                attn_matrix_macro.append(1)
            else:
                attn_matrix_macro.append(0)
            attn_matrix_local = [0]
            for j in features:
                field_id_j = data_index[j]['field_id']
                item_id_j = data_index[j]['item_id']
                group_id_j = data_schema[field_id_j]['group_id']
                target_level_j = task_spec[field_id_j]['target_level']
                if i == j:
                    attn_matrix_local.append(1)
                elif group_id_i == group_id_j and item_id_i != item_id_j:
                    attn_matrix_local.append(0)
                elif target_level_j == 0:
                    attn_matrix_local.append(1)
                elif target_level_i > target_level_j:
                    attn_matrix_local.append(1)
                else:
                    attn_matrix_local.append(0)
            attn_matrix.append(attn_matrix_local)
        attn_matrix = [attn_matrix_macro] + attn_matrix

        # get encodes
        tfstruct_list = [global_tfstruct] + [features[i]['masked_tfstruct'] for i in features]
        tfstruct_list = encode_tfstructs(word_encoder, tfstruct_list, attn_matrix)
        global_tfstruct = tfstruct_list[0]
        global_encodes = tf.squeeze(global_tfstruct.encodes, [1])
        for i in features:
            features[i]['masked_tfstruct'] = tfstruct_list[i+1]

        # get the loss of each feature
        regulation_losses = {}
        target_losses = {}
        for i, feature in features.items():

            field_id = data_index[i]['field_id']
            item_id = data_index[i]['item_id']
            group_id = data_schema[field_id]['group_id']
            feature_type = data_schema[field_id]['type']
            limited_vocab = data_schema[field_id]['limited_vocab']
            token_vocab = data_schema[field_id].get('token_vocab')
            token_ids = data_schema[field_id].get('token_ids')
            token_embeds = data_schema[field_id].get('token_embeds')
            candidate_ids = data_schema[field_id].get('candidate_ids')
            candidate_embeds = data_schema[field_id].get('candidate_embeds')
            target_seqs = None if candidate_ids is None else feature['seqs']-1
            max_token_length = data_schema[field_id].get('max_token_length')
            target_level = task_spec[field_id]['target_level']
            copy_from = task_spec[field_id].get('copy_from')
            copy_from = [] if copy_from is None else copy_from

            # prepare resources
            field_query_embeds = data_schema[field_id]['field_query_embedding']
            field_key_embeds = data_schema[field_id]['field_key_embedding']
            field_value_embeds = data_schema[field_id]['field_value_embedding']
            field_context_embeds = tf.nn.embedding_lookup(
                field_context_embedding, [field_id+1,])
            field_word_embeds = tf.nn.embedding_lookup(
                field_word_embedding, [field_id+1,])
            copy_word_ids, copy_encodes, copy_masks = [],[],[]
            for j in features:
                if data_index[j]['field_id'] in copy_from and i != j:
                    copy_word_ids.append(features[j]['segmented_seqs'])
                    copy_encodes.append(features[j]['masked_tfstruct'].encodes)
                    copy_masks.append(features[j]['masked_tfstruct'].masks)
            if len(copy_word_ids) > 0:
                copy_word_ids = model_utils_py3.pad_vectors(
                    copy_word_ids)
                copy_word_ids = tf.concat(copy_word_ids, axis=1)
                copy_encodes = tf.concat(copy_encodes, axis=1)
                copy_masks = tf.concat(copy_masks, axis=1)
            else:
                copy_word_ids, copy_encodes, copy_masks = None, None, None
            if feature_type == 'sequence':
                if not copy_word_ids is None:
                    self_copy_word_ids = tf.concat(
                        model_utils_py3.pad_vectors([feature['segmented_seqs'], copy_word_ids]),
                        axis=1)
                    self_copy_encodes = tf.concat(
                        [feature['masked_tfstruct'].encodes, copy_encodes],
                        axis=1)
                    self_copy_masks = tf.concat(
                        [feature['masked_tfstruct'].masks, copy_masks],
                        axis=1)
                else:
                    self_copy_word_ids = feature['segmented_seqs']
                    self_copy_encodes = feature['masked_tfstruct'].encodes
                    self_copy_masks = feature['masked_tfstruct'].masks
            elif feature_type == 'class':
                self_copy_word_ids = copy_word_ids
                self_copy_encodes = copy_encodes
                self_copy_masks = copy_masks
            else:
                self_copy_word_ids = None
                self_copy_encodes = None
                self_copy_masks = None

            # regulation loss
            regulation_loss = word_trainer(
                limited_vocab,
                field_context_embeds, field_word_embeds,
                feature['segmented_seqs'], feature['word_embeds'],
                feature['masked_tfstruct'].encodes,
                feature['word_masks'], feature['pick_masks'],
                global_encodes,
                candidate_ids, candidate_embeds, target_seqs,
                self_copy_word_ids, self_copy_encodes, self_copy_masks,
            )
            if regulation_losses.get(field_id) is None:
                regulation_losses[field_id] = []
            regulation_losses[field_id].append(regulation_loss)

            # target loss
            if target_level > 0:
                extra_tfstruct_list, extra_feature_id_list = [], []
                for j in features:
                    field_id_j = data_index[j]['field_id']
                    item_id_j = data_index[j]['item_id']
                    group_id_j = data_schema[field_id_j]['group_id']
                    target_level_j = task_spec[field_id_j]['target_level']
                    if target_level_j >= target_level:
                        continue
                    elif group_id == group_id_j and item_id != item_id_j:
                        continue
                    else:
                        extra_tfstruct_list.append(features[j]['masked_tfstruct'])
                        extra_feature_id_list.append(j)
                if len(extra_tfstruct_list) > 0:
                    extra_tfstruct = model_utils_py3.concat_tfstructs(extra_tfstruct_list)
                else:
                    extra_tfstruct = None
                if feature_type == 'sequence':
                    initial_state = TransformerState(
                        field_query_embedding=field_query_embeds,
                        field_key_embedding=field_key_embeds,
                        field_value_embedding=field_value_embeds,
                        dec_masks=tf.zeros([batch_size, 0], dtype=tf.bool),
                        dec_keys=(
                            tf.zeros([batch_size, 0, layer_size]),
                        )*num_layers,
                        dec_values=(
                            tf.zeros([batch_size, 0, layer_size]),
                        )*num_layers,
                        enc_tfstruct=extra_tfstruct,
                    )
                    target_loss = sent_trainer(
                        initial_state,
                        limited_vocab,
                        field_context_embeds, field_word_embeds,
                        feature['segmented_seqs'], feature['word_embeds'],
                        feature['word_masks'],
                        global_encodes,
                        candidate_ids, candidate_embeds, target_seqs,
                        copy_word_ids, copy_encodes, copy_masks,
                    )
                elif feature_type == 'class':
                    tfstruct = model_utils_py3.TransformerStruct(
                        field_query_embeds=feature['field_query_embeds'],
                        field_key_embeds=feature['field_key_embeds'],
                        field_value_embeds=feature['field_value_embeds'],
                        posit_embeds=feature['posit_embeds'],
                        token_embeds=tf.zeros([batch_size, 1, embed_size]),
                        masks=tf.zeros([batch_size, 1], dtype=tf.bool),
                        querys=None,
                        keys=None,
                        values=None,
                        encodes=None,
                    )
                    tfstruct = encode_tfstructs(
                        word_encoder, [tfstruct], [[1,1]], extra_tfstruct_list)[0]
                    target_loss = word_trainer(
                        limited_vocab,
                        field_context_embeds, field_word_embeds,
                        feature['segmented_seqs'], feature['word_embeds'],
                        tfstruct.encodes,
                        feature['word_masks'], tf.ones([batch_size, 1], dtype=tf.bool),
                        global_encodes,
                        candidate_ids, candidate_embeds, target_seqs,
                        copy_word_ids, copy_encodes, copy_masks,
                    )
                else:
                    target_loss = None
                if target_losses.get(field_id) is None:
                    target_losses[field_id] = []
                target_losses[field_id].append(target_loss)

        # gather losses
        loss_regulation = tf.reduce_mean(
            tf.stack(
                [tf.reduce_mean(
                    tf.stack(regulation_losses[key], axis=0)) for key in regulation_losses],
                axis=0))
        if len(target_losses) == 0:
            loss_train = loss_regulation
            loss_show = loss_regulation
        else:
            loss_target = tf.reduce_sum(
                tf.stack(
                    [tf.reduce_mean(
                        tf.stack(target_losses[key], axis=0)) for key in target_losses],
                    axis=0))
            loss_train = loss_regulation + loss_target
            loss_show = loss_target

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
            hyper_params.get('max_lr'),
            params['num_steps'],
            hyper_params.get('pct_start'),
            hyper_params.get('wd'))
        if self.isFreeze:
            var_list = [field_query_embedding, field_key_embedding, field_value_embedding]
        else:
            var_list = None
        train_op = model_utils_py3.optimize_loss(
            loss_train,
            global_step,
            optimizer,
            wd=hyper_params.get('wd'),
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
        params: {
            'data_index': [ ## list by feature_id
                {
                    'field_id': int,
                    'item_id': int,
                },
            ],
            'data_schema': [ ## list by field_id
                {
                    'type': 'sequence'|'class',
                    'limited_vocab': bool,
                    'token_vocab': MoleculeVocab,
                    'max_token_length': int,
                    'min_seq_length': int,
                    'max_seq_length': int,
                    'group_id': int,
                },
            ],
            'hyper_params': {
                'batch_size': int,
                'max_train_steps': int,
                'max_lr': float,
                'pct_start': 0~1,
                'dropout': 0~1,
                'wd': float,
            },
            schedule: '1cycle'|'lr_finder'
            num_steps: int
            distributed: bool
        """

        data_index = params['data_index']
        data_schema = params['data_schema']
        hyper_params = params['hyper_params']

        task_config = self.task_config
        task_spec = task_config['task_spec']

        model_config = self.model_config
        layer_size = model_config['layer_size']
        posit_size = int(layer_size/4)
        embed_size = layer_size
        num_layers = model_config['num_layers']
        num_heads = model_config['num_heads']
        word_match_size = 2*layer_size
        global_match_size = layer_size
        speller_layer_size = int(layer_size/2)
        speller_posit_size = int(speller_layer_size/4)
        speller_embed_size = speller_layer_size
        speller_num_layers = 2
        speller_num_heads = 4
        speller_match_size = speller_layer_size

        input_embedding, spellin_embedding, \
        field_query_embedding, \
        field_key_embedding, \
        field_value_embedding, \
        field_context_embedding, \
        field_word_embedding, \
        speller_context_embedding = get_embeddings(
            len(self.char_vocab), model_config['char_embed_dim'],
            self.char_vocab.embedding_init, num_layers, layer_size,
            word_match_size, speller_embed_size, speller_match_size, training=False)

        word_embedder = Embedder(
            input_embedding, embed_size,
            "Word_Embedder", dropout=None, training=False)
        word_encoder = Encoder(
            embed_size, posit_size, layer_size, num_layers, num_heads,
            "Word_Encoder", dropout=None, training=False)
        word_matcher = Matcher(
            word_match_size,
            "Word_Matcher", dropout=None, training=False)
        global_matcher = Matcher(
            global_match_size,"Global_Matcher",
            dropout=None, training=False)
        speller_encoder = Encoder(
            speller_embed_size, speller_posit_size,
            speller_layer_size, speller_num_layers, speller_num_heads,
            "Speller_Encoder", dropout=None, training=False)
        speller_matcher = Matcher(
            speller_match_size,
            "Speller_Matcher", dropout=None, training=False)
        speller_cell = SpellerCell(
            "Speller_Cell",
            speller_encoder,
            dropout=None,
            training=False)

        batch_size = tf.shape(features[0]['seqs'])[0]
        max_target_level = max([item['target_level'] for item in task_spec])

        for i, feature in features.items():

            field_id = data_index[i]['field_id']
            feature_type = data_schema[field_id]['type']
            limited_vocab = data_schema[field_id]['limited_vocab']
            token_vocab = data_schema[field_id].get('token_vocab')
            max_token_length = data_schema[field_id].get('max_token_length')
            target_level = task_spec[field_id]['target_level']
            seqs = feature['seqs']
            segs = feature['segs']

            # segment
            if not token_vocab is None:
                token_ids = data_schema[field_id].get('token_ids')
                candidate_ids = data_schema[field_id].get('candidate_ids')
                if token_ids is None:
                    token_ids = tf.constant(token_vocab.decompose_table, tf.int32)
                    if not max_token_length is None:
                        token_ids = tf.cond(
                            tf.less(tf.shape(token_ids)[1], max_token_length),
                            lambda: token_ids,
                            lambda: token_ids[:,:max_token_length])
                    candidate_ids = token_ids[1:]
                    data_schema[field_id]['token_ids'] = token_ids
                    data_schema[field_id]['candidate_ids'] = candidate_ids
            if limited_vocab:
                segmented_seqs = tf.gather(token_ids, seqs)
            else:
                if feature_type == 'class':
                    segmented_seqs = tf.expand_dims(seqs, axis=1)
                elif feature_type == 'sequence':
                    segmented_seqs = segment_words(seqs, segs)
                if not max_token_length is None:
                    segmented_seqs = tf.cond(
                        tf.less(tf.shape(segmented_seqs)[2], max_token_length),
                        lambda: segmented_seqs,
                        lambda: segmented_seqs[:,:,:max_token_length])
            feature['segmented_seqs'] = segmented_seqs

            seq_length = tf.shape(segmented_seqs)[1]
            feature['seq_length'] = seq_length
            if token_vocab is None:
                feature['token_length'] = tf.shape(segmented_seqs)[2]
            else:
                feature['token_length'] = tf.maximum(
                    tf.shape(segmented_seqs)[2], tf.shape(token_ids)[1])

            # embed words
            if not token_vocab is None:
                token_embeds = data_schema[field_id].get('token_embeds')
                candidate_embeds = data_schema[field_id].get('candidate_embeds')
                if token_embeds is None:
                    candidate_embeds, _ = word_embedder(
                        tf.expand_dims(candidate_ids, 0))
                    candidate_embeds = tf.squeeze(candidate_embeds, [0])
                    token_embeds = tf.pad(candidate_embeds, [[1,0],[0,0]])
                    data_schema[field_id]['token_embeds'] = token_embeds
                    data_schema[field_id]['candidate_embeds'] = candidate_embeds
            if limited_vocab:
                word_embeds = tf.gather(token_embeds, seqs)
                word_masks = tf.greater(seqs, 0)
            else:
                word_embeds, word_masks = word_embedder(
                    segmented_seqs)
            feature['word_embeds'] = word_embeds
            feature['word_masks'] = word_masks

            # field_encodes and posit_embeds
            field_query_embeds = tf.tile(
                tf.nn.embedding_lookup(field_query_embedding, [[field_id+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            feature['field_query_embeds'] = tuple(tf.split(field_query_embeds, num_layers, axis=2))
            field_key_embeds = tf.tile(
                tf.nn.embedding_lookup(field_key_embedding, [[field_id+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            feature['field_key_embeds'] = tuple(tf.split(field_key_embeds, num_layers, axis=2))
            field_value_embeds = tf.tile(
                tf.nn.embedding_lookup(field_value_embedding, [[field_id+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            feature['field_value_embeds'] = tuple(tf.split(field_value_embeds, num_layers, axis=2))
            if feature_type == 'sequence':
                posit_ids = tf.tile(
                    tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])
                posit_embeds = model_utils_py3.embed_position(
                    posit_ids,
                    posit_size)
            else:
                posit_embeds = tf.zeros([batch_size, seq_length, posit_size])
            feature['posit_embeds'] = posit_embeds

            # picking tokens
            if feature_type == 'sequence':
                pick_prob = 0.2
            elif feature_type == 'class':
                pick_prob = 1.0
            pick_masks = tf.less(tf.random_uniform([batch_size, seq_length]), pick_prob)
            pick_masks = tf.logical_and(pick_masks, word_masks)
            feature['pick_masks'] = pick_masks

            # tfstruct
            feature['tfstruct'] = model_utils_py3.TransformerStruct(
                field_query_embeds=feature['field_query_embeds'],
                field_key_embeds=feature['field_key_embeds'],
                field_value_embeds=feature['field_value_embeds'],
                posit_embeds=feature['posit_embeds'],
                token_embeds=feature['word_embeds'],
                masks=feature['word_masks'],
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )
            valid_masks = tf.logical_and(
                feature['word_masks'], tf.logical_not(feature['pick_masks']))
            feature['masked_tfstruct'] = model_utils_py3.TransformerStruct(
                field_query_embeds=feature['field_query_embeds'],
                field_key_embeds=feature['field_key_embeds'],
                field_value_embeds=feature['field_value_embeds'],
                posit_embeds=feature['posit_embeds'],
                token_embeds=feature['word_embeds'] * \
                    tf.cast(tf.expand_dims(valid_masks, axis=2), tf.float32),
                masks=valid_masks,
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )

        # add extra sample-level tfstruct
        global_tfstruct = model_utils_py3.TransformerStruct(
            field_query_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_query_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                num_layers,
                axis=-1)),
            field_key_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_key_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                num_layers,
                axis=-1)),
            field_value_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_value_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                num_layers,
                axis=-1)),
            posit_embeds=tf.zeros([batch_size, 1, posit_size]),
            token_embeds=tf.zeros([batch_size, 1, embed_size]),
            masks=tf.zeros([batch_size, 1], dtype=tf.bool),
            querys=None,
            keys=None,
            values=None,
            encodes=None,
        )

        # loop for target_level
        extra_tfstruct_list = []
        extra_feature_id_list = []
        for tlevel in range(0, max_target_level+1):

            # gather tfstructs
            tfstruct_list, masked_tfstruct_list, feature_id_list = [], [], []
            for i in features:
                if task_spec[data_index[i]['field_id']]['target_level'] == tlevel:
                    feature_id_list.append(i)
                    tfstruct_list.append(features[i]['tfstruct'])
                    masked_tfstruct_list.append(features[i]['masked_tfstruct'])

            # prepare attn_matrix
            attn_matrix = []
            for i in feature_id_list:
                field_id_i = data_index[i]['field_id']
                item_id_i = data_index[i]['item_id']
                group_id_i = data_schema[field_id_i]['group_id']
                attn_matrix_local = []
                for j in feature_id_list:
                    field_id_j = data_index[j]['field_id']
                    item_id_j = data_index[j]['item_id']
                    group_id_j = data_schema[field_id_j]['group_id']
                    if i == j:
                        attn_matrix_local.append(1)
                    elif group_id_i == group_id_j and item_id_i != item_id_j:
                        attn_matrix_local.append(0)
                    elif tlevel == 0:
                        attn_matrix_local.append(1)
                    else:
                        attn_matrix_local.append(0)
                for j in extra_feature_id_list:
                    field_id_j = data_index[j]['field_id']
                    item_id_j = data_index[j]['item_id']
                    group_id_j = data_schema[field_id_j]['group_id']
                    if group_id_i == group_id_j and item_id_i != item_id_j:
                        attn_matrix_local.append(0)
                    else:
                        attn_matrix_local.append(1)
                attn_matrix.append(attn_matrix_local)

            # get encodes
            if tlevel == 0:
                masked_tfstruct_list = [global_tfstruct] + masked_tfstruct_list
                attn_matrix = [[0]+item for item in attn_matrix]
                attn_matrix = [[0]+[1]*len(attn_matrix)]+attn_matrix
                masked_tfstruct_list = encode_tfstructs(
                    word_encoder, masked_tfstruct_list, attn_matrix)
                masked_global_tfstruct = masked_tfstruct_list[0]
                for i, feature_id in enumerate(feature_id_list):
                    features[feature_id]['masked_tfstruct'] = masked_tfstruct_list[i+1]
                if max_target_level > 0:
                    tfstruct_list = [global_tfstruct] + tfstruct_list
                    tfstruct_list = encode_tfstructs(
                        word_encoder, tfstruct_list, attn_matrix)
                    global_tfstruct = tfstruct_list[0]
                    for i, feature_id in enumerate(feature_id_list):
                        features[feature_id]['tfstruct'] = tfstruct_list[i+1]
                    extra_feature_id_list.extend(feature_id_list)
                    extra_tfstruct_list.extend(tfstruct_list[1:])
            elif len(feature_id_list) > 0:
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

        # get the loss of each feature
        metrics = {}
        loss_list = []
        for i, feature in features.items():

            field_id = data_index[i]['field_id']
            feature_type = data_schema[field_id]['type']
            limited_vocab = data_schema[field_id]['limited_vocab']
            token_vocab = data_schema[field_id].get('token_vocab')
            token_ids = data_schema[field_id].get('token_ids')
            token_embeds = data_schema[field_id].get('token_embeds')
            candidate_ids = data_schema[field_id].get('candidate_ids')
            candidate_embeds = data_schema[field_id].get('candidate_embeds')
            max_token_length = data_schema[field_id].get('max_token_length')
            target_level = task_spec[field_id]['target_level']
            copy_from = task_spec[field_id].get('copy_from')
            copy_from = [] if copy_from is None else copy_from

            # picked tokens
            pick_segmented_seqs = tf.boolean_mask(
                feature['segmented_seqs'],
                feature['pick_masks'])
            _, pick_word_encodes = tf.dynamic_partition(
                feature['masked_tfstruct'].encodes,
                tf.cast(feature['pick_masks'], tf.int32), 2)
            num_pick_words = tf.shape(pick_word_encodes)[0]
            item1 = (pick_word_encodes, None, 'context')

            def true_fn():
                # form candidates
                if limited_vocab:
                    match_idxs = tf.boolean_mask(feature['seqs'], feature['pick_masks']) - 1 # seqs is 1-based
                    match_matrix = tf.one_hot(match_idxs, tf.shape(candidate_embeds)[0])
                    local_candidate_ids = candidate_ids
                    local_candidate_embeds = candidate_embeds
                else:
                    # all features to select token from
                    select_from_segmented_seqs, select_from_word_embeds, select_from_word_masks = [], [], []
                    for j in features:
                        if j == i or data_index[j]['field_id'] in copy_from:
                            select_from_segmented_seqs.append(features[j]['segmented_seqs'])
                            select_from_word_embeds.append(features[j]['word_embeds'])
                            select_from_word_masks.append(features[j]['word_masks'])
                    select_from_segmented_seqs = model_utils_py3.pad_vectors(
                        select_from_segmented_seqs)
                    select_from_segmented_seqs = tf.concat(select_from_segmented_seqs, axis=1)
                    select_from_word_embeds = tf.concat(select_from_word_embeds, axis=1)
                    select_from_word_masks = tf.concat(select_from_word_masks, axis=1)

                    valid_segmented_seqs = tf.boolean_mask(
                        select_from_segmented_seqs, select_from_word_masks)
                    _, valid_word_embeds = tf.dynamic_partition(
                        select_from_word_embeds, tf.cast(select_from_word_masks, tf.int32), 2)

                    unique_segmented_seqs, unique_idxs = model_utils_py3.unique_vector(
                        valid_segmented_seqs)
                    # use tf.dynamic_partition to replace tf.gather
                    unique_1hots = tf.reduce_sum(
                        tf.one_hot(unique_idxs, tf.shape(valid_word_embeds)[0], dtype=tf.int32),
                        axis=0)
                    _, unique_word_embeds = tf.dynamic_partition(
                        valid_word_embeds,
                        unique_1hots, 2)
                    if candidate_ids is None:
                        local_candidate_ids = unique_segmented_seqs
                        local_candidate_embeds = unique_word_embeds
                    else:
                        extra_cand_ids = candidate_ids
                        unique_segmented_seqs, extra_cand_ids = model_utils_py3.pad_vectors(
                            [unique_segmented_seqs, extra_cand_ids])
                        extra_cand_match_matrix = model_utils_py3.match_vector(
                            extra_cand_ids,
                            unique_segmented_seqs)
                        extra_cand_masks = tf.logical_not(
                            tf.reduce_any(extra_cand_match_matrix, axis=-1))
                        local_candidate_ids = tf.concat(
                            [unique_segmented_seqs,
                             tf.boolean_mask(extra_cand_ids, extra_cand_masks)],
                            axis=0)
                        local_candidate_embeds = tf.concat(
                            [unique_word_embeds,
                             tf.boolean_mask(candidate_embeds, extra_cand_masks)],
                            axis=0)

                    local_pick_segmented_seqs, local_candidate_ids = model_utils_py3.pad_vectors(
                        [pick_segmented_seqs, local_candidate_ids])
                    match_matrix = model_utils_py3.match_vector(
                        local_pick_segmented_seqs,
                        local_candidate_ids)
                    match_matrix = tf.cast(match_matrix, tf.float32)
                    match_idxs = tf.argmax(match_matrix, axis=-1, output_type=tf.int32)
                num_candidates = tf.shape(local_candidate_ids)[0]
                item2 = (local_candidate_embeds, None, 'embed')
                word_select_logits = word_matcher(item1, item2)

                # retrieve copy encodes and add copy logits
                if len(copy_from) > 0:
                    copy_segmented_seqs, copy_encodes, \
                        copy_valid_masks, copy_pick_masks = [],[],[],[]
                    for j in features:
                        if data_index[j]['field_id'] in copy_from:
                            copy_segmented_seqs.append(features[j]['segmented_seqs'])
                            copy_encodes.append(features[j]['masked_tfstruct'].encodes)
                            copy_valid_masks.append(features[j]['masked_tfstruct'].masks)
                            copy_pick_masks.append(features[j]['pick_masks'])
                    # first we get the matrix indicates whether x copy to y
                    copy_segmented_seqs = model_utils_py3.pad_vectors(
                        copy_segmented_seqs)
                    copy_segmented_seqs = tf.concat(copy_segmented_seqs, axis=1)
                    copy_encodes = tf.concat(copy_encodes, axis=1)
                    copy_valid_masks = tf.concat(copy_valid_masks, axis=1)
                    copy_pick_masks = tf.concat(copy_pick_masks, axis=1)
                    copy_match_matrix = model_utils_py3.match_vector(
                        copy_segmented_seqs, copy_segmented_seqs)
                    copy_match_matrix = tf.logical_and(
                        copy_match_matrix, tf.expand_dims(copy_pick_masks, 1))
                    copy_match_matrix = tf.logical_and(
                        copy_match_matrix, tf.expand_dims(copy_valid_masks, 2))
                    # calculate the normalized prob each slot being copied
                    copy_scores = tf.cast(copy_match_matrix, tf.float32)
                    copy_scores /= (tf.reduce_sum(copy_scores, axis=1, keepdims=True)+1e-12)
                    copy_scores = tf.reduce_sum(copy_scores, axis=2)
                    # gather all valid slots which can be selected from
                    valid_segmented_seqs = tf.boolean_mask(
                        copy_segmented_seqs, copy_valid_masks)
                    valid_scores = tf.boolean_mask(
                        copy_scores, copy_valid_masks)
                    _, valid_encodes = tf.dynamic_partition(
                        copy_encodes, tf.cast(copy_valid_masks, tf.int32), 2)
                    valid_encodes = tf.pad(valid_encodes, [[0,1],[0,0]])
                    # for each candidate token, we gather the probs of all corresponding slots
                    local_candidate_ids, valid_segmented_seqs = model_utils_py3.pad_vectors(
                        [local_candidate_ids, valid_segmented_seqs])
                    valid_match_matrix = model_utils_py3.match_vector(
                        local_candidate_ids,
                        valid_segmented_seqs)
                    valid_match_matrix = tf.cast(valid_match_matrix, tf.float32)
                    valid_match_scores = valid_match_matrix * tf.expand_dims(
                        valid_scores+1e-12, axis=0)
                    # copy / no copy is 1:1
                    valid_pad_score = tf.reduce_sum(valid_match_scores, axis=1, keepdims=True)
                    valid_pad_score = tf.maximum(valid_pad_score, 1e-12)
                    valid_match_scores = tf.concat([valid_match_scores, valid_pad_score], axis=1)
                    sample_ids = tf.squeeze(
                        tf.random.categorical(tf.log(valid_match_scores), 1, dtype=tf.int32),
                        axis=[-1])
                    copy_masks = tf.not_equal(sample_ids, tf.shape(valid_encodes)[0]-1)
                    sample_onehots = tf.one_hot(sample_ids, tf.shape(valid_encodes)[0])
                    candidate_encodes = tf.matmul(sample_onehots, valid_encodes)
                    item2 = (candidate_encodes, copy_masks, 'encode')
                    word_select_logits += word_matcher(item1, item2)

                # sample-level logits
                if target_level == 0:
                    sample_token_logits = global_matcher(
                        (tf.squeeze(masked_global_tfstruct.encodes, [1]), None, 'context'),
                        (local_candidate_embeds, None, 'embed'))
                    sample_token_logits = tf.tile(
                        tf.expand_dims(sample_token_logits, axis=1),
                        [1,feature['seq_length'],1])
                    sample_token_logits = tf.boolean_mask(
                        sample_token_logits, feature['pick_masks'])
                    word_select_logits += sample_token_logits
                else:
                    sample_token_logits = global_matcher(
                        (tf.squeeze(global_tfstruct.encodes, [1]), None, 'context'),
                        (local_candidate_embeds, None, 'embed'))
                    sample_token_logits = tf.tile(
                        tf.expand_dims(sample_token_logits, axis=1),
                        [1,feature['seq_length'],1])
                    sample_token_logits = tf.boolean_mask(
                        sample_token_logits, feature['pick_masks'])
                    word_select_logits += sample_token_logits

                # word_select_loss
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

            if target_level > 0 or max_target_level == 0:
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
        params: {
            'data_index': [ ## list by feature_id
                {
                    'field_id': int,
                    'item_id': int,
                },
            ],
            'data_schema': [ ## list by field_id
                {
                    'type': 'sequence'|'class',
                    'limited_vocab': bool,
                    'token_vocab': MoleculeVocab,
                    'max_token_length': int,
                    'min_seq_length': int,
                    'max_seq_length': int,
                    'group_id': int,
                },
            ],
            'hyper_params': {
                'batch_size': int,
                'max_train_steps': int,
                'max_lr': float,
                'pct_start': 0~1,
                'dropout': 0~1,
                'wd': float,
            },
            schedule: '1cycle'|'lr_finder'
            num_steps: int
            distributed: bool
        """

        data_index = params['data_index']
        data_schema = params['data_schema']
        hyper_params = params['hyper_params']

        task_config = self.task_config
        task_spec = task_config['task_spec']

        model_config = self.model_config
        layer_size = model_config['layer_size']
        posit_size = int(layer_size/4)
        embed_size = layer_size
        num_layers = model_config['num_layers']
        num_heads = model_config['num_heads']
        word_match_size = 2*layer_size
        global_match_size = layer_size
        speller_layer_size = int(layer_size/2)
        speller_posit_size = int(speller_layer_size/4)
        speller_embed_size = speller_layer_size
        speller_num_layers = 2
        speller_num_heads = 4
        speller_match_size = speller_layer_size

        input_embedding, spellin_embedding, \
        field_query_embedding, \
        field_key_embedding, \
        field_value_embedding, \
        field_context_embedding, \
        field_word_embedding, \
        speller_context_embedding = get_embeddings(
            len(self.char_vocab), model_config['char_embed_dim'],
            self.char_vocab.embedding_init, num_layers, layer_size,
            word_match_size, speller_embed_size, speller_match_size, training=False)

        word_embedder = Embedder(
            input_embedding, embed_size,
            "Word_Embedder", dropout=None, training=False)
        word_encoder = Encoder(
            embed_size, posit_size, layer_size, num_layers, num_heads,
            "Word_Encoder", dropout=None, training=False)
        word_matcher = Matcher(
            word_match_size,
            "Word_Matcher", dropout=None, training=False)
        global_matcher = Matcher(
            global_match_size,
            "Global_Matcher", dropout=None, training=False)
        word_cell = TransformerCell("Word_Cell",
                                    word_encoder,
                                    dropout=None,
                                    training=False)
        speller_encoder = Encoder(
            speller_embed_size, speller_posit_size,
            speller_layer_size, speller_num_layers, speller_num_heads,
            "Speller_Encoder", dropout=None, training=False)
        speller_matcher = Matcher(
            speller_match_size,
            "Speller_Matcher", dropout=None, training=False)
        speller_cell = SpellerCell("Speller_Cell",
                                   speller_encoder,
                                   dropout=None,
                                   training=False)
        word_generator = WordGenerator(speller_cell, speller_matcher, spellin_embedding,
                                       self.char_vocab.token2id[self.char_vocab.sep])
        sent_generator = SentGenerator(
            word_cell, word_matcher, global_matcher, word_embedder, word_generator)
        class_generator = ClassGenerator(
            word_encoder, word_matcher, global_matcher, word_embedder, word_generator)

        batch_size = tf.shape(features[0]['seqs'])[0]
        max_target_level = max([item['target_level'] for item in task_spec])

        for i, feature in features.items():

            field_id = data_index[i]['field_id']
            feature_type = data_schema[field_id]['type']
            limited_vocab = data_schema[field_id]['limited_vocab']
            token_vocab = data_schema[field_id].get('token_vocab')
            max_token_length = data_schema[field_id].get('max_token_length')
            max_seq_length = data_schema[field_id].get('max_seq_length')
            target_level = task_spec[field_id]['target_level']
            seqs = feature['seqs']
            segs = feature['segs']

            # segment
            if not token_vocab is None:
                token_ids = data_schema[field_id].get('token_ids')
                candidate_ids = data_schema[field_id].get('candidate_ids')
                if token_ids is None:
                    token_ids = tf.constant(token_vocab.decompose_table, tf.int32)
                    if not max_token_length is None:
                        token_ids = tf.cond(
                            tf.less(tf.shape(token_ids)[1], max_token_length),
                            lambda: token_ids,
                            lambda: token_ids[:,:max_token_length])
                    candidate_ids = token_ids[1:]
                    data_schema[field_id]['token_ids'] = token_ids
                    data_schema[field_id]['candidate_ids'] = candidate_ids
            if limited_vocab:
                segmented_seqs = tf.gather(token_ids, seqs)
            else:
                if feature_type == 'class':
                    segmented_seqs = tf.expand_dims(seqs, axis=1)
                elif feature_type == 'sequence':
                    segmented_seqs = segment_words(seqs, segs)
                if not max_token_length is None:
                    segmented_seqs = tf.cond(
                        tf.less(tf.shape(segmented_seqs)[2], max_token_length),
                        lambda: segmented_seqs,
                        lambda: segmented_seqs[:,:,:max_token_length])
            feature['segmented_seqs'] = segmented_seqs

            # seq_length and token_length
            seq_length = tf.shape(segmented_seqs)[1]
            if token_vocab is None:
                token_length = tf.shape(segmented_seqs)[2]
            else:
                token_length = tf.maximum(
                    tf.shape(segmented_seqs)[2], tf.shape(token_ids)[1])
            feature['seq_length'] = seq_length
            feature['token_length'] = token_length

            # embed words
            if not token_vocab is None:
                token_embeds = data_schema[field_id].get('token_embeds')
                candidate_embeds = data_schema[field_id].get('candidate_embeds')
                if token_embeds is None:
                    candidate_embeds, _ = word_embedder(
                        tf.expand_dims(candidate_ids, 0))
                    candidate_embeds = tf.squeeze(candidate_embeds, [0])
                    token_embeds = tf.pad(candidate_embeds, [[1,0],[0,0]])
                    data_schema[field_id]['token_embeds'] = token_embeds
                    data_schema[field_id]['candidate_embeds'] = candidate_embeds
            if limited_vocab:
                word_embeds = tf.gather(token_embeds, seqs)
                word_masks = tf.greater(seqs, 0)
            else:
                word_embeds, word_masks = word_embedder(
                    segmented_seqs)
            feature['word_embeds'] = word_embeds
            feature['word_masks'] = word_masks

            # field_encodes and posit_embeds
            field_query_embeds = tf.tile(
                tf.nn.embedding_lookup(field_query_embedding, [[field_id+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            feature['field_query_embeds'] = tuple(tf.split(field_query_embeds, num_layers, axis=2))
            field_key_embeds = tf.tile(
                tf.nn.embedding_lookup(field_key_embedding, [[field_id+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            feature['field_key_embeds'] = tuple(tf.split(field_key_embeds, num_layers, axis=2))
            field_value_embeds = tf.tile(
                tf.nn.embedding_lookup(field_value_embedding, [[field_id+1,]]),# field embeds 0 is reserved
                [batch_size, seq_length, 1])
            feature['field_value_embeds'] = tuple(tf.split(field_value_embeds, num_layers, axis=2))
            if feature_type == 'sequence':
                posit_ids = tf.tile(
                    tf.expand_dims(tf.range(seq_length), 0), [batch_size, 1])
                posit_embeds = model_utils_py3.embed_position(
                    posit_ids,
                    posit_size)
            else:
                posit_embeds = tf.zeros([batch_size, seq_length, posit_size])
            feature['posit_embeds'] = posit_embeds

            # tfstruct
            feature['tfstruct'] = model_utils_py3.TransformerStruct(
                field_query_embeds=feature['field_query_embeds'],
                field_key_embeds=feature['field_key_embeds'],
                field_value_embeds=feature['field_value_embeds'],
                posit_embeds=feature['posit_embeds'],
                token_embeds=feature['word_embeds'],
                masks=feature['word_masks'],
                querys=None,
                keys=None,
                values=None,
                encodes=None,
            )

        # add extra sample-level tfstruct
        global_tfstruct = model_utils_py3.TransformerStruct(
            field_query_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_query_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                num_layers,
                axis=-1)),
            field_key_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_key_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                num_layers,
                axis=-1)),
            field_value_embeds=tuple(tf.split(
                tf.tile(
                    tf.nn.embedding_lookup(field_value_embedding, [[0,]]),
                    [batch_size, 1, 1]),
                num_layers,
                axis=-1)),
            posit_embeds=tf.zeros([batch_size, 1, posit_size]),
            token_embeds=tf.zeros([batch_size, 1, embed_size]),
            masks=tf.zeros([batch_size, 1], dtype=tf.bool),
            querys=None,
            keys=None,
            values=None,
            encodes=None,
        )

        # loop for non targets
        tfstruct_list, feature_id_list = [], []
        for i in features:
            if task_spec[data_index[i]['field_id']]['target_level'] == 0:
                feature_id_list.append(i)
                tfstruct_list.append(features[i]['tfstruct'])

        # prepare attn_matrix
        attn_matrix = []
        for i in feature_id_list:
            field_id_i = data_index[i]['field_id']
            item_id_i = data_index[i]['item_id']
            group_id_i = data_schema[field_id_i]['group_id']
            attn_matrix_local = []
            for j in feature_id_list:
                field_id_j = data_index[j]['field_id']
                item_id_j = data_index[j]['item_id']
                group_id_j = data_schema[field_id_j]['group_id']
                if i == j:
                    attn_matrix_local.append(1)
                elif group_id_i == group_id_j and item_id_i != item_id_j:
                    attn_matrix_local.append(0)
                else:
                    attn_matrix_local.append(1)
            attn_matrix.append(attn_matrix_local)
        attn_matrix = [[0]+item for item in attn_matrix]
        attn_matrix = [[0]+[1]*len(attn_matrix)] + attn_matrix

        # get encodes
        tfstruct_list = encode_tfstructs(
            word_encoder, [global_tfstruct]+tfstruct_list, attn_matrix)
        global_tfstruct = tfstruct_list[0]
        global_encodes = tf.squeeze(global_tfstruct.encodes, [1])
        for i, feature_id in enumerate(feature_id_list):
            features[feature_id]['tfstruct'] = tfstruct_list[i+1]

        # null tfstruct
        null_tfstruct = model_utils_py3.init_tfstruct(
            batch_size, word_encoder.embed_size, word_encoder.posit_size,
            word_encoder.layer_size, word_encoder.num_layers)

        # loop for target_level
        predictions = {}
        start_level = 1 if max_target_level > 0 else 0
        for tlevel in range(start_level, max_target_level+1):

            # generate one-by-one
            for i, feature in features.items():

                field_id = data_index[i]['field_id']
                item_id = data_index[i]['item_id']
                group_id = data_schema[field_id]['group_id']
                feature_type = data_schema[field_id]['type']
                limited_vocab = data_schema[field_id]['limited_vocab']
                token_vocab = data_schema[field_id].get('token_vocab')
                token_ids = data_schema[field_id].get('token_ids')
                token_embeds = data_schema[field_id].get('token_embeds')
                candidate_ids = data_schema[field_id].get('candidate_ids')
                candidate_embeds = data_schema[field_id].get('candidate_embeds')
                max_token_length = data_schema[field_id].get('max_token_length')
                max_seq_length = data_schema[field_id].get('max_seq_length')
                target_level = task_spec[field_id]['target_level']
                copy_from = task_spec[field_id].get('copy_from')
                copy_from = [] if copy_from is None else copy_from

                if target_level == tlevel:
                    extra_tfstruct_list, extra_feature_id_list = [], []
                    for j in features:
                        field_id_j = data_index[j]['field_id']
                        item_id_j = data_index[j]['item_id']
                        group_id_j = data_schema[field_id_j]['group_id']
                        target_level_j = task_spec[field_id_j]['target_level']
                        if i == j:
                            continue
                        elif group_id == group_id_j and item_id != item_id_j:
                            continue
                        elif target_level_j >= tlevel and tlevel > 0:
                            continue
                        else:
                            extra_tfstruct_list.append(features[j]['tfstruct'])
                            extra_feature_id_list.append(j)
                    if len(extra_tfstruct_list) > 0:
                        extra_tfstruct = model_utils_py3.concat_tfstructs(extra_tfstruct_list)
                    else:
                        extra_tfstruct = null_tfstruct

                    # generate
                    if feature_type == 'sequence':
                        initial_state = TransformerState(
                            field_query_embedding=tuple(tf.split(
                                tf.tile(
                                    tf.nn.embedding_lookup(field_query_embedding, [field_id+1,]),
                                    [batch_size, 1]),
                                num_layers,
                                axis=-1)),
                            field_key_embedding=tuple(tf.split(
                                tf.tile(
                                    tf.nn.embedding_lookup(field_key_embedding, [field_id+1,]),
                                    [batch_size, 1]),
                                num_layers,
                                axis=-1)),
                            field_value_embedding=tuple(tf.split(
                                tf.tile(
                                    tf.nn.embedding_lookup(field_value_embedding, [field_id+1,]),
                                    [batch_size, 1]),
                                num_layers,
                                axis=-1)),
                            dec_masks=tf.zeros([batch_size, 0], dtype=tf.bool),
                            dec_keys=(
                                tf.zeros([batch_size, 0, layer_size]),
                            )*num_layers,
                            dec_values=(
                                tf.zeros([batch_size, 0, layer_size]),
                            )*num_layers,
                            enc_tfstruct=extra_tfstruct,
                        )
                        if limited_vocab:
                            sep_id = token_vocab.token2id[token_vocab.sep]
                            word_embedding = candidate_embeds
                            word_embedding = tf.concat(
                                [word_embedding[sep_id-1:sep_id],
                                 word_embedding[:sep_id-1], word_embedding[sep_id:]],
                                axis=0)
                            word_ids = tf.range(1, tf.shape(word_embedding)[0]+1, dtype=tf.int32)
                            word_ids = tf.concat(
                                [word_ids[sep_id-1:sep_id],
                                 word_ids[:sep_id-1], word_ids[sep_id:]],
                                axis=0)
                            seqs, scores = sent_generator.generate(
                                initial_state, max_seq_length, global_encodes,
                                word_embedding=word_embedding, word_ids=word_ids)
                            feature['seqs'] = seqs[:,0]
                            feature['segs'] = tf.zeros([batch_size, 0])
                            feature['segmented_seqs'] = tf.gather(
                                token_ids, feature['seqs'])
                        else:
                            if candidate_embeds is None:
                                seqs, scores = sent_generator.generate(
                                    initial_state, max_seq_length, global_encodes,
                                    gen_word_len=max_token_length)
                            else:
                                unk_ids = tf.constant(
                                    [[self.char_vocab.token2id[self.char_vocab.unk]]],
                                    dtype=tf.int32)
                                unk_ids = tf.pad(
                                    unk_ids,
                                    [[0,0],[0,tf.shape(candidate_ids)[1]-1]])
                                unk_masks = tf.reduce_all(
                                    tf.equal(candidate_ids, unk_ids), axis=1)
                                valid_masks = tf.logical_not(unk_masks)
                                candidate_ids = tf.boolean_mask(candidate_ids, valid_masks)
                                candidate_embeds = tf.boolean_mask(candidate_embeds, valid_masks)
                                sep_ids = tf.constant(
                                    [[self.char_vocab.token2id[self.char_vocab.sep]]],
                                    dtype=tf.int32)
                                sep_ids = tf.pad(
                                    sep_ids,
                                    [[0,0],[0,tf.shape(candidate_ids)[1]-1]])
                                sep_embeds, _ = word_embedder(tf.expand_dims(sep_ids, axis=0))
                                sep_embeds = tf.squeeze(sep_embeds, axis=0)
                                candidate_embeds = tf.concat(
                                    [sep_embeds, candidate_embeds], axis=0)
                                candidate_ids = tf.concat(
                                    [sep_ids, candidate_ids], axis=0)
                                seqs, scores = sent_generator.generate(
                                    initial_state, max_seq_length, global_encodes,
                                    word_embedding=candidate_embeds,
                                    word_ids=candidate_ids)
                            feature['segmented_seqs'] = seqs[:,0]
                            feature['seqs'], feature['segs'] = model_utils_py3.stitch_chars(
                                feature['segmented_seqs'])
                            feature['segs'] = tf.pad(
                                feature['segs'], [[0,0],[1,1]], constant_values=1.0)
                    elif feature_type == 'class':
                        tfstruct = model_utils_py3.TransformerStruct(
                            field_query_embeds=tuple(tf.split(
                                tf.tile(
                                    tf.nn.embedding_lookup(field_query_embedding, [[field_id+1,]]),
                                    [batch_size, 1, 1]),
                                num_layers,
                                axis=-1)),
                            field_key_embeds=tuple(tf.split(
                                tf.tile(
                                    tf.nn.embedding_lookup(field_key_embedding, [[field_id+1,]]),
                                    [batch_size, 1, 1]),
                                num_layers,
                                axis=-1)),
                            field_value_embeds=tuple(tf.split(
                                tf.tile(
                                    tf.nn.embedding_lookup(field_value_embedding, [[field_id+1,]]),
                                    [batch_size, 1, 1]),
                                num_layers,
                                axis=-1)),
                            posit_embeds=tf.zeros([batch_size, 1, posit_size]),
                            token_embeds=tf.zeros([batch_size, 1, embed_size]),
                            masks=tf.zeros([batch_size, 1], dtype=tf.bool),
                            querys=None,
                            keys=None,
                            values=None,
                            encodes=None,
                        )
                        if limited_vocab:
                            word_embedding = candidate_embeds
                            word_ids = tf.range(1, tf.shape(word_embedding)[0]+1, dtype=tf.int32)
                            classes, scores = class_generator.generate(
                                tfstruct, extra_tfstruct, global_encodes,
                                word_embedding=word_embedding, word_ids=word_ids)
                            feature['seqs'] = classes[:,0]
                            feature['segs'] = tf.zeros([batch_size, 0])
                            feature['segmented_seqs'] = tf.gather(
                                token_ids, feature['seqs'])
                        else:
                            if candidate_embeds is None:
                                classes, scores = class_generator.generate(
                                    tfstruct, extra_tfstruct, global_encodes,
                                    gen_word_len=max_token_len)
                            else:
                                unk_ids = tf.constant(
                                    [[self.char_vocab.token2id[self.char_vocab.unk]]],
                                    dtype=tf.int32)
                                unk_ids = tf.pad(
                                    unk_ids,
                                    [[0,0],[0,tf.shape(candidate_ids)[1]-1]])
                                unk_masks = tf.reduce_all(
                                    tf.equal(candidate_ids, unk_ids), axis=1)
                                valid_masks = tf.logical_not(unk_masks)
                                candidate_ids = tf.boolean_mask(candidate_ids, valid_masks)
                                candidate_embeds = tf.boolean_mask(candidate_embeds, valid_masks)
                                classes, scores = class_generator.generate(
                                    tfstruct, extra_tfstruct, global_encodes,
                                    word_embedding=candidate_embeds, word_ids=candidate_ids)
                            feature['segmented_seqs'] = seqs[:,0]
                            feature['seqs'], feature['segs'] = model_utils_py3.stitch_chars(
                                feature['segmented_seqs'])
                            feature['segs'] = tf.pad(
                                feature['segs'], [[0,0],[1,1]], constant_values=1.0)

                    # add to predictions
                    predictions[str(i)+'-seqs'] = feature['seqs']
                    predictions[str(i)+'-segs'] = feature['segs']

                    if tlevel < max_target_level:
                        # embed words
                        if limited_vocab:
                            word_embeds = tf.gather(
                                token_embeds, feature['seqs'])
                            word_masks = tf.greater(feature['seqs'], 0)
                        else:
                            word_embeds, word_masks = word_embedder(
                                feature['segmented_seqs'])
                        feature['word_embeds'] = word_embeds
                        feature['word_masks'] = word_masks

                        # encode
                        tfstruct = model_utils_py3.TransformerStruct(
                            field_query_embeds=feature['field_query_embeds'],
                            field_key_embeds=feature['field_key_embeds'],
                            field_value_embeds=feature['field_value_embeds'],
                            posit_embeds=feature['posit_embeds'],
                            token_embeds=feature['word_embeds'],
                            masks=feature['word_masks'],
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
                        feature['tfstruct'] = word_encoder(tfstruct, attn_masks, extra_tfstruct)


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
