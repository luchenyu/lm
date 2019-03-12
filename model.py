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
        tf.logging.info('Create InsideHook.')
        self.fetches = fetches
        self.num_steps = num_steps
        self.learning_rates = []
        self.losses = []
        self.losses_smoothed = []
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)


    def begin(self):
      # You can add ops to the graph here.

        print('Before starting the session.')

      # 1. Create saver

      #exclusions = []
      #if self.checkpoint_exclude_scopes:
      #  exclusions = [scope.strip()
      #                for scope in self.checkpoint_exclude_scopes.split(',')]
      #
      #variables_to_restore = []
      #for var in slim.get_model_variables(): #tf.global_variables():
      #  excluded = False
      #  for exclusion in exclusions:
      #    if var.op.name.startswith(exclusion):
      #      excluded = True
      #      break
      #  if not excluded:
      #    variables_to_restore.append(var)
      #inclusions
      #[var for var in tf.trainable_variables() if var.op.name.startswith('InceptionResnetV1')]

#       self.saver = tf.train.Saver()


    def after_create_session(self, session, coord):
      # When this is called, the graph is finalized and
      # ops can no longer be added to the graph.

        print('Session created.')

#       tf.logging.info('Fine-tuning from %s' % self.checkpoint_path)
#       self.saver.restore(session, os.path.expanduser(self.checkpoint_path))
#       tf.logging.info('End fineturn from %s' % self.checkpoint_path)

    def before_run(self, run_context):
        
        return tf.train.SessionRunArgs(self.fetches)

    def after_run(self, run_context, run_values):
#         print(run_values.results)
        learning_rate = run_values.results['learning_rate']
        loss = run_values.results['loss']
        self.learning_rates.append(learning_rate)
        self.losses.append(loss)
        self.ax.clear()
        self.ax.semilogx(self.learning_rates, self.losses)
        if len(self.losses_smoothed) == 0:
            self.losses_smoothed.append(loss)
        else:
            self.losses_smoothed.append(0.5*self.losses_smoothed[-1] + 0.5*loss)
        window_size = int(0.04*self.num_steps)
        if min(self.losses_smoothed[-window_size:]) > min(self.losses_smoothed) \
            and self.losses_smoothed[-1] >= max(self.losses_smoothed[-window_size:-1]):
            run_context.request_stop()

    def end(self, session):
        self.fig.show()
        self.fig.canvas.draw()


""" lm model """

class Model(object):
    def __init__(self, model_config):
        """
        args:
            model_config: {'char_vocab_size': int, 'char_vocab_dim': int, 'char_vocab_emb': np.array, 'layer_size': int, 'num_layers': int}
        """
        self.model_config = model_config
        
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
        input_embedding, spellin_embedding, spellout_embedding, field_embedding = get_embeddings(
            model_config['char_vocab_size'], model_config['char_vocab_dim'], model_config.get('char_vocab_emb'),
            model_config['layer_size'], training)

        segmented_seqs_ref_list = []
        word_embeds_ref_list, word_masks_ref_list = [],[]
        seq_length_ref_list, pick_masks_ref_list, field_embeds_ref_list = [],[],[]
        word_select_loss_ref_list, word_gen_loss_ref_list = [],[]
        reuse=None
        for i, piece in features.items():

            seqs = piece['seqs']
            segs = piece['segs']
            token_char_ids = schema[i].get('token_char_ids')
            if token_char_ids != None:
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
            segmented_seqs_ref_list.append(segmented_seqs_ref)

            batch_size = tf.shape(segmented_seqs_ref)[0]
            max_length = tf.shape(segmented_seqs_ref)[1]
            seq_length_ref_list.append(max_length)

            # embed words
            word_embeds_ref, word_masks_ref = embed_words(
                segmented_seqs_ref, input_embedding, model_config['layer_size'],
                run_config.get('dropout'), training, reuse=reuse)
            word_embeds_ref_list.append(word_embeds_ref)
            word_masks_ref_list.append(word_masks_ref)

            field_embeds = field_embedding[i+1] + tf.zeros_like(word_embeds_ref)
            if schema[i]['type'] == 'sequence':
                posit_ids = tf.tile(tf.expand_dims(tf.range(max_length), 0), [batch_size, 1])
                posit_embeds = model_utils_py3.embed_position(
                    posit_ids,
                    model_config['layer_size'])
                field_embeds += posit_embeds
            field_embeds_ref_list.append(field_embeds)

            if training:
                pick_masks_ref = tf.less(tf.random_uniform([batch_size, max_length]), 0.15)
                pick_masks_ref = tf.logical_and(pick_masks_ref, word_masks_ref)
            elif run_config['data'][i]['is_target']:
                if schema[i]['type'] == 'sequence':
                    pick_masks_ref = tf.less(tf.random_uniform([batch_size, max_length]), 0.15)
                    pick_masks_ref = tf.logical_and(pick_masks_ref, word_masks_ref)
                elif schema[i]['type'] == 'class':
                    pick_masks_ref = tf.ones([batch_size, max_length], dtype=tf.bool)
            else:
                pick_masks_ref = tf.zeros([batch_size, max_length], dtype=tf.bool)
            pick_masks_ref_list.append(pick_masks_ref)

            reuse=True

        # get the encodes
        word_masks_ref = tf.concat(word_masks_ref_list, axis=1)
        pick_masks_ref = tf.concat(pick_masks_ref_list, axis=1)
        field_embeds_ref = tf.concat(field_embeds_ref_list, axis=1)
        field_embeds_ref = tf.concat(
            [tf.tile(tf.nn.embedding_lookup(field_embedding, [[0,]]), [tf.shape(field_embeds_ref)[0],1,1]),
            field_embeds_ref],
            axis=1)
        word_embeds_ref = tf.pad(tf.concat(word_embeds_ref_list, axis=1), [[0,0],[1,0],[0,0]])
        masked_word_embeds_ref = word_embeds_ref * \
            tf.cast(tf.expand_dims(
                tf.logical_not(tf.pad(pick_masks_ref, [[0,0],[1,0]])), axis=-1), tf.float32)
        masked_attn_masks = tf.logical_and(word_masks_ref, tf.logical_not(pick_masks_ref))
        masked_attn_masks = tf.pad(masked_attn_masks, [[0,0],[1,0]], constant_values=True)
        masked_encodes_ref = encode_words(
            field_embeds_ref, masked_word_embeds_ref, masked_attn_masks,
            model_config['num_layers'], run_config.get('dropout'), training)

        # get the loss of each piece
        masked_encodes_ref_list = tf.split(
            masked_encodes_ref, [1,]+seq_length_ref_list, axis=1)
        metrics = {}
        reuse = None
        for i, (segmented_seqs_ref, word_embeds_ref, masked_word_encodes_ref, \
            word_masks_ref, pick_masks_ref) in \
            enumerate(zip(segmented_seqs_ref_list, word_embeds_ref_list, masked_encodes_ref_list[1:],
                word_masks_ref_list, pick_masks_ref_list)):

            loss_weight = 1.0 if run_config['data'][i]['is_target'] else 0.1

            _, pick_word_encodes_ref = tf.dynamic_partition(
                masked_word_encodes_ref, tf.cast(pick_masks_ref, tf.int32), 2)
            token_char_ids = schema[i].get('token_char_ids')

            # word_select_loss
            if token_char_ids != None:
                candidate_word_embeds = embed_words(
                    tf.expand_dims(token_char_ids, 0), input_embedding, model_config['layer_size'],
                    run_config.get('dropout'), training, reuse=True)
                candidate_word_embeds = tf.squeeze(candidate_word_embeds, 0)
                match_idxs = tf.boolean_mask(features[i]['seqs'], pick_masks_ref)
            else:
                valid_segmented_seqs_ref = tf.boolean_mask(segmented_seqs_ref, word_masks_ref)
                pick_segmented_seqs_ref = tf.boolean_mask(segmented_seqs_ref, pick_masks_ref)
                _, valid_word_embeds_ref = tf.dynamic_partition(
                    word_embeds_ref, tf.cast(word_masks_ref, tf.int32), 2)
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
            word_select_loss_ref_list.append(loss_weight*word_select_loss_ref)

            # eval metrics
            if mode == tf.estimator.ModeKeys.EVAL and run_config['data'][i]['is_target']:
                predicted_classes = tf.argmax(word_select_logits_ref, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=match_idxs,
                    predictions=predicted_classes)
                metrics[i] = accuracy

            # word_gen_loss
            if token_char_ids == None:
                word_gen_loss_ref = train_speller(
                    tf.expand_dims(pick_word_encodes_ref, 1),
                    tf.ones([num_pick_words, 1], dtype=tf.bool),
                    pick_segmented_seqs_ref,
                    model_config['layer_size'], spellin_embedding, spellout_embedding,
                    run_config.get('dropout'), training, reuse=reuse)
                word_gen_loss_ref /= float(run_config['data'][i]['max_token_length'])
                word_gen_loss_ref = tf.cond(
                    tf.greater(tf.size(word_gen_loss_ref), 0),
                    lambda: tf.reduce_mean(word_gen_loss_ref),
                    lambda: tf.zeros([]))
            else:
                word_gen_loss_ref = tf.zeros([])
            word_gen_loss_ref_list.append(loss_weight*word_gen_loss_ref)

            reuse = True

        loss_mat = tf.stack([tf.stack(word_select_loss_ref_list), tf.stack(word_gen_loss_ref_list)], axis=-1)
        loss = tf.reduce_sum(loss_mat)

        # print total num of parameters
        total_params = 0
        for var in tf.trainable_variables():
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
            learning_rate, momentum = training_schedule(
                params['schedule'], global_step, run_config.get('max_lr'), params['num_steps'], run_config.get('pct_start'))
            init_op1 = tf.variables_initializer(tf.global_variables())
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=momentum,
                beta2=0.99)
            train_op = model_utils_py3.optimize_loss(
                loss,
                global_step,
                optimizer,
                wd=run_config.get('wd'),
                scope=None)
            with tf.control_dependencies([init_op1]):
                init_op2 = tf.variables_initializer(tf.global_variables())
            def init_fn(scaffold, session):
                session.run(init_op2)
            if params.get('distributed') == True:
                scaffold = tf.train.Scaffold(init_op=init_op2)
            else:
                scaffold = tf.train.Scaffold(init_op=init_op1, init_fn=init_fn)
            if params.get('schedule') == 'lr_finder':
                scaffold = tf.train.Scaffold(init_op=init_op2)
                fetches = {'global_step': global_step, 'learning_rate': learning_rate, 'loss': loss}
                hooks = [LRFinderHook(fetches, params['num_steps'])]
            else:
                hooks = []
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op, training_hooks=hooks, scaffold=scaffold)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'features': features,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
