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
        tf.logging.info("Create InsideHook.")
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


""" lm model fn """

def lm_model_fn(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode,     # An instance of tf.estimator.ModeKeys
    params):  # Additional configuration
    """
    lm model function for tf.estimator
    features:
        [(seqs, segs),...]
    params:
        max_lr: maximum learning rate
        num_steps: num steps to train
        pct_start: % of increasing lr phase
        dropout: dropout ratio
        wd: weight decay
        char_vocab_size: size of character vocab
        char_vocab_dim: dimension of character embedding
        char_vocab_emb: pretrained character embedding
        max_char_length: maximum character length of a word
        layer_size: size of the layer
        num_layers: num of layers
        data_spec: [{'type': 'seq'|'class', 'is_target': bool, 'vocab': None|np.array}]
    """

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    input_embedding, spellin_embedding, spellout_embedding, field_embedding = get_embeddings(
        params['char_vocab_size'], params['char_vocab_dim'], params.get('char_vocab_emb'),
        params['layer_size'], training)

    segmented_seqs_ref_list = []
    word_embeds_ref_list, word_masks_ref_list, word_ids_ref_list = [],[],[]
    seq_length_ref_list, pick_masks_ref_list, field_embeds_ref_list = [],[],[]
    word_embed_loss_ref_list, word_select_loss_ref_list, word_gen_loss_ref_list = [],[],[]
    reuse=None
    for i, piece in features.items():
            
        seqs = piece['seq']
        segs = piece['seg']
        segmented_seqs_ref = segment_words(
            seqs, segs, params['max_char_length'], reuse=reuse)
        segmented_seqs_ref_list.append(segmented_seqs_ref)

        batch_size = tf.shape(segmented_seqs_ref)[0]
        max_length = tf.shape(segmented_seqs_ref)[1]
        seq_length_ref_list.append(max_length)

        # embed words
        word_embeds_ref, word_masks_ref = embed_words(
            segmented_seqs_ref, input_embedding, params['layer_size'],
            params.get('dropout'), training, reuse=reuse)
        word_embeds_ref_list.append(word_embeds_ref)
        word_masks_ref_list.append(word_masks_ref)

        posit_ids = tf.tile(tf.expand_dims(tf.range(max_length), 0), [batch_size, 1])
        posit_embeds = model_utils_py3.embed_position(
            posit_ids,
            word_embeds_ref.get_shape()[-1].value)
        field_embeds = field_embedding[i+1] + posit_embeds
        field_embeds_ref_list.append(field_embeds)

        pick_masks_ref = tf.less(tf.random_uniform([batch_size, max_length]), 0.15)
        pick_masks_ref = tf.logical_and(pick_masks_ref, word_masks_ref)
        pick_masks_ref_list.append(pick_masks_ref)
        masked_word_embeds_ref = word_embeds_ref * \
            tf.cast(tf.expand_dims(tf.logical_not(pick_masks_ref), axis=-1), tf.float32)

        reuse=True

    # get the encodes
    word_masks_ref = tf.concat(word_masks_ref_list, axis=1)
    pick_masks_ref = tf.concat(pick_masks_ref_list, axis=1)
    field_embeds_ref = tf.concat(
        [tf.tile(tf.nn.embedding_lookup(field_embedding, [[0,]]), [batch_size,1,1])] + \
            field_embeds_ref_list,
        axis=1)
    word_embeds_ref = tf.pad(tf.concat(word_embeds_ref_list, axis=1), [[0,0],[1,0],[0,0]])
    masked_word_embeds_ref = word_embeds_ref * \
        tf.cast(tf.expand_dims(
            tf.logical_not(tf.pad(pick_masks_ref, [[0,0],[1,0]])), axis=-1), tf.float32)
    attn_masks = tf.pad(word_masks_ref, [[0,0],[1,0]], constant_values=True)
    masked_attn_masks = tf.logical_and(word_masks_ref, tf.logical_not(pick_masks_ref))
    masked_attn_masks = tf.pad(masked_attn_masks, [[0,0],[1,0]], constant_values=True)
    encodes_ref = encode_words(
        field_embeds_ref, word_embeds_ref, attn_masks,
        params['num_layers'], params.get('dropout'), training)
    masked_encodes_ref = encode_words(
        field_embeds_ref, masked_word_embeds_ref, masked_attn_masks,
        params['num_layers'], params.get('dropout'), training, reuse=True)

    # get the loss of each piece
    encodes_ref_list = tf.split(encodes_ref, [1,]+seq_length_ref_list, axis=1)
    masked_encodes_ref_list = tf.split(
        masked_encodes_ref, [1,]+seq_length_ref_list, axis=1)
    reuse = None
    for i, (segmented_seqs_ref, word_embeds_ref, masked_word_encodes_ref, \
        word_masks_ref, pick_masks_ref) in \
        enumerate(zip(segmented_seqs_ref_list, word_embeds_ref_list, masked_encodes_ref_list[1:],
            word_masks_ref_list, pick_masks_ref_list)):

        loss_weight = 1.0 if params['data_spec'][i]['is_target'] else 0.1

        valid_segmented_seqs_ref = tf.boolean_mask(segmented_seqs_ref, word_masks_ref)
        pick_segmented_seqs_ref = tf.boolean_mask(segmented_seqs_ref, pick_masks_ref)
        _, valid_word_embeds_ref = tf.dynamic_partition(
            word_embeds_ref, tf.cast(word_masks_ref, tf.int32), 2)
        _, pick_word_encodes_ref = tf.dynamic_partition(
            masked_word_encodes_ref, tf.cast(pick_masks_ref, tf.int32), 2)
        num_pick_words = tf.shape(pick_word_encodes_ref)[0]

        unique_segmented_seqs_ref, unique_idxs = model_utils_py3.unique_2d(valid_segmented_seqs_ref)
        # use tf.dynamic_partition to replace th.gather
        unique_1hots = tf.reduce_sum(
            tf.one_hot(unique_idxs, tf.shape(valid_word_embeds_ref)[0], dtype=tf.int32),
            axis=0)
        _, unique_word_embeds_ref = tf.dynamic_partition(
            valid_word_embeds_ref,
            unique_1hots, 2)

        match_matrix = tf.equal(
            tf.expand_dims(pick_segmented_seqs_ref, 1),
            tf.expand_dims(unique_segmented_seqs_ref, 0))
        match_matrix = tf.reduce_all(match_matrix, axis=-1)
        match_idxs = tf.argmax(tf.cast(match_matrix, tf.int32), axis=-1)

        word_select_logits_ref = match_embeds(
            pick_word_encodes_ref, unique_word_embeds_ref,
            params.get('dropout'), training, reuse=reuse)
        word_select_loss_ref = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=match_idxs, logits=word_select_logits_ref)
        word_select_loss_ref = tf.reduce_mean(word_select_loss_ref)
        word_select_loss_ref_list.append(loss_weight*word_select_loss_ref)

        word_gen_loss_ref = train_speller(
            tf.expand_dims(pick_word_encodes_ref, 1),
            tf.ones([num_pick_words, 1], dtype=tf.bool),
            pick_segmented_seqs_ref,
            params['layer_size'], spellin_embedding, spellout_embedding,
            params.get('dropout'), training, reuse=reuse)
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
    print("total number of parameters is: {}".format(total_params))
    
    # return EstimatorSpec
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        learning_rate, momentum = training_schedule(
            params['schedule'], global_step, params.get('max_lr'), params['num_steps'], params.get('pct_start'))
        init_op1 = tf.variables_initializer(tf.global_variables())
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=momentum,
            beta2=0.99)
        train_op = model_utils_py3.optimize_loss(
            loss,
            global_step,
            optimizer,
            wd=params.get('wd'),
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
            mode, loss=loss)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'features': features,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
