import logging, math, os, random
import numpy as np
import tensorflow as tf
from utils import data_utils_py3, model_utils_py3


"""training hook"""

class InsideHook(tf.train.SessionRunHook):
    def __init__(
        self,
        fetches,
        history):
        tf.logging.info("Create InsideHook.")
        self.fetches = fetches
        self.history = history

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
#         if self.history != None:
#             self.history.append(run_values.results)
        pass

    def end(self, session):
        
        pass

class OutsideHook(tf.train.SessionRunHook):
    def __init__(
        self):
        tf.logging.info("Create OutsideHook.")

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
        
        return None

    def after_run(self, run_context, run_values):
        
        print(run_values.results)

    def end(self, session):
        
        pass


""" input_fn """

def file_input_fn(
    vocab,
    data_path,
    num_pieces,
    batch_size,
    max_lengths,
    training):
    """
    An input function for training
    each line is an example
    each example contains one or more pieces separated by '***|||***'
    each piece contains one or more paragraphs separated by '\t'
    each paragraph contains one or more words separated by ' '
    args:
        vocab: Vocab object
        data_path: location of data files
        num_pieces: how many field pieces each example has
        batch_size: batch size
        max_lengths: list of max lengths of words for each piece
        training: bool
    """

    if os.path.isdir(data_path):
        filenames = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]
        random.shuffle(filenames)
    else:
        filenames = [data_path]
    dataset = tf.data.TextLineDataset(filenames)

    if training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=50000)

    def _featurize(text):
        text = text.numpy().decode('utf-8').strip()
        pieces = text.split('***|||***')
        def _tokenize(i, piece):
            paras = piece.split('\t')
            paras = [p.split(' ')+['\t'] for p in paras]
            words = [w for p in paras for w in p]
            # random sampling if too long
            if len(words) > max_lengths[i]:
                start = random.randint(0, len(words)-max_lengths[i]+1)
                words = words[start:start+max_lengths[i]]
            seq, seg = data_utils_py3.words_to_token_ids(words, vocab)
            seq = np.array(seq, dtype=np.int32)
            seg = np.array(seg, dtype=np.float32)
            return [seq, seg]
        features = sum([_tokenize(i, piece) for i, piece in enumerate(pieces)], [])
        return features
    
    def _format(features):
        seqs = features[::2]
        segs = features[1::2]
        features = {}
        for i, (seq, seg) in enumerate(zip(seqs, segs)):
            features[i] = {'seq': seq, 'seg': seg}
        return features, tf.zeros([]) # (features, labels)

    dataset = dataset.map(
        lambda text: _format(tf.py_function(_featurize, [text], [tf.int32, tf.float32]*num_pieces)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(
        lambda features, labels: tf.reduce_all(tf.stack([tf.greater(tf.shape(features[idx]['seq'])[0],2) for idx in features])))
    
    padded_shapes = {}
    for i in range(num_pieces):
        padded_shapes[i] = {'seq': [None], 'seg': [None]}
    padded_shapes = (padded_shapes, [])
    dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


""" lm components """

def training_schedule(
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
        elif schedule == 'lr_finder':
            """lr range test"""
            x = (tf.cast(global_step, tf.float32) % num_steps) / num_steps
            log_lr = -7.0 + x*(1.0 - (-7.0))
            learning_rate = tf.pow(10.0, log_lr)
            momentum = 0.9
        else:
            learning_rate = None
            momentum = None

    return learning_rate, momentum

def get_embeddings(
    char_vocab_size,
    char_vocab_dim,
    char_vocab_emb,
    layer_size,
    training,
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

        if type(char_vocab_emb) == np.ndarray:
            char_vocab_initializer = tf.initializers.constant(char_vocab_emb)
        else:
            char_vocab_initializer = tf.initializers.variance_scaling(mode='fan_out')
        char_embedding = tf.get_variable(
            "char_embedding",
            shape=[char_vocab_size, char_vocab_dim],
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
            layer_size,
            is_training=training,
            scope="spellin")
        spellin_embedding = model_utils_py3.layer_norm(
            spellin_embedding, begin_norm_axis=-1, is_training=training)

        spellout_embedding = model_utils_py3.fully_connected(
            spellin_embedding,
            layer_size,
            is_training=training,
            scope="spellout")

        field_embedding = tf.get_variable(
            "field_embedding",
            shape=[10, layer_size],
            dtype=tf.float32,
            initializer=tf.initializers.variance_scaling(mode='fan_out'),
            trainable=training,
            collections=collections,
            aggregation=tf.VariableAggregation.MEAN)

    return input_embedding, spellin_embedding, spellout_embedding, field_embedding

def segment_words(
    seqs,
    segs,
    max_char_length,
    reuse=None):
    """
    segment seqs according to segs
    args:
        seqs: batch_size x seq_length
        segs: batch_size x (seq_length + 1)
        max_char_length: maximum char length allowed
    """
    with tf.variable_scope("segmentor", reuse=reuse):

        batch_size = tf.shape(seqs)[0]
        length = tf.shape(seqs)[1]

        segmented_seqs_ref, segment_idxs_ref = model_utils_py3.slice_words(
            seqs, segs[:,1:-1], get_idxs=True)
        segmented_seqs_ref = tf.stop_gradient(segmented_seqs_ref)
        segmented_seqs_ref = tf.cond(
            tf.less(tf.shape(segmented_seqs_ref)[2], max_char_length),
            lambda: segmented_seqs_ref,
            lambda: segmented_seqs_ref[:,:,:max_char_length])

    return segmented_seqs_ref

def embed_words(
    segmented_seqs,
    input_embedding,
    layer_size,
    dropout,
    training,
    reuse=None):
    """
    embed seq of words into vectors
    args:
        segmented_seqs: batch_size x word_length x char_length
        input_embedding: num_embedding x embedding_dim
        layer_size: size of layer
        dropout: dropout ratio
        training: bool
    """
    with tf.variable_scope("embedder", reuse=reuse):

        batch_size = tf.shape(segmented_seqs)[0]
        max_word_length = tf.shape(segmented_seqs)[1]
        max_char_length = tf.shape(segmented_seqs)[2]
        masks = tf.reduce_any(tf.not_equal(segmented_seqs, 0), axis=2)

        char_embeds = tf.nn.embedding_lookup(input_embedding, segmented_seqs)
        l1_embeds = model_utils_py3.convolution2d(
            char_embeds,
            [layer_size]*2,
            [[1,2],[1,3]],
            activation_fn=tf.nn.relu,
            is_training=training,
            scope="l1_convs")
        char_embeds = tf.nn.max_pool(char_embeds, [1,1,2,1], [1,1,2,1], padding='SAME')
        l2_embeds = model_utils_py3.convolution2d(
            char_embeds,
            [layer_size]*2,
            [[1,2],[1,3]],
            activation_fn=tf.nn.relu,
            is_training=training,
            scope="l2_convs")
        concat_embeds = tf.concat(
            [tf.reduce_max(tf.nn.relu(char_embeds), axis=2),
             tf.reduce_max(l1_embeds, axis=2),
             tf.reduce_max(l2_embeds, axis=2)],
            axis=-1)
        word_embeds = model_utils_py3.highway(
            concat_embeds,
            2,
            activation_fn=tf.nn.relu,
            dropout=dropout,
            is_training=training,
            scope="highway")
        word_embeds = model_utils_py3.layer_norm(
            word_embeds, begin_norm_axis=-1, is_training=training)
        word_embeds = model_utils_py3.fully_connected(
            word_embeds,
            layer_size,
            is_training=training,
            scope="projs")
        word_embeds += model_utils_py3.MLP(
            tf.nn.relu(word_embeds),
            2,
            2*layer_size,
            layer_size,
            dropout=dropout,
            is_training=training,
            scope="MLP")
        word_embeds = model_utils_py3.layer_norm(
            word_embeds, begin_norm_axis=-1, is_training=training)

        masksLeft = tf.pad(masks, [[0,0],[1,0]])[:,:-1]
        masksRight = tf.pad(masks, [[0,0],[0,1]])[:,1:]
        word_masks = tf.logical_or(masks, tf.logical_or(masksLeft, masksRight))

    return word_embeds, word_masks

def encode_words(
    field_embeds,
    value_embeds,
    attn_masks,
    num_layers,
    dropout,
    training,
    reuse=None):
    """
    encode seq of words, include embeds and contexts
    args:
        field_embeds: batch_size x seq_length x dim
        value_embeds: batch_size x seq_length x dim
        attn_masks: batch_size x seq_length
        num_layers: num of layers
        dropout: dropout ratio
        training: bool
    """

    with tf.variable_scope("encoder", reuse=reuse):

        encodes = model_utils_py3.transformer(
            field_embeds,
            num_layers,
            values=value_embeds,
            masks=attn_masks,
            dropout=dropout,
            is_training=training,
            scope="transformer")

    return encodes

def match_embeds(
    encodes,
    token_embeds,
    dropout,
    training,
    reuse=None):
    """
    outputs the degree of matchness of the word embeds and contexts
    args:
        encodes: batch_size x dim
        token_embeds: batch_size x num_candidates x dim or num_candidates x dim
        task_embeds: dim
        dropout: dropout ratio
        training: bool
    """

    with tf.variable_scope("matcher", reuse=reuse):

        dim = encodes.get_shape()[-1].value
        encodes = model_utils_py3.fully_connected(
            encodes,
            dim,
            dropout=dropout,
            is_training=training,
            scope="enc_projs")
        token_embeds = model_utils_py3.fully_connected(
            token_embeds,
            dim,
            is_training=training,
            scope="tok_projs")

        if len(token_embeds.get_shape()) == 2:
            logits = tf.matmul(
                encodes, token_embeds, transpose_b=True)
        else:
            logits = tf.matmul(
                token_embeds, tf.expand_dims(encodes, axis=-1))
            logits = tf.squeeze(logits, axis=-1)
        logits /= tf.sqrt(float(dim))

    return logits

def train_speller(
    encodes,
    encMasks,
    targetSeqs,
    layer_size,
    spellin_embedding,
    spellout_embedding,
    dropout,
    training,
    reuse=None):
    """
    get the training loss of speller
    args:
        encodes: batch_size x enc_seq_length x dim
        encMasks: batch_size x enc_seq_length
        targetSeqs: batch_size x dec_seq_length
        layer_size: size of the layer
        spellin_embedding: input embedding for spelling
        spellout_embedding: output embedding for spelling
        dropout: dropout ratio
        training: bool
    """

    with tf.variable_scope("speller", reuse=reuse):

        attn_cell = model_utils_py3.AttentionCell(
            layer_size,
            num_layer=2,
            dropout=dropout,
            is_training=training)

        batch_size = tf.shape(encodes)[0]
        vocab_size = spellin_embedding.get_shape()[0].value
        dim = spellout_embedding.get_shape()[-1].value

        decInputs = tf.TensorArray(tf.float32, 0,
            dynamic_size=True, clear_after_read=False, infer_shape=False)
        initialState = (decInputs, encodes, encMasks)
        inputs = tf.nn.embedding_lookup(
            spellin_embedding,
            tf.pad(targetSeqs, [[0,0],[1,0]]))
        targetIds = tf.pad(targetSeqs, [[0,0],[0,1]])
        decMasks = tf.not_equal(tf.pad(targetSeqs, [[0,0],[0,1]]), 0)
        decMasks = tf.logical_or(decMasks, tf.pad(tf.not_equal(targetIds, 0), [[0,0],[1,0]])[:,:-1])
        decLength = tf.shape(inputs)[1]
        outputs, state = attn_cell(inputs, initialState)
        state[0].mark_used()
        logits = tf.matmul(
            tf.reshape(outputs, [batch_size*decLength, outputs.get_shape()[-1].value]),
            spellout_embedding,
            transpose_b=True) / tf.sqrt(float(dim))
        logits = tf.reshape(logits, [batch_size, decLength, vocab_size])
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targetIds,
            logits=logits) * tf.cast(decMasks, tf.float32)
        loss = tf.reduce_sum(losses) / tf.reduce_sum(tf.cast(decMasks, tf.float32))

    return loss


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
        data_spec: [{'type': 'seq'|'class', 'is_target': bool, 'select_from': None|np.array}]
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
        valid_word_embeds_ref = tf.boolean_mask(word_embeds_ref, word_masks_ref)
        pick_word_encodes_ref = tf.boolean_mask(masked_word_encodes_ref, pick_masks_ref)
        pick_segmented_seqs_ref = tf.boolean_mask(segmented_seqs_ref, pick_masks_ref)
        num_pick_words = tf.shape(pick_word_encodes_ref)[0]

        unique_segmented_seqs_ref, unique_idxs = model_utils_py3.unique_2d(valid_segmented_seqs_ref)
        unique_word_embeds_ref = tf.gather(valid_word_embeds_ref, unique_idxs)

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
            params.get('schedule'), global_step, params.get('max_lr'), params.get('num_steps'), params.get('pct_start'))
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
#         scaffold = tf.train.Scaffold(init_op=init_op1, init_fn=init_fn)
        scaffold = tf.train.Scaffold(init_op=init_op2)
        fetches = {'global_step': global_step, 'learning_rate': learning_rate, 'loss': loss}
        inside_hook = InsideHook(fetches, params.get('history'))
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, train_op=train_op, training_hooks=[inside_hook], scaffold=scaffold)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'features': features,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


"""high level api"""

def lr_range_test(
    lm_model_fn,
    train_input_fn,
    params,
    num_steps=1000):
    """
    train the model and evaluate every eval_every steps
    """
    
#     gpu_id = '2'
#     session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(visible_device_list=gpu_id))
#     strategy = tf.distribute.MirroredStrategy()
    config=tf.estimator.RunConfig(
#         train_distribute=strategy,
#         eval_distribute=strategy,
#         session_config=session_config,
        save_checkpoints_steps=None,
        save_checkpoints_secs=None,
        log_step_count_steps=10)
    local_params = {}
    local_params.update(params)
    local_params['schedule'] = 'lr_finder'
    local_params['num_steps'] = num_steps
    history = []
    local_params['history'] = history
    lm = tf.estimator.Estimator(
        model_fn=lm_model_fn,
        model_dir='',
        params=local_params,
        config=config)
    history = lm.params['history']
    # start lr range test
    try:
        lm.train(
            input_fn=train_input_fn,
            steps=num_steps)
    except:
        print(history)
        return history
    finally:
        print(history)
        return history

def train_and_evaluate(
    lm_model_fn,
    train_input_fn,
    eval_input_fn,
    train_dir,
    params,
    eval_every=10000):
    """
    train the model and evaluate every eval_every steps
    """
    
#     gpu_id = '2'
#     session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(visible_device_list=gpu_id))
    strategy = tf.distribute.MirroredStrategy()
    config=tf.estimator.RunConfig(
        train_distribute=strategy,
        eval_distribute=strategy,
#         session_config=session_config,
        log_step_count_steps=1000)
    local_params = {}
    local_params.update(params)
    local_params['schedule'] = '1cycle'
    lm = tf.estimator.Estimator(
        model_fn=lm_model_fn,
        model_dir=train_dir,
        params=local_params,
        config=config)
    
    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    os.makedirs(train_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(train_dir, 'tensorflow.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    
    # start train and eval loop
    for _ in range(int(params['num_steps'] / eval_every)):
        lm.train(
            input_fn=train_input_fn,
            steps=eval_every)
        lm.evaluate(
            input_fn=eval_input_fn)