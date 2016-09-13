
import tensorflow as tf

### Building Blocks ###

# fully_connected layer
def fully_connected(inputs,
                    num_outputs, 
                    decay=0.99999, 
                    activation_fn=None, 
                    is_training=True, 
                    reuse=None, 
                    scope=None):
  """Adds a fully connected layer.

  """
  if not isinstance(num_outputs, int):
    raise ValueError('num_outputs should be integer, got %s.', num_outputs)
  with tf.variable_op_scope([inputs], 
                            scope, 
                            'fully_connected', 
                            reuse=reuse) as sc:
    dtype = inputs.dtype.base_dtype
    num_input_units = tf.contrib.layers.utils.last_dimension(inputs.get_shape(), min_rank=2)

    static_shape = inputs.get_shape().as_list()
    static_shape[-1] = num_outputs

    out_shape = tf.unpack(tf.shape(inputs))
    out_shape[-1] = num_outputs

    weights_shape = [num_input_units, num_outputs]
    weights = tf.contrib.framework.model_variable('weights', 
                                                  shape=weights_shape, 
                                                  dtype=dtype, 
                                                  initializer=tf.contrib.layers.xavier_initializer(), 
                                                  collections=tf.GraphKeys.WEIGHTS, 
                                                  trainable=True)
    biases = tf.contrib.framework.model_variable('biases', 
                                                 shape=[num_outputs,], 
                                                 dtype=dtype, 
                                                 initializer=tf.zeros_initializer, 
                                                 collections=tf.GraphKeys.BIASES, 
                                                 trainable=True)
    if len(static_shape) > 2:
      # Reshape inputs
      inputs = tf.reshape(inputs, [-1, num_input_units])
    outputs = tf.matmul(inputs, weights)
    moving_mean = tf.contrib.framework.model_variable('moving_mean', 
                                                      shape=[num_outputs,], 
                                                      dtype=dtype, 
                                                      initializer=tf.zeros_initializer, 
                                                      trainable=False)
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, _ = tf.nn.moments(outputs, [0], shift=moving_mean)
      # Update the moving_mean moments.
      update_moving_mean = tf.assign_sub(moving_mean, (moving_mean - mean) * (1.0 - decay))
      #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
      outputs = outputs + biases
    else:
      outputs = outputs + biases
    if activation_fn:
      outputs = activation_fn(outputs)
    if len(static_shape) > 2:
      # Reshape back outputs
      outputs = tf.reshape(outputs, tf.pack(out_shape))
      outputs.set_shape(static_shape)
    return outputs

# convolutional layer
def convolution2d(inputs, 
                  num_outputs, 
                  kernel_size, 
                  pool_size=None, 
                  decay=0.99999, 
                  activation_fn=None, 
                  is_training=True, 
                  reuse=None, 
                  scope=None):
  """Adds a 2D convolution followed by a maxpool layer.

  """
  with tf.variable_op_scope([inputs], 
                            scope, 
                            'Conv', 
                            reuse=reuse) as sc:
    dtype = inputs.dtype.base_dtype
    num_filters_in = tf.contrib.layers.utils.last_dimension(inputs.get_shape(), min_rank=4)
    weights_shape = list(kernel_size) + [num_filters_in, num_outputs]
    weights = tf.contrib.framework.model_variable('weights', 
                                                  shape=weights_shape, 
                                                  dtype=dtype, 
                                                  initializer=tf.contrib.layers.xavier_initializer(),
                                                  collections=tf.GraphKeys.WEIGHTS,
                                                  trainable=True)
    biases = tf.contrib.framework.model_variable('biases',
                                                 shape=[num_outputs,],
                                                 dtype=dtype,
                                                 initializer=tf.zeros_initializer,
                                                 collections=tf.GraphKeys.BIASES,
                                                 trainable=True)
    outputs = tf.nn.conv2d(inputs, weights, [1,1,1,1], padding='SAME')
    moving_mean = tf.contrib.framework.model_variable('moving_mean',
                                                      shape=[num_outputs,],
                                                      dtype=dtype,
                                                      initializer=tf.zeros_initializer,
                                                      trainable=False)
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, _ = tf.nn.moments(outputs, [0, 1, 2], shift=moving_mean)
      # Update the moving_mean moments.
      update_moving_mean = tf.assign_sub(moving_mean, (moving_mean - mean) * (1.0 - decay))
      #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
      outputs = outputs + biases
    else:
      outputs = outputs + biases
    if pool_size:
      pool_shape = [1] + list(pool_size) + [1]
      outputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')
    if activation_fn:
      outputs = activation_fn(outputs)
    return outputs


### Regularization ###

def params_decay(decay):
  """ Add ops to decay weights and biases

  """
  params = tf.get_collection_ref(tf.GraphKeys.WEIGHTS) + tf.get_collection_ref(tf.GraphKeys.BIASES)
  while len(params) > 0:
    p = params.pop()
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, 
        p.assign(decay*p + (1-decay)*tf.truncated_normal(p.get_shape(), stddev=0.01)))


### Nets ###

# dnn
def dnn(inputs, 
        hid_size, 
        out_size, 
        num_layers, 
        decay=0.99999, 
        activation_fn=tf.nn.relu, 
        is_training=True, 
        reuse=None, 
        scope=None):
  """ a deep neural net with fully connected layers

  """
  with tf.variable_op_scope([inputs], scope, "dnn", reuse=reuse):
    for i in xrange(num_layers-1):
      # hid layers
      with tf.variable_scope("layer{0}".format(i)):
        inputs = fully_connected(inputs, hid_size, decay=decay, activation_fn=activation_fn, is_training=is_training)
    # output layer
    with tf.variable_scope("layer{0}".format(num_layers-1)):
      outputs = fully_connected(inputs, out_size, decay=decay, is_training=is_training)
  return outputs

# cnn
def cnn(inputs, 
        hid_size, 
        out_size, 
        num_layers, 
        kernel_size, 
        pool_size, 
        decay=0.99999, 
        activation_fn=tf.nn.relu, 
        is_training=True, 
        reuse=None, 
        scope=None):
  """ a convolutaional neural net with conv2d and max_pool layers

  """
  with tf.variable_op_scope([inputs], scope, "cnn", reuse=reuse):
    for i in xrange(num_layers):
      with tf.variable_scope("layer{0}".format(i)):
        inputs = convolution2d(activation_fn(inputs), hid_size, kernel_size, decay=decay, activation_fn=None, 
            is_training=is_training) + inputs
        if i != num_layers-1 and pool_size:
          pool_shape = [1] + list(pool_size) + [1]
          inputs = tf.nn.max_pool(inputs, pool_shape, pool_shape, padding='SAME')
    return inputs


### RNN ###
class newGRUCell(tf.nn.rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tf.nn.relu, linear=fully_connected):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated." % self)
    self._num_units = num_units
    self._activation = activation
    self._linear = linear

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("InputGates"):
        u = self._linear(state, self._num_units)
        u = tf.sigmoid(u)
      with tf.variable_scope("ForgetGatesAndUpdates"):
        x = self._linear(inputs, 2 * self._num_units)
        c, f = tf.split(1, 2, x)
        c = self._activation(c)
        f = tf.sigmoid(f)
      new_h = f * state + u * c
    return new_h, new_h

def create_cell(size, num_layers, cell_type="GRU", decay=0.99999, is_training=True):
  # fully connected layers inside the rnn cell
  def _linear(inputs, num_outputs):
    return fully_connected(inputs, num_outputs, decay=decay, is_training=is_training) 
 
  # build single cell
  if cell_type == "GRU":
    single_cell = newGRUCell(size, activation=tf.nn.relu, linear=_linear)
  elif cell_type == "LSTM":
    single_cell = tf.nn.rnn_cell.LSTMCell(size, use_peepholes=True, cell_clip=5.0, num_proj=size)
  else:
    raise ValueError('Incorrect cell type! (GRU|LSTM)')
  cell = single_cell
  # stack multiple cells
  if num_layers > 1:
    cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)
  return cell


### Recurrent Decoders ###

# greedy decoder
def greedy_dec(length, 
               initial_state, 
               context, 
               iter_fn, 
               embedding, 
               beam_size=1, 
               topn=1, 
               eos_id=2):
  """ A greedy decoder.

  """
  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = embedding.get_shape()[1].value
  inputs = tf.zeros([batch_size, inputs_size])

  outputs, state = iter_fn(inputs, initial_state, context)
  logits = tf.matmul(outputs, tf.transpose(embedding))

  symbol = tf.argmax(logits, 1)
  seq = [symbol]
  tf.get_variable_scope().reuse_variables()
  for _ in xrange(length-1):

    inputs = tf.nn.relu(tf.nn.embedding_lookup(embedding, symbol))

    outputs, state = iter_fn(inputs, state, context)
    logits = tf.matmul(outputs, tf.transpose(embedding))

    symbol = tf.argmax(logits, 1)
    seq.append(symbol)

  return tf.pack(seq, 1)

# stochastic decoder
def stochastic_dec(length,
                   initial_state,
                   context,
                   iter_fn,
                   embedding,
                   beam_size=1,
                   topn=1,
                   eos_id=2):
  """ A stochastic decoder.

  """
  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = embedding.get_shape()[1].value
  inputs = tf.zeros([batch_size, inputs_size])

  outputs, state = iter_fn(inputs, initial_state, context)
  logits = tf.matmul(outputs, tf.transpose(embedding))

  symbol = tf.squeeze(tf.multinomial(logits, 1), [1])
  seq = [symbol]
  tf.get_variable_scope().reuse_variables()
  for _ in xrange(length-1):

    inputs = tf.nn.relu(tf.nn.embedding_lookup(embedding, symbol))

    outputs, state = iter_fn(inputs, state, context)
    logits = tf.matmul(outputs, tf.transpose(embedding))

    symbol = tf.squeeze(tf.multinomial(logits, 1), [1])
    seq.append(symbol)

  return tf.pack(seq, 1)

# beam decoder
def beam_dec(length, 
             initial_state, 
             context, 
             iter_fn, 
             embedding, 
             beam_size=100, 
             topn=10, 
             eos_id=2):
  """ A basic beam decoder

  """

  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = embedding.get_shape()[1].value
  inputs = tf.zeros([batch_size, inputs_size])
  vocab_size = tf.shape(embedding)[0]

  # iter
  outputs, state = iter_fn(inputs, initial_state, context)
  logits = tf.matmul(outputs, tf.transpose(embedding))

  prev = tf.nn.log_softmax(logits)
  probs = tf.reshape(prev, [-1])
  best_probs, indices = tf.nn.top_k(probs, beam_size)

  symbols = indices % vocab_size
  beam_parent = indices // vocab_size
  paths = tf.reshape(symbols, [-1, 1])

  tf.get_variable_scope().reuse_variables()
  for _ in xrange(length-1):

    if isinstance(state, tuple):
      state = tuple([tf.gather(s, beam_parent) for s in state])
    else:
      state = tf.gather(state, beam_parent)
    inputs = tf.nn.relu(tf.nn.embedding_lookup(embedding, symbols))

    # iter
    outputs, state = iter_fn(inputs, state, tf.tile(context, [tf.shape(inputs)[0], 1]))
    logits = tf.matmul(outputs, tf.transpose(embedding))

    prev = tf.nn.log_softmax(logits)
    probs = tf.reshape(prev + tf.reshape(best_probs, [-1, 1]), [-1])
    best_probs, indices = tf.nn.top_k(probs, beam_size)

    symbols = indices % vocab_size
    beam_parent = indices // vocab_size
    paths = tf.gather(paths, beam_parent)
    paths = tf.concat(1, [paths, tf.reshape(symbols, [-1, 1])])

  return tf.unpack(paths)[:topn]

# another beam decoder, likely better than the other one
def beam_dec_v2(length, 
                initial_state, 
                context, 
                iter_fn, 
                embedding, 
                beam_size=100, 
                topn=10, 
                eos_id=2):
  """ A beam decoder. We keep monitoring the beam to add finished entries to the top lists

  """

  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = embedding.get_shape()[1].value
  inputs = tf.zeros([batch_size, inputs_size])
  vocab_size = tf.shape(embedding)[0]

  # iter
  outputs, state = iter_fn(inputs, initial_state, context)
  logits = tf.matmul(outputs, tf.transpose(embedding))

  # pruning with beam
  prev = tf.nn.log_softmax(logits)
  probs = tf.reshape(prev, [-1])
  best_probs, indices = tf.nn.top_k(probs, beam_size)

  # throw the candidate if eos appear at the beginning
  symbols = indices % vocab_size
  mask = tf.not_equal(symbols, eos_id)
  best_probs = tf.boolean_mask(best_probs, mask)
  indices = tf.boolean_mask(indices, mask)
  symbols = indices % vocab_size
  beam_parent = indices // vocab_size
  paths = tf.reshape(symbols, [-1, 1])

  tf.get_variable_scope().reuse_variables()
  nbest_path = []
  nbest_score = []
  for i in xrange(length-1):

    # gather the state and inputs
    if isinstance(state, tuple):
      state = tuple([tf.gather(s, beam_parent) for s in state])
    else:
      state = tf.gather(state, beam_parent)
    inputs = tf.nn.relu(tf.nn.embedding_lookup(embedding, symbols))

    # iter
    if isinstance(context, tuple):
      context_prime = []
      for c in context:
        cdim = [d.value for d in c.get_shape()]
        context_prime.append(tf.reshape(tf.tile(c, [tf.shape(inputs)[0]] + [1]*(len(c.get_shape())-1)), 
            [-1]+cdim[1:]))
      context_prime = tuple(context_prime)
    else:
      cdim = [d.value for d in context.get_shape()]
      context_prime = tf.reshape(tf.tile(context, [tf.shape(inputs)[0]] + [1]*(len(context.get_shape())-1)), 
          [-1]+cdim[1:])
    outputs, state = iter_fn(inputs, state, context_prime)
    logits = tf.matmul(outputs, tf.transpose(embedding))

    # prune
    prev = tf.nn.log_softmax(logits)
    probs = tf.reshape(prev + tf.reshape(best_probs, [-1, 1]), [-1])
    best_probs, indices = tf.nn.top_k(probs, beam_size)

    symbols = indices % vocab_size
    beam_parent = indices // vocab_size
    paths = tf.gather(paths, beam_parent)
    paths = tf.concat(1, [paths, tf.reshape(symbols, [-1, 1])])

    # add the finished candidates to the nbest list
    ended = tf.equal(symbols, eos_id)
    nbest_path.append(tf.pad(tf.boolean_mask(paths, ended), [[0,0], [0,length-2-i]]))
    nbest_score.append(tf.boolean_mask(best_probs, ended))

    # select the remaining entries in the beam
    mask = tf.not_equal(symbols, eos_id)
    best_probs = tf.boolean_mask(best_probs, mask)
    symbols = tf.boolean_mask(symbols, mask)
    beam_parent = tf.boolean_mask(beam_parent, mask)
    paths = tf.boolean_mask(paths, mask)

  # add the remaining entries to the nbest list
  nbest_path.append(paths)
  nbest_score.append(best_probs)

  nbest_path = tf.concat(0, nbest_path)
  nbest_score = tf.concat(0, nbest_score)
  _, indices = tf.nn.top_k(nbest_score, beam_size)
  nbest_path = tf.gather(nbest_path, indices)
  return tf.unpack(nbest_path)[:topn]


### Attention on Memory ###

def attention(query, 
              keys, 
              values, 
              is_training=True):
  """ implements the attention mechanism

  query: [batch_size x dim]
  keys: [batch_size x length x dim]
  values: [batch_size x length x dim]
  """
  query = tf.expand_dims(query, 1)
  logits = convolution2d(tf.expand_dims(tf.nn.relu(query+keys), 1), 1, [1, 3], 
      is_training=is_training, scope="attention")
  logits = tf.squeeze(logits, [1, 3])
  results = tf.reduce_sum(tf.expand_dims(tf.nn.softmax(logits), 2) * values, [1])
  return results

def attention_iter(inputs, state, memory, cell, is_training):
  """ implements an attention iter function

  """
  keys, values = memory
  if not values.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % values.get_shape())

  attn_feat = state[0]
  state = state[1:]
  if len(state) == 1:
    state = state[0]
  batch_size = tf.shape(inputs)[0]
  size = inputs.get_shape()[1].value
  mem_size = values.get_shape()[2].value

  with tf.variable_scope("attention_decoder"):

    inputs = tf.concat(1, [inputs, attn_feat])
    cell_outputs, state = cell(inputs, state)

    with tf.variable_scope("query"):
      state = tf.concat(1, list(state)) if isinstance(state, (tuple, list)) else state
      query = fully_connected(state, mem_size, activation_fn=None, is_training=is_training)

    with tf.variable_scope("attention"):
      results = attention(query, keys, values, is_training)

    with tf.variable_scope("output_proj"):
      outputs = fully_connected(tf.concat(1, [cell_outputs, results]), size, activation_fn=None, 
          is_training=is_training)

  if isinstance(state, tuple):
    state = (attn_feat,) + state
  else:
    state = (attn_feat, state)

  return outputs, state
