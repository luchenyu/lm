
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

# ResDNN
def ResDNN(inputs, 
           out_size, 
           num_layers, 
           decay=0.99999, 
           activation_fn=tf.nn.relu, 
           is_training=True, 
           reuse=None, 
           scope=None):
  """ a deep neural net with fully connected layers

  """
  with tf.variable_op_scope([inputs], scope, "ResDNN", reuse=reuse):
    # first layer
    with tf.variable_scope("layer{0}".format(0)):
      outputs = fully_connected(inputs, out_size, decay=decay, activation_fn=None, is_training=is_training)
    # residual layers
    for i in xrange(num_layers-1):
      inputs = activation_fn(outputs)
      with tf.variable_scope("layer{0}".format(i+1)):
        outputs += fully_connected(inputs, out_size, decay=decay, activation_fn=None, is_training=is_training)
  return outputs

# ResCNN
def ResCNN(inputs, 
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
  with tf.variable_op_scope([inputs], scope, "ResCNN", reuse=reuse):
    # first layer
    with tf.variable_scope("layer{0}".format(0)):
      outputs = convolution2d(inputs, out_size, kernel_size, decay=decay, activation_fn=None, 
          is_training=is_training)
    # residual layers
    for i in xrange(num_layers-1):
      with tf.variable_scope("layer{0}".format(i+1)):
        if pool_size:
          pool_shape = [1] + list(pool_size) + [1]
          outputs = tf.nn.max_pool(outputs, pool_shape, pool_shape, padding='SAME')
        inputs = activation_fn(outputs)
        outputs += convolution2d(inputs, out_size, kernel_size, decay=decay, activation_fn=None, 
            is_training=is_training)
    return outputs


### RNN ###
class GRUCell(tf.nn.rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tf.tanh, linear=fully_connected):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
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
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = tf.split(1, 2, self._linear(tf.concat(1, [inputs, state]),
                                             2 * self._num_units))
        r, u = tf.sigmoid(r), tf.sigmoid(u)
      with tf.variable_scope("Candidate"):
        c = self._activation(self._linear(tf.concat(1, [inputs, r * state]),
                                     self._num_units))
      new_h = u * state + (1 - u) * c
    return new_h, new_h

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

  def __call__(self, inputs, state, scope="newGRUCell"):
    """Gated recurrent unit (GRU) with nunits cells."""
    with tf.variable_op_scope([inputs, state], scope, None, reuse=None):  # "GRUCell"
      with tf.variable_scope("InputGates"):
        i = self._linear(state, self._num_units)
        i = tf.sigmoid(i)
      with tf.variable_scope("ForgetGatesAndUpdates"):
        x = self._linear(inputs, 2 * self._num_units)
        c, f = tf.split(1, 2, x)
        c = self._activation(c)
        f = tf.sigmoid(f)
        update = state + i * c
        new_h = f * update
    return update - new_h, new_h

def create_cell(size, num_layers, cell_type="GRU", decay=0.99999, is_training=True):
  # fully connected layers inside the rnn cell
  def _linear(inputs, num_outputs):
    return fully_connected(inputs, num_outputs, decay=decay, is_training=is_training) 
 
  # build single cell
  if cell_type == "GRU":
    single_cell = newGRUCell(size, activation=tf.tanh, linear=_linear)
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
               memory, 
               iter_fn, 
               embedding, 
               beam_size=1, 
               num_candidates=1, 
               eos_id=2):
  """ A greedy decoder.

  """
  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = embedding.get_shape()[1].value
  inputs = tf.zeros([batch_size, inputs_size])

  outputs, state = iter_fn(inputs, initial_state, memory)
  logits = tf.matmul(outputs, tf.transpose(embedding))

  symbol = tf.argmax(logits, 1)
  seq = [symbol]
  tf.get_variable_scope().reuse_variables()
  for _ in xrange(length-1):

    inputs = tf.nn.relu(tf.nn.embedding_lookup(embedding, symbol))

    outputs, state = iter_fn(inputs, state, memory)
    logits = tf.matmul(outputs, tf.transpose(embedding))

    symbol = tf.argmax(logits, 1)
    seq.append(symbol)

  return tf.pack(seq, 1)

def stochastic_dec(length,
                   initial_state,
                   memory,
                   iter_fn,
                   embedding,
                   beam_size=1,
                   num_candidates=1,
                   eos_id=2):
  """ A stochastic decoder.

  """
  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = embedding.get_shape()[1].value
  inputs = tf.zeros([batch_size, inputs_size])

  outputs, state = iter_fn(inputs, initial_state, memory)
  logits = tf.matmul(outputs, tf.transpose(embedding))

  symbol = tf.reshape(tf.multinomial(logits, num_candidates), [-1])
  seq = [symbol]
  tf.get_variable_scope().reuse_variables()
  if isinstance(state, tuple):
    state = tuple([tf.reshape(tf.pack([s]*num_candidates, axis=1), 
        [batch_size*num_candidates, s.get_shape()[1].value]) for s in state])
  else:
    state = tf.reshape(tf.pack([state]*num_candidates, axis=1), 
        [batch_size*num_candidates, state.get_shape()[1].value])

  for _ in xrange(length-1):

    inputs = tf.nn.relu(tf.nn.embedding_lookup(embedding, symbol))

    outputs, state = iter_fn(inputs, state, memory)
    logits = tf.matmul(outputs, tf.transpose(embedding))

    symbol = tf.squeeze(tf.multinomial(logits, 1), [1])
    seq.append(symbol)

  return tf.pack(seq, 1)

# beam decoder
def beam_dec(length,
             initial_state,
             memory,
             iter_fn,
             embedding,
             beam_size=100,
             num_candidates=10,
             eos_id=2):
  """ A basic beam decoder

  """

  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = embedding.get_shape()[1].value
  inputs = tf.zeros([batch_size, inputs_size])
  vocab_size = tf.shape(embedding)[0]

  # iter
  outputs, state = iter_fn(inputs, initial_state, memory)
  logits = tf.matmul(outputs, tf.transpose(embedding))

  prev = tf.nn.log_softmax(logits)
  probs = prev
  best_probs, indices = tf.nn.top_k(probs, beam_size)

  symbols = indices % vocab_size
  beam_parent = indices // vocab_size
  beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size), 1) + beam_parent, [-1])
  paths = tf.reshape(symbols, [-1, 1])

  if isinstance(memory, tuple):
    memory_prime = []
    for m in memory:
      mdim = [d.value for d in m.get_shape()]
      m = tf.expand_dims(m, 1)
      memory_prime.append(tf.reshape(tf.tile(m, [1] + [beam_size] + [1]*(len(mdim)-1)),
          [-1]+mdim[1:]))
    memory = tuple(memory_prime)
  else:
    mdim = [d.value for d in memory.get_shape()]
    memory = tf.expand_dims(memory, 1)
    memory = tf.reshape(tf.tile(memory, [1] + [beam_size] + [1]*(len(mdim)-1)),
        [-1]+mdim[1:])

  tf.get_variable_scope().reuse_variables()
  for _ in xrange(length-1):

    if isinstance(state, tuple):
      state = tuple([tf.gather(s, beam_parent) for s in state])
    else:
      state = tf.gather(state, beam_parent)

    inputs = tf.reshape(tf.nn.relu(tf.nn.embedding_lookup(embedding, symbols)), [-1, inputs_size])

    # iter
    outputs, state = iter_fn(inputs, state, memory)
    logits = tf.matmul(outputs, tf.transpose(embedding))

    prev = tf.reshape(tf.nn.log_softmax(logits), [batch_size, beam_size, vocab_size])
    probs = tf.reshape(prev + tf.expand_dims(best_probs, 2), [batch_size, -1])
    best_probs, indices = tf.nn.top_k(probs, beam_size)

    symbols = indices % vocab_size
    beam_parent = indices // vocab_size
    beam_parent = tf.reshape(tf.expand_dims(tf.range(batch_size) * beam_size, 1) + beam_parent, [-1])
    paths = tf.gather(paths, beam_parent)
    paths = tf.concat(1, [paths, tf.reshape(symbols, [-1, 1])])

  return tf.reshape(tf.slice(tf.reshape(paths, [batch_size, beam_size, -1]), [0, 0, 0], [-1, num_candidates, -1]),
      [batch_size*num_candidates, -1])

# another beam decoder, likely better than the other one
def beam_dec_v2(length, 
                initial_state, 
                memory, 
                iter_fn, 
                embedding, 
                beam_size=100, 
                num_candidates=10, 
                eos_id=2):
  """ A beam decoder. We keep monitoring the beam to add finished entries to the top lists

  """

  batch_size = tf.shape(initial_state[0])[0] if isinstance(initial_state, tuple) else tf.shape(initial_state)[0]
  inputs_size = embedding.get_shape()[1].value
  inputs = tf.zeros([batch_size, inputs_size])
  vocab_size = tf.shape(embedding)[0]

  # iter
  outputs, state = iter_fn(inputs, initial_state, memory)
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
    if isinstance(memory, tuple):
      memory_prime = []
      for m in memory:
        mdim = [d.value for d in m.get_shape()]
        memory_prime.append(tf.reshape(tf.tile(m, [tf.shape(inputs)[0]] + [1]*(len(m.get_shape())-1)), 
            [-1]+mdim[1:]))
      memory_prime = tuple(memory_prime)
    else:
      mdim = [d.value for d in memory.get_shape()]
      memory_prime = tf.reshape(tf.tile(memory, [tf.shape(inputs)[0]] + [1]*(len(memory.get_shape())-1)), 
          [-1]+mdim[1:])
    outputs, state = iter_fn(inputs, state, memory_prime)
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
  return tf.slice(nbest_path, [0, 0], [num_candidates, -1])


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
  logits = convolution2d(tf.expand_dims(tf.nn.relu(query+keys), 1), 1, [1, 1], 
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
  cell_state = state[1]
  batch_size = tf.shape(inputs)[0]
  size = inputs.get_shape()[1].value
  mem_size = values.get_shape()[2].value

  with tf.variable_scope("attention_decoder"):

    with tf.variable_scope("dec_cell"):
      cell_outputs, cell_state = cell(inputs, cell_state)

    with tf.variable_scope("query"):
      cell_state_concat = tf.concat(1, list(cell_state)) if isinstance(cell_state, (tuple, list)) else cell_state
      query = fully_connected(cell_state_concat, mem_size, activation_fn=None, is_training=is_training)

    with tf.variable_scope("attention"):
      attn_feat = attention(query, keys, values, is_training)

    with tf.variable_scope("output_proj"):
      outputs = fully_connected(tf.concat(1, [cell_state_concat, attn_feat]), size, 
          activation_fn=None, is_training=is_training)

    state = (attn_feat, cell_state)

  return outputs, state
