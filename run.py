# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99999,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_string("cell_type", "GRU", "GRU|LSTM")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./text_corpus", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_limit", 200000,
                            "How many steps to train")
tf.app.flags.DEFINE_integer("gpu_id", 0, "Select which gpu to use.")
tf.app.flags.DEFINE_boolean("interactive_test", False,
                            "Set to True for interactive testing.")
tf.app.flags.DEFINE_boolean("test", False,
                            "Run a test on the eval set.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [10, 15, 25, 40]

def read_data(data_path, vocab, max_size=None):
  """Read data from file and put into buckets.

  """
  if not os.path.exists(data_path):
    return None
  if os.path.isdir(data_path+"_tfrecords"):
    data_set = []
    for i in xrange(len(_buckets)):
      data_set.append(os.listdir(os.path.join(data_path+"_tfrecords", "bucket_"+str(i))))
    return data_set
  os.makedirs(data_path+"_tfrecords")
  data_set = []
  writer_set = []
  counter_set = []
  for i in xrange(len(_buckets)):
    bucket_path = os.path.join(data_path+"_tfrecords", "bucket_"+str(i))
    os.makedirs(bucket_path)
    filename = os.path.join(bucket_path, str(0))
    data_set.append([filename])
    writer_set.append(tf.python_io.TFRecordWriter(filename))
    counter_set.append(0)
  
  counter_limit = 10000
  with open(data_path, 'r') as f:
    for line in f:
      # preprocess the sample
      text = line.strip()
      seq = data_utils.sentence_to_token_ids(text, vocab)
      if vocab.key2idx("_UNK") in seq:
        continue
      length = len(seq)
      # insert to one of the buckets
      for bucket_id, bucket_size in enumerate(_buckets):
        if length <= bucket_size:
          seq = np.array(seq + [vocab.key2idx("_PAD")] * (bucket_size-length), dtype=np.int32)
          seq_str = seq.tostring()
          example = tf.train.Example(features=tf.train.Features(feature={
                    'seq': tf.train.Feature(bytes_list=tf.train.BytesList(value=[seq_str]))}))
          writer_set[bucket_id].write(example.SerializeToString())
          counter_set[bucket_id] += 1
          if counter_set[bucket_id] >= counter_limit:
            writer_set[bucket_id].close()
            bucket_path = os.path.join(data_path+"_tfrecords", "bucket_"+str(bucket_id))
            filename = os.path.join(bucket_path, str(len(data_set[bucket_id])))
            data_set[bucket_id].append(filename)
            writer_set[bucket_id] = tf.python_io.TFRecordWriter(filename)
            counter_set[bucket_id] = 0
          break

  return True


def create_model(session, is_decoding):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.vocab_size, _buckets, FLAGS.size, FLAGS.num_layers, 
      FLAGS.max_gradient, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      FLAGS.data_dir, FLAGS.cell_type, is_decoding=is_decoding)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  """Train a text classifier."""

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # Read data into buckets and compute their sizes.
    train_path = os.path.join(FLAGS.data_dir, "train")
    dev_path = os.path.join(FLAGS.data_dir, "dev")
    vocab = data_utils.Vocab(os.path.join(FLAGS.data_dir, "vocab"))
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(dev_path, vocab)
    if not dev_set:
      print("Error reading dev data!")
      sys.exit()
    train_set = read_data(train_path, vocab)
    if not train_set:
      print("Error reading train data!")
      sys.exit()

    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
      model = create_model(sess, False)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    eval_loss = 999999999999.0
    step_loss = 0.0
    bucket_id = 0
    try:
      while not coord.should_stop():
        start_time = time.time()
        step_loss = model.step(sess, bucket_id, update=True)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
          # Print statistics for the previous epoch.
          print ("global step %d learning rate %.4f step-time %.4f loss "
                 "%.4f" % (model.global_step.eval(), model.learning_rate.eval(),
                           step_time, loss))
          # Run evals on development set and print their perplexity.
          eval_losses = []
          for _ in xrange(1000):
            eval_losses.append(model.step(sess, bucket_id, update=False))
          eval_loss = sum(eval_losses) / len(eval_losses)
          print("  eval loss %.4f" % eval_loss)
          previous_losses.append(eval_loss)
          sys.stdout.flush()
          # Save checkpoint and zero timer and loss.
          checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          step_time, loss = 0.0, 0.0
        if model.global_step.eval() >= FLAGS.steps_limit:
          break
    except tf.errors.OutOfRangeError:
      print('Done training -- epoch limit reached')
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()


def interactive_test():
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
      # Create model and load parameters.
      FLAGS.batch_size = 1
      model = create_model(sess, True)

    # Load vocabularies.
    vocab = data_utils.Vocab(os.path.join(FLAGS.data_dir, "vocab"))

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      bucket_id = 0
      # Get output logits for the sentence.
      outputs = model.step(sess, bucket_id)
      # Print output text
      for seq in outputs:
        sent = [int(i) for i in list(seq)]
        if vocab.key2idx("_PAD") in sent:
          sent = sent[:sent.index(vocab.key2idx("_PAD"))]
        print("".join([tf.compat.as_str(vocab.idx2key(w)) for w in sent]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def test():
  """Test the text classification model."""
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    print("test for text classification model.")
    with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
      model = create_model(sess, True)
      #model.batch_size = 1

    # Read test set data
    eval_path = os.path.join(FLAGS.data_dir, "eval")
    vocab = data_utils.Vocab(os.path.join(FLAGS.data_dir, "vocab"))
    slots = data_utils.Vocab(os.path.join(FLAGS.data_dir, "slots"))
    print ("Reading eval data")
    eval_set = read_data(eval_path, vocab, slots)

    total = 0
    error = 0
    # Loop through all the test sample
    with open(os.path.join(FLAGS.data_dir, "test_error_samples"), 'w') as f:
      for bucket_id in xrange(len(eval_set)):
        while True:
          encoder_inputs, slotids, targets = model.get_batch(
              eval_set[bucket_id], put_back=False)
          if targets != None:
            outputs = model.step(sess, encoder_inputs, slotids, targets, bucket_id)
            encoder_inputs = list(np.transpose(np.array(encoder_inputs)))
            targets = list(np.transpose(np.array(targets[1:])))
            outputs = list(outputs)
            slotids = list(slotids)
            for idx in xrange(len(outputs)):
              ref = [int(i) for i in list(encoder_inputs[idx])]
              hyp = [int(i) for i in list(outputs[idx])]
              grd = [int(i) for i in list(targets[idx])]
              total += 1
              if hyp != grd:
                error += 1
                ref = data_utils.token_ids_to_sentence(ref, vocab)
                slotname = slots.idx2key(int(slotids[idx]))
                hyp = data_utils.token_ids_to_sentence(hyp, vocab)
                grd = data_utils.token_ids_to_sentence(grd, vocab)
                f.write(ref+"\t"+slotname+"\toutputs: "+hyp+"\ttargets: "+grd+"\n")
          else:
            break

    print ("Test set error rate: " + str(float(error) / float(total) * 100.0) + "%")

def main(_):
  if FLAGS.test:
    test()
  elif FLAGS.interactive_test:
    interactive_test()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
