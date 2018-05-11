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

from utils import data_utils
import lm_dataset, lm_model

tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_integer("clr_period", 10000, "Period of cyclic learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_integer("vocab_dim", 512, "Size of embedding.")
tf.app.flags.DEFINE_string("data_dir", "./text_corpus", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_string("embedding_files", "./embeddings/glove.txt", "Pretrained embedding files.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_limit", 2000000,
                            "How many steps to train")
tf.app.flags.DEFINE_integer("gpu_id", 0, "Select which gpu to use.")
tf.app.flags.DEFINE_boolean("test", False,
                            "Run a test on the eval set.")
tf.app.flags.DEFINE_boolean("sample", False,
                            "Run a sample using the model.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.


def create_train_graph(session, vocab):

    data_paths = {
        'train': [os.path.join(FLAGS.data_dir, "train"), 'repeat'],
        'valid': [os.path.join(FLAGS.data_dir, "dev"), 'one-shot']}
    dataset = lm_dataset.LM_Dataset(
        session,
        vocab,
        FLAGS.batch_size,
        data_paths)

    """Load pretrained embeddings."""
    word2vecs = []
    w2v_sizes = []
    for path in FLAGS.embedding_files.split(','):
        word2vecs.append(data_utils.FastWord2vec(path))
        w2v_sizes.append(word2vecs[-1].syn0.shape[1])
    embedding_init = np.random.normal(0.0, 0.01, (FLAGS.vocab_size, FLAGS.vocab_dim))
    for idx, word in enumerate(vocab.vocab_list[:vocab.size()]):
        ptr = 0
        for i, word2vec in enumerate(word2vecs):
            try:
                hit = word2vec[word]
                embedding_init[idx, ptr:ptr+w2v_sizes[i]] = hit
            except:
                pass
            finally:
                ptr += w2v_sizes[i]

    """Create language model and initialize or load parameters in session."""
    model = lm_model.LM_Model(
        session,
        FLAGS.train_dir,
        dataset.next_batch,
        True,
        FLAGS.vocab_size, FLAGS.vocab_dim,
        FLAGS.size, FLAGS.num_layers,
        embedding_init=embedding_init,
        dropout=0.2,
        learning_rate=FLAGS.learning_rate, clr_period=FLAGS.clr_period)
    return dataset, model

def create_infer_graph(session, vocab):

    seqs_placeholder = tf.placeholder(tf.int32, shape=[None, None])

    """Create language model and initialize or load parameters in session."""
    model = lm_model.LM_Model(
        session,
        FLAGS.train_dir,
        seqs_placeholder,
        False,
        FLAGS.vocab_size, FLAGS.vocab_dim,
        FLAGS.size, FLAGS.num_layers)
    return seqs_placeholder, model

def train():

    """Train a language model."""
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Read data into buckets and compute their sizes.
        vocab = data_utils.Vocab(os.path.join(FLAGS.data_dir, "vocab"))
        FLAGS.vocab_size = vocab.size()

        # Create model.
        with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
            dataset, model = create_train_graph(sess, vocab)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        eval_loss = 999999999999.0
        step_loss = 0.0
        while True:
            start_time = time.time()
            input_feed = {dataset.handle: dataset.handles['train']}
            step_loss = model.train_one_step(sess, input_feed)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                print ("global step %d learning rate %.8f step-time %.4f loss "
                       "%.4f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, loss))
                # Run evals on development set and print their perplexity.
                eval_losses = []
                input_feed = {dataset.handle:dataset.handles['valid']}
                try:
                    while True:
                        eval_losses.append(model.valid_one_step(sess, input_feed))
                except:
                    dataset.reset(sess, 'valid')
                eval_loss = sum(eval_losses) / len(eval_losses)
                print("  eval loss %.4f" % eval_loss)
                previous_losses.append(eval_loss)
                sys.stdout.flush()
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "lm.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
            if model.global_step.eval() >= FLAGS.steps_limit:
                break

def sample():

    """Sample a language model."""
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Read data into buckets and compute their sizes.
        vocab = data_utils.Vocab(os.path.join(FLAGS.data_dir, "vocab"))
        FLAGS.vocab_size = vocab.size()

        # Create model.
        with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
            seqs_placeholder, model = create_infer_graph(sess, vocab)

        sample_len = 20
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        sample_seqs = np.array([[0]*sample_len])
        while sentence:
            sentence = sentence.strip()
            if sentence != '':
                sample_seqs = vocab.sentence_to_token_ids(sentence)
                sample_seqs = sample_seqs[:sample_len]
                sample_seqs += [0]*(sample_len-len(sample_seqs))
                sample_seqs = np.array([sample_seqs])
            # This is the training loop.
            input_feed = {seqs_placeholder.name: sample_seqs}
            sample_seqs = model.sample_one_step(sess, input_feed)
            token_ids = list(sample_seqs[0])
            text = vocab.token_ids_to_sentence(token_ids)
            print(text)
            print("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def test():

    """Train a language model."""
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Read data into buckets and compute their sizes.
        vocab = data_utils.Vocab(os.path.join(FLAGS.data_dir, "vocab"))
        FLAGS.vocab_size = vocab.size()

        # Create model.
        with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
            dataset, model = create_train_graph(sess, vocab)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            next_batch = dataset.get_batch(sess, 'train')
            print(next_batch)
            print("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def main(_):
    if FLAGS.test:
        test()
    elif FLAGS.sample:
        sample()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
