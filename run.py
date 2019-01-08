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
np.set_printoptions(threshold=np.nan)
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from utils import data_utils
import lm_dataset, lm_model

tf.app.flags.DEFINE_float("learning_rate", -1, "Learning rate.")
tf.app.flags.DEFINE_integer("clr_period", -1, "Period of cyclic learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_float("dropout", 0.1, "Dropout rate.")
tf.app.flags.DEFINE_integer("size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "vocabulary size.")
tf.app.flags.DEFINE_integer("vocab_dim", 300, "Size of embedding.")
tf.app.flags.DEFINE_string("block_type", "transformer2", "Block type: lstm|transformer")
tf.app.flags.DEFINE_string("decoder_type", "attn", "Decoder type: lstm|attn")
tf.app.flags.DEFINE_string("loss_type", "unsup", "Loss type: sup|unsup")
tf.app.flags.DEFINE_string("model", "ultra_lm", "Model: simple_lm|ultra_lm")
tf.app.flags.DEFINE_boolean("early_stop", True, "Set True to turn on early stop.")
tf.app.flags.DEFINE_boolean("segmented", False, "Set True to read segmented text data.")
tf.app.flags.DEFINE_string("data_dir", "./text_corpus", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_string("embedding_files", "./embeddings/zh_char_300_nlpcc.txt", "Pretrained embedding files.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 20000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("steps_limit", 100000000,
                            "How many steps to train")
tf.app.flags.DEFINE_integer("gpu_id", 0, "Select which gpu to use.")
tf.app.flags.DEFINE_boolean("test", False,
                            "Run a test on the eval set.")
tf.app.flags.DEFINE_boolean("sample", False,
                            "Run a sample using the model.")
tf.app.flags.DEFINE_boolean("posseg", False,
                            "Run a seg using the model.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.


def create_train_graph(session, vocab, posseg_vocab):

    data_paths = {
        'train': [os.path.join(FLAGS.data_dir, "train"), 'repeat'],
        'valid': [os.path.join(FLAGS.data_dir, "dev"), 'one-shot']}
    dataset = lm_dataset.LM_Dataset(
        vocab,
        posseg_vocab,
        FLAGS.batch_size,
        data_paths,
        segmented=FLAGS.segmented)
    dataset.init(session)

    """Create language model and initialize or load parameters in session."""
    seqs, segs, = dataset.next_batch
    kwargs = {'segs': [segs],
              'block_type': FLAGS.block_type,
              'decoder_type': FLAGS.decoder_type,
              'loss_type': FLAGS.loss_type,
              'model': FLAGS.model,
              'embedding_init': vocab.embedding_init,
              'dropout': FLAGS.dropout}
    if FLAGS.learning_rate > 0:
        kwargs['learning_rate'] = FLAGS.learning_rate
    if FLAGS.clr_period > 0:
        kwargs['clr_period'] = FLAGS.clr_period
    model = lm_model.LM_Model(
        [seqs],
        True,
        FLAGS.vocab_size, FLAGS.vocab_dim,
        FLAGS.size, FLAGS.num_layers,
        **kwargs)
    model.init(session, FLAGS.train_dir, kwargs)

    return dataset, model

def create_infer_graph(session, vocab, posseg_vocab):

    seqs_placeholder = tf.placeholder(tf.int32, shape=[None, None])

    """Create language model and initialize or load parameters in session."""
    model = lm_model.LM_Model(
        [seqs_placeholder],
        False,
        FLAGS.vocab_size, FLAGS.vocab_dim,
        FLAGS.size, FLAGS.num_layers,
        block_type=FLAGS.block_type,
        decoder_type=FLAGS.decoder_type,
        loss_type=FLAGS.loss_type,
        model=FLAGS.model)
    model.init(session, FLAGS.train_dir)

    return seqs_placeholder, model

def train():

    """Train a language model."""
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Read data into buckets and compute their sizes.
        vocab = data_utils.Vocab(
            os.path.join(FLAGS.data_dir, "vocab"),
            embedding_files=FLAGS.embedding_files)
        FLAGS.vocab_size = vocab.size()
        if FLAGS.embedding_files != "":
            FLAGS.vocab_dim = vocab.embedding_init.shape[1]
        posseg_vocab = data_utils.Vocab(
            os.path.join(data_utils.__location__, 'pos_vocab'))

        # Create model.
        dataset, model = create_train_graph(sess, vocab, posseg_vocab)
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        checkpoint_path = os.path.join(FLAGS.train_dir, "lm.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        eval_loss_best = sys.float_info.max
        log_file = open(os.path.join(FLAGS.train_dir, 'train_log.txt'), 'a', 0)
        while True:
            start_time = time.time()
            input_feed = {dataset.handle: dataset.handles['train']}
            output_feed = [model.loss, model.update]
            step_loss, _ = model.step(sess, input_feed, output_feed, training=True)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                log_file.write("global step %d learning rate %.8f step-time %.4f loss "
                       "%.4f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, loss) + '\n')
                # Run evals on development set and print their perplexity.
                eval_losses = []
                input_feed = {dataset.handle:dataset.handles['valid']}
                output_feed = model.loss
                try:
                    while True:
                        eval_losses.append(model.step(sess, input_feed, output_feed, training=False))
                except:
                    dataset.reset(sess, 'valid')
                eval_loss = sum(eval_losses) / len(eval_losses)
                log_file.write("  eval loss %.4f" % eval_loss + '\n')
                previous_losses.append(eval_loss)
                sys.stdout.flush()
                # Save checkpoint and zero timer and loss.
                threshold = 10
                if len(previous_losses) > threshold and \
                    eval_loss > max(previous_losses[-threshold-1:-1]) and \
                    eval_loss_best < min(previous_losses[-threshold:]) and \
                    FLAGS.early_stop:
                    break
                # Save checkpoint and zero timer and loss.
                if eval_loss < eval_loss_best or (not FLAGS.early_stop):
                    checkpoint_path = os.path.join(FLAGS.train_dir, "lm.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    eval_loss_best = eval_loss
                step_time, loss = 0.0, 0.0
            if model.global_step.eval() >= FLAGS.steps_limit:
                break
        log_file.close()

def sample():

    """Sample a language model."""
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Read data into buckets and compute their sizes.
        vocab = data_utils.Vocab(
            os.path.join(FLAGS.data_dir, "vocab"),
            embedding_files=FLAGS.embedding_files)
        FLAGS.vocab_size = vocab.size()
        if FLAGS.embedding_files != "":
            FLAGS.vocab_dim = vocab.embedding_init.shape[1]
        posseg_vocab = data_utils.Vocab(
            os.path.join(data_utils.__location__, 'pos_vocab'))

        # Create model.
        with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
            seqs_placeholder, model = create_infer_graph(sess, vocab, posseg_vocab)

        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        text = ''
        while sentence:
            sentence = sentence.strip()
            if sentence == '':
                sentence = ''.join(text.split())
            sentence = data_utils.normalize(sentence)
            sample_seqs = vocab.sentence_to_token_ids(sentence)
            sample_seqs = np.array([sample_seqs+[0]])
            # This is the training loop.
            input_feed = {seqs_placeholder.name: sample_seqs}
            output_feed = [model.encodes, model.sample_seqs]
            encodes, sample_seqs = model.step(sess, input_feed, output_feed, training=False)
            print(map(lambda x: np.sum(x, axis=-1), np.split(encodes,2,axis=-1)))
            token_ids = list(sample_seqs[0])
            text = vocab.token_ids_to_sentence(token_ids)
            print(text)
            print("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def posseg():

    """Seg a language model."""
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Read data into buckets and compute their sizes.
        vocab = data_utils.Vocab(
            os.path.join(FLAGS.data_dir, "vocab"),
            embedding_files=FLAGS.embedding_files)
        FLAGS.vocab_size = vocab.size()
        if FLAGS.embedding_files != "":
            FLAGS.vocab_dim = vocab.embedding_init.shape[1]
        posseg_vocab = data_utils.Vocab(
            os.path.join(data_utils.__location__, 'pos_vocab'))

        # Create model.
        with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
            seqs_placeholder, model = create_infer_graph(sess, vocab, posseg_vocab)

        sample_len = 20
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            sentence = data_utils.normalize(sentence.strip())
            sample_seqs = vocab.sentence_to_token_ids(sentence)
            sample_seqs = np.array([sample_seqs])
            # This is the training loop.
            input_feed = {seqs_placeholder.name: sample_seqs}
            output_feed = model.segs
            segs = model.step(sess, input_feed, output_feed, training=False)
            segs = list(segs[0])[1:]
           # pos_labels = list(pos_labels[0])
           # pos_labels = map(lambda i: posseg_vocab.idx2key(i), pos_labels)
           # pos_labels.reverse()
            charlist = list(sentence.decode('utf-8'))
            charlist.reverse()
            possegs = []
            word = []
            length = len(charlist)
            for i in range(length):
                word.append(charlist.pop())
                if segs[i] > 0.0 or i == length-1:
                    possegs.append((''.join(word).encode('utf-8'), ))#pos_labels.pop()))
                    word = []
            print(possegs)
            print("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def test():

    """Train a language model."""
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Read data into buckets and compute their sizes.
        vocab = data_utils.Vocab(
            os.path.join(FLAGS.data_dir, "vocab"),
            embedding_files=FLAGS.embedding_files)
        FLAGS.vocab_size = vocab.size()
        if FLAGS.embedding_files != "":
            FLAGS.vocab_dim = vocab.embedding_init.shape[1]
        posseg_vocab = data_utils.Vocab(
            os.path.join(data_utils.__location__, 'pos_vocab'))

        # Create model.
        with tf.device('/gpu:{0}'.format(FLAGS.gpu_id)):
            dataset, model = create_train_graph(sess, vocab, posseg_vocab)

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
    elif FLAGS.posseg:
        posseg()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
