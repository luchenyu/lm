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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math, os, random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.client import timeline

from utils import data_utils, model_utils



class LM_Dataset(object):
    """Sequence-to-class model with multiple buckets.

       implements multiple classifiers
    """

    def __init__(self,
                 session,
                 vocab,
                 batch_size,
                 data_paths):
        """Create the dataset.

        Args:
            data_dir: path to the data files
        """

        def __sentence_to_token_ids(text):
            text = text.strip()
            seq = vocab.sentence_to_token_ids(text)
            return np.array(seq, dtype=np.int32)

        self.handles = {}
        self.iterators = {}
        for key in data_paths:
            data_path, mode = data_paths[key]
            dataset = tf.data.TextLineDataset([data_path])
            dataset = dataset.map(
                lambda text: tf.py_func(__sentence_to_token_ids, [text], tf.int32),
                num_parallel_calls=64)
            dataset = dataset.prefetch(buffer_size=10000)
            if mode == "repeat":
                dataset = dataset.filter(lambda seq: tf.less(tf.shape(seq)[0], 50))
                dataset = dataset.repeat()
                dataset = dataset.shuffle(buffer_size=1000000)
            dataset = dataset.padded_batch(batch_size, padded_shapes=[None])
            dataset = dataset.prefetch(buffer_size=100)
            iterator = dataset.make_initializable_iterator()
            self.iterators[key] = iterator
            handle = session.run(iterator.string_handle())
            self.handles[key] = handle
            session.run(iterator.initializer)

        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            self.handle, dataset.output_types, dataset.output_shapes)
        self.next_batch = iterator.get_next()

    def reset(self, session, key):
        """Reset the given iterator

        Args:
            key: key of the iterator

        """

        session.run(self.iterators[key].initializer)


    def get_batch(self, session, key):
        """Run the graph and get a batch.

        Args:
            session: tensorflow session to use.

        Returns:
            a batch of seqs

        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
        """

        next_batch = session.run(self.next_batch, feed_dict={self.handle: self.handles[key]})

        return next_batch

