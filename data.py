import os, random, re
import numpy as np
import tensorflow as tf
from utils import data_utils_py3


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

    para_delim = re.compile(r'[ \t]*\t[ \t]*')
    word_delim = re.compile(r' +')
    def _featurize(text):
        text = text.numpy().decode('utf-8').strip()
        pieces = text.split('***|||***')
        def _tokenize(i, piece):
            paras = ' \t '.join(re.split(para_delim, piece.strip(' \t')))
            words = ['\t']+re.split(word_delim, paras)+['\t']
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

