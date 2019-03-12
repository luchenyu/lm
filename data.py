import json, os, random, re
import numpy as np
import tensorflow as tf
from utils import data_utils_py3

""" dataset object
    data_config: {'field_delim': string, 'schema': [{'type': 'sequence'|'class', 'token_vocab_file': none|path}, ],}
"""

class Dataset(object):
    def __init__(self, path, char_vocab):
        self.path = path
        with open(os.path.join(path, 'data_config'), 'r') as json_file:
            data_config = json.load(json_file)
        self.schema = data_config['schema']
        self.field_delim = data_config.get('field_delim')
        self.char_vocab = char_vocab
        for piece in self.schema:
            token_vocab_file = piece.get('token_vocab_file')
            if token_vocab_file != None:
                token_vocab = data_utils_py3.Vocab(token_vocab_file)
                piece['token_vocab'] = token_vocab
                piece['token_char_ids'] = np.array(data_utils_py3.tokens_to_char_ids(
                    [list(i) if i != '_PAD' and i != '_EOS' and i != '_UNK' else [i] for i in token_vocab.vocab_list], char_vocab))

    def file_input_fn(self, data_name, run_config, mode):
        """
        An input function for training
        each line is an example
        each example contains one or more pieces separated by '***|||***'
        each piece contains one or more paragraphs separated by '\t'
        each paragraph contains one or more words separated by ' '
        args:
            self: Dataset object
            data_name: name of data file or folder
            run_config: {'batch_size': int, 'max_train_steps': int,
                'max_lr': float, 'pct_start': [0,1], 'dropout': [0,1], 'wd': float,
                'data': [{'is_target': true|false, 'max_token_length': int, 'min_seq_length': int, 'max_seq_length': int},],}
            mode: ModeKeys
        """

        data_path = os.path.join(self.path, data_name)
        num_pieces = len(self.schema)
        if os.path.isdir(data_path):
            filenames = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]
            random.shuffle(filenames)
        else:
            filenames = [data_path]
        dataset = tf.data.TextLineDataset(filenames)

        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=50000)

        para_delim = re.compile(r'[ \t]*\t[ \t]*')
        word_delim = re.compile(r' +')
        def _featurize(text):
            text = text.numpy().decode('utf-8').strip()
            pieces = text.split(self.field_delim)
            def _tokenize(i, piece):
                if self.schema[i]['type'] == 'sequence':
                    paras = ' \t '.join(re.split(para_delim, piece.strip(' \t')))
                    words = ['\t']+re.split(word_delim, paras)+['\t']
                    # random sampling if too long
                    max_seq_length = run_config['data'][i].get('max_seq_length')
                    if max_seq_length != None and len(words) > max_seq_length:
                        start = random.randint(0, len(words)-max_seq_length+1)
                        words = words[start:start+max_seq_length]
                    # tokenize
                    token_vocab = self.schema[i].get('token_vocab')
                    if token_vocab != None:
                        tokens = ['_EOS' if word == '\t' else word for word in words]
                        seqs = data_utils_py3.tokens_to_token_ids(tokens, token_vocab)
                        segs = []
                    else:
                        tokens = [['_EOS'] if word == '\t' else list(word) for word in words]
                        seqs, segs = data_utils_py3.tokens_to_seqs_segs(tokens, self.char_vocab)
                elif self.schema[i]['type'] == 'class':
                    # tokenize
                    token_vocab = self.schema[i].get('token_vocab')
                    if token_vocab != None:
                        tokens = [piece.strip()]
                        seqs = data_utils_py3.tokens_to_token_ids(tokens, token_vocab)
                        segs = []
                    else:
                        tokens = [list(piece.strip())]
                        seqs, segs = data_utils_py3.tokens_to_seqs_segs(tokens, self.char_vocab)
                else:
                    raise NameError('wrong data type!')
                seqs = np.array(seqs, dtype=np.int32)
                segs = np.array(segs, dtype=np.float32)
                return [seqs, segs]
            features = sum([_tokenize(i, piece) for i, piece in enumerate(pieces)], [])
            return features

        def _format(features):
            seqs_list = features[::2]
            segs_list = features[1::2]
            features = {}
            for i, (seqs, segs) in enumerate(zip(seqs_list, segs_list)):
                features[i] = {'seqs': seqs, 'segs': segs}
            return features, tf.zeros([]) # (features, labels)
        
        def _filter(features, labels):
            valids = []
            for i, feature in features.items():
                min_seq_length = run_config['data'][i].get('min_seq_length')
                if min_seq_length != None:
                    valids.append(tf.greater(tf.shape(feature['seqs'])[0], min_seq_length))
            if len(valids) > 0:
                return tf.reduce_all(tf.stack(valids))
            else:
                return True

        dataset = dataset.map(
            lambda text: _format(tf.py_function(_featurize, [text], [tf.int32, tf.float32]*num_pieces)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(_filter)

        padded_shapes = {}
        for i in range(num_pieces):
            padded_shapes[i] = {'seqs': [None], 'segs': [None]}
        padded_shapes = (padded_shapes, [])
        dataset = dataset.padded_batch(run_config['batch_size'], padded_shapes=padded_shapes)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

