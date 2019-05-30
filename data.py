import json, os, random, re
import numpy as np
import tensorflow as tf
from utils import data_utils_py3

""" dataset object
    data_config:
        {
            'segment_delim': str,
            'data_index': [ ## list by segment_id
                {
                    'field_id': int,
                    'item_id': int,
                },
            ],
            'data_schema': [ ## list by field_id
                {
                    'type': 'sequence'|'class',
                    'limited_vocab': bool,
                    'token_vocab': None|path,
                    'max_token_length': int,
                    'min_seq_length': int,
                    'max_seq_length': int,
                    'group_id': int,
                },
            ],
        }
"""

class Dataset(object):
    def __init__(self, path, char_vocab):
        self.path = path
        self.char_vocab = char_vocab
        self.data_config = self.parse_data_config(path, char_vocab)
        self.para_delim = re.compile(r'[ \t]*\t[ \t]*')
        self.word_delim = re.compile(r' +')

    def parse_data_config(self, path, char_vocab):
        """parse a dataset"""
        with open(os.path.join(path, 'data_config'), 'r') as json_file:
            data_config = json.load(json_file)
        for segment_id, item in enumerate(data_config['data_index']):
            field_id = item['field_id']
            segment_list = data_config['data_schema'][field_id].get('segment_list')
            if segment_list is None:
                data_config['data_schema'][field_id]['segment_list'] = []
            data_config['data_schema'][field_id]['segment_list'].append(segment_id)
        for field_id, item in enumerate(data_config['data_schema']):
            item['num_items'] = len(item['segment_list'])
            token_vocab = item.get('token_vocab')
            if not token_vocab is None:
                token_vocab = data_utils_py3.MolecularVocab(
                    char_vocab, filename=os.path.join(path, token_vocab))
                item['token_vocab'] = token_vocab
        return data_config

    def task_mapping(self, field_mapping, task_spec):
        """
        map a dataset for a task
        args:
            field_mapping: list by field_id, list elem is field_id of dataset
            'task_spec': [
                {
                    'type': 'sequence'|'class',
                    'copy_from': [field_ids],
                    'target_level': int >= 0,
                    'group_id': int,
                },
            ],
        return:
            mapped_index: [ ## list by feature_id
                {
                    'field_id': int,
                    'item_id': int,
                    'segment_id': int,
                },
            ],
            mapped_schema: [ ## list by field_id
                {
                    'type': 'sequence'|'class',
                    'limited_vocab': bool,
                    'token_vocab': None|path,
                    'max_token_length': int,
                    'min_seq_length': int,
                    'max_seq_length': int,
                    'group_id': int,
                },
            ],
        """
        mapped_index, mapped_schema = [], []
        assert(len(field_mapping) == len(task_spec))
        group_mapping = {}
        for target_field_id, source_field_id in enumerate(field_mapping):
            assert(
                task_spec[target_field_id]['type'] == \
                    self.data_config['data_schema'][source_field_id]['type'])
            target_group_id = task_spec[target_field_id]['group_id']
            source_group_id = self.data_config['data_schema'][source_field_id]['group_id']
            cached_group_id = group_mapping.get(target_group_id)
            if cached_group_id is None:
                group_mapping[target_group_id] = source_group_id
            else:
                assert(cached_group_id == source_group_id)

            num_items = self.data_config['data_schema'][source_field_id]['num_items']
            for item_id in range(num_items):
                if item_id < len(self.data_config['data_schema'][source_field_id]['segment_list']):
                    segment_id = \
                        self.data_config['data_schema'][source_field_id]['segment_list'][item_id]
                    assert(self.data_config['data_index'][segment_id]['item_id'] == item_id)
                else:
                    segment_id = None
                mapped_index.append({
                    'field_id': target_field_id,
                    'item_id': item_id,
                    'segment_id': segment_id,
                })

            mapped_schema.append({
                'type': task_spec[target_field_id]['type'],
                'limited_vocab': self.data_config['data_schema'][source_field_id]['limited_vocab'],
                'token_vocab': self.data_config['data_schema'][source_field_id].get('token_vocab'),
                'max_token_length': self.data_config['data_schema'][source_field_id].get('max_token_length'),
                'min_seq_length': self.data_config['data_schema'][source_field_id].get('min_seq_length'),
                'max_seq_length': self.data_config['data_schema'][source_field_id].get('max_seq_length'),
                'group_id': task_spec[target_field_id]['group_id'],
            })
        return mapped_index, mapped_schema

    def tokenize(self, feature_id, segment, random_start_end, mapped_index, mapped_schema, mode):
        field_id = mapped_index[feature_id]['field_id']
        item_id = mapped_index[feature_id]['item_id']
        group_id = mapped_schema[field_id]['group_id']
        segment = segment.strip()
        if segment == '':
            seqs = []
            segs = []
        elif mapped_schema[field_id]['type'] == 'sequence':
            paras = ' \t '.join(re.split(self.para_delim, segment))
            words = ['\t']+re.split(self.word_delim, paras)+['\t']
            # random sampling if too long
            max_seq_length = mapped_schema[field_id].get('max_seq_length')
            if mode == tf.estimator.ModeKeys.PREDICT:
                max_seq_length = None
            if (not max_seq_length is None) and len(words) > max_seq_length:
                if not random_start_end.get(str(group_id)+'-'+str(item_id)) is None:
                    start, end = random_start_end[str(group_id)+'-'+str(item_id)]
                else:
                    start = random.randint(0, len(words)-max_seq_length+1)
                    end = start+max_seq_length
                    random_start_end[str(group_id)+'-'+str(item_id)] = (start, end)
                words = words[start:end]
            # tokenize
            if mapped_schema[field_id]['limited_vocab']:
                token_vocab = mapped_schema[field_id]['token_vocab']
                seqs = data_utils_py3.tokens_to_seqs(words, token_vocab)
                segs = [1.0,]*(len(seqs)+1)
            else:
                seqs, segs = data_utils_py3.tokens_to_seqs_segs(
                    words, self.char_vocab)
        elif mapped_schema[field_id]['type'] == 'class':
            # tokenize
            tokens = [segment]
            if mapped_schema[field_id]['limited_vocab']:
                token_vocab = mapped_schema[field_id]['token_vocab']
                seqs = data_utils_py3.tokens_to_seqs(tokens, token_vocab)
                segs = []
            else:
                seqs, segs = data_utils_py3.tokens_to_seqs_segs(
                    tokens, self.char_vocab)
        else:
            raise NameError('wrong data type!')
        seqs = np.array(seqs, dtype=np.int32)
        segs = np.array(segs, dtype=np.float32)
        return [seqs, segs]

    def file_input_fn(self, filename,
                      mapped_index, mapped_schema,
                      batch_size, mode):
        """
        An input function for training
        each line is an example
        each example contains one or more features separated by segment_delim ('***|||***')
        each feature contains one or more paragraphs separated by '\t'
        each paragraph contains one or more words separated by ' '
        args:
            filename: name of data file or folder
            mapped_index: [ ## list by feature_id
                {
                    'field_id': int,
                    'item_id': int,
                    'segment_id': int,
                },
            ],
            'mapped_schema': [ ## list by field_id
                {
                    'type': 'sequence'|'class',
                    'limited_vocab': bool,
                    'token_vocab': None|path,
                    'max_token_length': int,
                    'min_seq_length': int,
                    'max_seq_length': int,
                    'group_id': int,
                },
            ],
            batch_size: int
            mode: ModeKeys
        """
        data_path = os.path.join(self.path, filename)
        if os.path.isdir(data_path):
            filenames = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]
            random.shuffle(filenames)
        else:
            filenames = [data_path]
        dataset = tf.data.TextLineDataset(filenames)

        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.repeat()
            shuffle_pool = max(100000, len(list(open(filenames[0], 'r'))))
            dataset = dataset.shuffle(buffer_size=shuffle_pool)

        def _featurize(text):
            text = text.numpy().decode('utf-8').strip()
            segments = text.split(self.data_config['segment_delim'])
            random_start_end = {}
            features = sum([
                self.tokenize(
                    i, segments[item['segment_id']], random_start_end, mapped_index, mapped_schema, mode)
                if not item['segment_id'] is None else
                self.tokenize(
                    i, '', random_start_end, mapped_index, mapped_schema, mode)
                for i, item in enumerate(mapped_index)], [])
            return features

        def _format(features):
            seqs_list = features[::2]
            segs_list = features[1::2]
            features = {}
            for i, (seqs, segs) in enumerate(zip(seqs_list, segs_list)):
                features[str(i)+'-seqs'] = seqs
                features[str(i)+'-segs'] = segs
            return features, tf.zeros([]) # (features, labels)

        def _filter(features, labels):
            valids = []
            for i in range(len(mapped_index)):
                field_id = mapped_index[i]['field_id']
                min_seq_length = mapped_schema[field_id].get('min_seq_length')
                if not min_seq_length is None:
                    valids.append(tf.logical_or(
                        tf.equal(tf.shape(features[str(i)+'-seqs'])[0], 0),
                        tf.greater(tf.reduce_sum(features[str(i)+'-segs']), min_seq_length+2)))
            if len(valids) > 0:
                return tf.reduce_all(tf.stack(valids))
            else:
                return True

        num_features = len(mapped_index)
        dataset = dataset.map(
            lambda text: _format(tf.py_function(
                _featurize, [text], [tf.int32, tf.float32]*num_features)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(_filter)

        num_features = len(mapped_index)
        padded_shapes = {}
        for i in range(num_features):
            padded_shapes[str(i)+'-seqs'] = [None]
            padded_shapes[str(i)+'-segs'] = [None]
        padded_shapes = (padded_shapes, [])
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def serving_input_fn(self, mapped_index, mapped_schema, task_spec):
        """
        input fn for serving
        """
        receiver_tensors = {}
        for i in range(len(mapped_index)):
            field_id = mapped_index[i]['field_id']
            target_level = task_spec[field_id]['target_level']
            if target_level == 0:
                receiver_tensors[str(i)+'-seqs'] = tf.placeholder(
                    dtype=tf.int32, shape=[None, None])
                receiver_tensors[str(i)+'-segs'] = tf.placeholder(
                    dtype=tf.float32, shape=[None, None])
        features = {}
        for key in receiver_tensors:
            features[key] = tf.identity(receiver_tensors[key])
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    def textify(self, field_id, seqs, segs):
        """
        turn ids to a text
        args:
            field_id:
            seqs: length
            segs: 0 or (length+1)
        return:
            text: string
        """
        if self.data_config['data_schema'][field_id]['limited_vocab']:
            text = ' '.join(
                data_utils_py3.seqs_to_tokens(
                    seqs, self.data_config['data_schema'][field_id]['token_vocab']))
        else:
            text = ' '.join(
                data_utils_py3.seqs_segs_to_tokens(
                    seqs, segs, self.char_vocab))
        return text

    def build_request(self, mapped_segments_list, mapped_index, mapped_schema, task_spec):
        """
        build request to the model server
        """
        req = {'instances':[]}
        for mapped_segments in mapped_segments_list:
            ins = {}
            for i, segment in enumerate(mapped_segments):
                field_id = mapped_index[i]['field_id']
                target_level = task_spec[field_id]['target_level']
                if target_level == 0:
                    seqs, segs = self.tokenize(
                        i, segment, None,
                        mapped_index, mapped_schema, tf.estimator.ModeKeys.PREDICT)
                    ins[str(i)+'-seqs'] = seqs.tolist()
                    ins[str(i)+'-segs'] = segs.tolist()
            req['instances'].append(ins)
        return req

    def parse_response(self, resp, mapped_index, mapped_schema):
        """
        parse response from the model server
        """
        mapped_segments_list = []
        for pred in resp['predictions']:
            mapped_segments = []
            for i in range(len(mapped_index)):
                seqs = pred.get(str(i)+'-seqs')
                segs = pred.get(str(i)+'-segs')
                if not seqs is None:
                    segment_id = mapped_index[i]['segment_id']
                    field_id = self.data_config['data_index'][segment_id]['field_id']
                    text = self.textify(field_id, seqs, segs)
                else:
                    text = ''
                mapped_segments.append(text)
            mapped_segments_list.append(mapped_segments)
        return mapped_segments_list