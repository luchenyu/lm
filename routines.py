import json, logging, os, requests, traceback
import tensorflow as tf


"""high level routines"""

def lr_range_test(
    dataset,
    model,
    field_mapping,
    hyper_params,
    num_steps=1000):
    """
    train the model and evaluate every eval_every steps
    """
    
    config=tf.estimator.RunConfig(
        save_checkpoints_steps=None,
        save_checkpoints_secs=None,
        log_step_count_steps=int(num_steps/10))

    mapped_index, mapped_schema = dataset.task_mapping(
        field_mapping, model.task_config['task_spec'])
    params = {
        'data_index': mapped_index,
        'data_schema': mapped_schema,
        'hyper_params': hyper_params,
        'schedule': 'lr_finder',
        'num_steps': num_steps,
    }

    # build estimator
    lm = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir='',
        params=params,
        config=config,
        warm_start_from=model.warm_start_from)

    # start lr range test
    lm.train(
        input_fn=lambda: dataset.file_input_fn(
            'train', mapped_index, mapped_schema,
            hyper_params['batch_size'], tf.estimator.ModeKeys.TRAIN),
        steps=num_steps)

def train_and_evaluate(
    dataset,
    model,
    field_mapping,
    hyper_params,
    train_file='train',
    dev_file='dev',
    eval_every=10000,
    distributed=True):
    """
    train the model and evaluate every eval_every steps
    """
    
    if distributed:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None

    config=tf.estimator.RunConfig(
        train_distribute=strategy,
        log_step_count_steps=1000)

    mapped_index, mapped_schema = dataset.task_mapping(
        field_mapping, model.task_config['task_spec'])
    params = {
        'data_index': mapped_index,
        'data_schema': mapped_schema,
        'hyper_params': hyper_params,
        'schedule': '1cycle',
        'num_steps': hyper_params['max_train_steps'],
    }
    if distributed:
        params['distributed'] = True

    # build estimator
    lm = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=model.train_dir,
        params=params,
        config=config,
        warm_start_from=model.warm_start_from)

    # get TF logger
    log = logging.getLogger('tensorflow')
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(model.train_dir, 'tensorflow.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    try:
        log.addHandler(fh)

        # start train and eval loop
        counter = model.get_global_step()
        while counter < hyper_params['max_train_steps']:
            steps = min(
                eval_every - (counter % eval_every),
                hyper_params['max_train_steps'] - counter)
            lm.train(
                input_fn=lambda: dataset.file_input_fn(
                    train_file, mapped_index, mapped_schema,
                    hyper_params['batch_size'], tf.estimator.ModeKeys.TRAIN),
                steps=steps)
            counter = model.get_global_step()
            lm.evaluate(
                input_fn=lambda: dataset.file_input_fn(
                    dev_file, mapped_index, mapped_schema,
                    hyper_params['batch_size'], tf.estimator.ModeKeys.EVAL))
    except:
        traceback.print_exc()
    finally:
        # clear logger handler
        log.removeHandler(fh)

def evaluate(
    dataset,
    model,
    field_mapping,
    hyper_params,
    eval_file='eval'):
    """
    train the model and evaluate every eval_every steps
    """

    config=tf.estimator.RunConfig(
        log_step_count_steps=1000)

    mapped_index, mapped_schema = dataset.task_mapping(
        field_mapping, model.task_config['task_spec'])
    params = {
        'data_index': mapped_index,
        'data_schema': mapped_schema,
        'hyper_params': hyper_params,
    }

    # build estimator
    lm = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=model.train_dir,
        params=params,
        config=config,
        warm_start_from=model.warm_start_from)

    # start evaluation
    lm.evaluate(
        input_fn=lambda: dataset.file_input_fn(
            eval_file, mapped_index, mapped_schema,
            hyper_params['batch_size'], tf.estimator.ModeKeys.EVAL))

def predict(
    dataset,
    model,
    field_mapping,
    hyper_params,
    test_file='test'):
    """
    use trained model to do prediction on test samples
    """

    config=tf.estimator.RunConfig(
        log_step_count_steps=1000)

    mapped_index, mapped_schema = dataset.task_mapping(
        field_mapping, model.task_config['task_spec'])
    params = {
        'data_index': mapped_index,
        'data_schema': mapped_schema,
        'hyper_params': hyper_params,
    }
    gen_num_cands = hyper_params.get('gen_num_cands')
    gen_num_cands = 1 if gen_num_cands is None else gen_num_cands

    # build estimator
    lm = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=model.train_dir,
        params=params,
        config=config,
        warm_start_from=model.warm_start_from)

    # start prediction
    predictions = lm.predict(
        input_fn=lambda: dataset.file_input_fn(
            test_file, mapped_index, mapped_schema,
            hyper_params['batch_size'], tf.estimator.ModeKeys.PREDICT))

    # get the target features
    task_spec = model.task_config['task_spec']
    max_target_id = max([item['target_level'] for item in task_spec])
    if max_target_id == 0:
        target_feature_ids = list(range(len(mapped_index)))
    else:
        target_feature_ids = []
        for i, item in enumerate(mapped_index):
            if task_spec[item['field_id']]['target_level'] > 0:
                target_feature_ids.append(i)

    # loop predictions and write
    data_path = os.path.join(model.train_dir, 'predictions')
    with open(data_path, 'w') as fwrite:
        for pred in predictions:
            segment_list = []
            for feature_id in target_feature_ids:
                cand_list = []
                source_field_id = field_mapping[mapped_index[feature_id]['field_id']]
                for k in range(gen_num_cands):
                    seqs = pred[str(feature_id)+'-seqs-'+str(k)]
                    segs = pred[str(feature_id)+'-segs-'+str(k)]
                    scores = pred[str(feature_id)+'-scores-'+str(k)]
                    text = dataset.textify(source_field_id, seqs, segs)
                    cand_list.append(text+'@@@'+str(scores))
                segment_list.append('|||'.join(cand_list))
            fwrite.write(
                dataset.data_config['segment_delim'].join(segment_list) + '\n')

def export(
    dataset,
    model,
    field_mapping,
    hyper_params):
    """
    export saved model
    """

    config=tf.estimator.RunConfig(
        log_step_count_steps=1000)

    mapped_index, mapped_schema = dataset.task_mapping(
        field_mapping, model.task_config['task_spec'])
    params = {
        'data_index': mapped_index,
        'data_schema': mapped_schema,
        'hyper_params': hyper_params,
    }

    # build estimator
    lm = tf.estimator.Estimator(
        model_fn=model.model_fn,
        model_dir=model.train_dir,
        params=params,
        config=config,
        warm_start_from=model.warm_start_from)

    # perform the export
    lm.export_savedmodel(
        model.train_dir, lambda: dataset.serving_input_fn(
            mapped_index, mapped_schema, model.task_config['task_spec']),
        strip_default_attrs=True)

def convert_graph_def_to_saved_model(
    export_dir,
    graph_filepath,
    inputs,
    outputs):
    """
    convert GraphDef to SavedModel
    """
    def get_graph_def_from_file(graph_filepath):
        with tf.Graph().as_default():
            with tf.gfile.GFile(graph_filepath, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                return graph_def
    if tf.gfile.Exists(export_dir):
        tf.gfile.DeleteRecursively(export_dir)
    graph_def = get_graph_def_from_file(graph_filepath)
    with tf.Session(graph=tf.Graph()) as session:
        tf.import_graph_def(graph_def, name='')
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={
                key: session.graph.get_tensor_by_name(inputs[key]) for key in inputs},
            outputs={
                key: session.graph.get_tensor_by_name(outputs[key]) for key in outputs},
        )
    print('Optimized graph converted to SavedModel!')

class Client(object):
    def __init__(self, dataset, task_spec, field_mapping, model_name, ip='http://localhost', port=8501):
        self.dataset = dataset
        self.headers = {"content-type": "application/json"}
        self.addr = ip.strip('/')+':'+str(port)+'/v1/models/'+model_name+':predict'
        self.task_spec = task_spec
        self.mapped_index, self.mapped_schema = dataset.task_mapping(
            field_mapping, task_spec)

    def make_request(self, mapped_segments_list):
        req = self.dataset.build_request(
            mapped_segments_list, self.mapped_index, self.mapped_schema, self.task_spec)
        req = json.dumps(req, ensure_ascii=False)
        resp = requests.post(self.addr, data=req, headers=self.headers)
        resp = json.loads(resp.text, encoding='utf-8')
        mapped_segments_list = self.dataset.parse_response(
            resp, self.mapped_index, self.mapped_schema)
        return mapped_segments_list