import logging, os, traceback
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
                    'train', mapped_index, mapped_schema,
                    hyper_params['batch_size'], tf.estimator.ModeKeys.TRAIN),
                steps=steps)
            counter = model.get_global_step()
            lm.evaluate(
                input_fn=lambda: dataset.file_input_fn(
                    'dev', mapped_index, mapped_schema,
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
    hyper_params):
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
            'eval', mapped_index, mapped_schema,
            hyper_params['batch_size'], tf.estimator.ModeKeys.EVAL))

def predict(
    dataset,
    model,
    field_mapping,
    hyper_params):
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
            'test', mapped_index, mapped_schema,
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
            text_list = []
            for feature_id in target_feature_ids:
                seqs = pred[str(feature_id)+'-seqs']
                segs = pred[str(feature_id)+'-segs']
                source_field_id = field_mapping[mapped_index[feature_id]['field_id']]
                text = dataset.textify(source_field_id, seqs, segs)
                text_list.append(text)
            fwrite.write(dataset.data_config['segment_delim'].join(text_list)+'\n')

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
        model.train_dir, lambda: dataset.serving_input_fn(mapped_index, mapped_schema), strip_default_attrs=True)
