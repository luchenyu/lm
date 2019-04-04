import logging, os, traceback
import tensorflow as tf


"""high level routines"""

def lr_range_test(
    dataset,
    model,
    run_config,
    num_steps=1000):
    """
    train the model and evaluate every eval_every steps
    """
    
    config=tf.estimator.RunConfig(
        save_checkpoints_steps=None,
        save_checkpoints_secs=None,
        log_step_count_steps=100)
    params = {'schema': dataset.schema, 'run_config': run_config, 'schedule': 'lr_finder', 'num_steps': num_steps}

    # build estimator
    lm = tf.estimator.Estimator(
        model_fn=model.lm_model_fn,
        model_dir='',
        params=params,
        config=config,
        warm_start_from=model.warm_start_from)

    # start lr range test
    lm.train(
        input_fn=lambda: dataset.file_input_fn('train', run_config, tf.estimator.ModeKeys.TRAIN),
        steps=num_steps)

def train_and_evaluate(
    dataset,
    model,
    run_config,
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
    params = {'schema': dataset.schema, 'run_config': run_config, 'schedule': '1cycle', 'num_steps': run_config['max_train_steps']}
    if distributed:
        params['distributed'] = True

    # build estimator
    lm = tf.estimator.Estimator(
        model_fn=model.lm_model_fn,
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
        while counter < run_config['max_train_steps']:
            steps = min(eval_every - (counter % eval_every), run_config['max_train_steps'] - counter)
            lm.train(
                input_fn=lambda: dataset.file_input_fn('train', run_config, tf.estimator.ModeKeys.TRAIN),
                steps=steps)
            counter = model.get_global_step()
            lm.evaluate(
                input_fn=lambda: dataset.file_input_fn('dev', run_config, tf.estimator.ModeKeys.EVAL))
    except:
        traceback.print_exc()
    finally:
        # clear logger handler
        log.removeHandler(fh)

def evaluate(
    dataset,
    model,
    run_config):
    """
    train the model and evaluate every eval_every steps
    """

    config=tf.estimator.RunConfig(
        log_step_count_steps=1000)
    params = {'schema': dataset.schema, 'run_config': run_config}

    # build estimator
    lm = tf.estimator.Estimator(
        model_fn=model.lm_model_fn,
        model_dir=model.train_dir,
        params=params,
        config=config,
        warm_start_from=model.warm_start_from)

    # start train and eval loop
    lm.evaluate(
        input_fn=lambda: dataset.file_input_fn('eval', run_config, tf.estimator.ModeKeys.EVAL))
