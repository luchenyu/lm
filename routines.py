import logging, os
import tensorflow as tf


"""high level routines"""

def lr_range_test(
    lm_model_fn,
    train_input_fn,
    params,
    num_steps=1000):
    """
    train the model and evaluate every eval_every steps
    """
    
#     gpu_id = '2'
#     session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(visible_device_list=gpu_id))
#     strategy = tf.distribute.MirroredStrategy()
    config=tf.estimator.RunConfig(
#         train_distribute=strategy,
#         eval_distribute=strategy,
#         session_config=session_config,
        save_checkpoints_steps=None,
        save_checkpoints_secs=None,
        log_step_count_steps=100)
    local_params = {}
    local_params.update(params)
    local_params['schedule'] = 'lr_finder'
    local_params['num_steps'] = num_steps
    lm = tf.estimator.Estimator(
        model_fn=lm_model_fn,
        model_dir='',
        params=local_params,
        config=config)

    # start lr range test
    lm.train(
        input_fn=train_input_fn,
        steps=num_steps)

def train_and_evaluate(
    lm_model_fn,
    train_input_fn,
    eval_input_fn,
    train_dir,
    params,
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
        eval_distribute=strategy,
        log_step_count_steps=1000)
    local_params = {}
    local_params.update(params)
    local_params['schedule'] = '1cycle'
    local_params['distributed'] = distributed
    lm = tf.estimator.Estimator(
        model_fn=lm_model_fn,
        model_dir=train_dir,
        params=local_params,
        config=config)
    
    # get TF logger
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    os.makedirs(train_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(train_dir, 'tensorflow.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    
    # start train and eval loop
    for _ in range(int(params['num_steps'] / eval_every)):
        lm.train(
            input_fn=train_input_fn,
            steps=eval_every)
        lm.evaluate(
            input_fn=eval_input_fn)
