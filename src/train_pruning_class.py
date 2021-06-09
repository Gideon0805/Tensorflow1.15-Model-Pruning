import os
import logging
from functools import partial

import tensorflow as tf

from input_pipeline import Pipeline
from losses import MSMSE
from python.tf1pruning import Pruning


slim = tf.contrib.slim
flags = tf.app.flags
flags.DEFINE_string(
    'dataset_path',
    '/datasets/t3/data/coco/intermediate/coco_keypoints_train.record-00000-of-00001,/datasets/t3/data/PoseTrack/intermediate/PoseTrack_keypoints_train.record-00000-of-00001,/datasets/t3/data/mpii/intermediate/mpii_keypoints_train.record-00000-of-00001',
    'Training data (separated by comma)'
)
flags.DEFINE_string(
    'validationset_path',
    '/datasets/t3/data/panoptic/intermediate/panoptic_keypoints_val.record-00000-of-00001',
    'Validation data'
)
flags.DEFINE_string(
    'output_model_path',
    '/workspace/model_pruning/Testing/pe_pruning',
    'Path of output human pose model'
)
flags.DEFINE_string(
    'pretrained_model_path',
    '/workspace/model_pruning/Testing/edited_ckpt/model.ckpt-993000',
    'Path of pretrained model(ckpt)'
)
flags.DEFINE_float(
    'layer_depth_multiplier',
    0.5,
    'Depth multiplier of mobilenetv1 architecture'
)
flags.DEFINE_integer(
    'number_keypoints',
    17,
    'Number of keypoints in [17, 12]'
)
flags.DEFINE_integer(
    'batch_size',
    32,
    'Size of batch data'
)
flags.DEFINE_boolean(
    'pretrained_model',
    True,
    'Use pretrained model or not'
)
flags.DEFINE_boolean(
    'data_augmentation',
    False,
    'Add data augmentation to preprocess'
)
flags.DEFINE_string(
    'optimizer',
    'Adam',
    'Optimizer in [Momentum, Adagrad, Adam, RMSProp, Nadam]'
)
flags.DEFINE_float(
    'learning_rate',
    0.001,
    'Learning rate for training process'
)
flags.DEFINE_integer(
    'decay_steps',
    1000000000,
    'Decay steps of learning rate'
)
flags.DEFINE_float(
    'decay_factor',
    0.1,
    'Decay factor of learning rate'
)
flags.DEFINE_integer(
    'training_steps',
    100000,
    'Train n steps'
)
flags.DEFINE_integer(
    'validation_interval',
    10000,
    'Evaluate validation loss for every n steps'
)
flags.DEFINE_integer(
    'validation_batch_size',
    256,
    'Size of batch data'
)
flags.DEFINE_integer(
    'ohem_top_k',
    8,
    'online hard example/keypoint mining choice top k keypoint'
)

FLAGS = flags.FLAGS
pretrained_model_steps = 13109116


class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, input_fn):
        self.estimator = estimator
        self.input_fn = input_fn

    def after_save(self, session, global_step):
        self.estimator.evaluate(
            self.input_fn, steps=len(list(tf.python_io.tf_record_iterator(FLAGS.validationset_path))) // FLAGS.validation_batch_size + 1
        )


def train_op(labels, net_dict, loss_fn, learning_rate, Optimizer, global_step=0, ohem_top_k=8):
    # loss function is MSE_OHEM
    loss_pre = tf.losses.mean_squared_error(labels[0], net_dict['heat_map'], reduction=tf.losses.Reduction.NONE)
    loss_pre = tf.reduce_mean(loss_pre, (1,2))
    sub_loss, _ = tf.nn.top_k(loss_pre, ohem_top_k, name="ohem_test")
    loss = tf.reduce_mean(sub_loss)

    if Optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    elif Optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif Optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif Optimizer == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif Optimizer == 'Nadam':
        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError('''{} optimizer is not supported. 
            Please choose one of ["Momentum", "Adagrad", "Adam", "RMSProp", "Nadam"]'''.format(Optimizer))

    train = optimizer.minimize(loss, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([train]):
        train_ops = tf.group(*update_ops)
    return train_ops, loss


def model_fn(features, labels, mode, params):
    from se_mobilepose import SEMobilePose
    model_arch = SEMobilePose
    output = 'heat_map'
    multi_layer_labels = labels[0]
    hm_labels = labels[0][0]
    labels_weight = labels[1]

    if mode == tf.estimator.ModeKeys.TRAIN:
        model = model_arch(backbone=params['backbone'],
                           is_training=True,
                           depth_multiplier=params['layer_depth_multiplier'],
                           number_keypoints=params['number_keypoints'])

        end_points = model.build(features)

    else:
        model = model_arch(backbone=params['backbone'],
                           is_training=False,
                           depth_multiplier=params['layer_depth_multiplier'],
                           number_keypoints=params['number_keypoints'])

        end_points = model.build(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={output: end_points[output]})

    learning_rate = tf.train.exponential_decay(
        params['initial_learning_rate'],
        tf.train.get_global_step(),
        params['decay_steps'],
        params['decay_factor'],
        staircase=True
    )
    train, loss = train_op(
        multi_layer_labels,
        end_points,
        loss_fn=params['loss_fn'],
        learning_rate=learning_rate,
        Optimizer=params['optimizer'],
        global_step=tf.train.get_global_step(),
        ohem_top_k=params['ohem_top_k']
    )

    def find_keypoints(heat_map):
        inds = []
        for k in range(params['number_keypoints']):
            ind = tf.unravel_index(
                tf.argmax(
                    tf.reshape(heat_map[:, :, k], [-1])),
                [80, 64]
            )
            inds.append(tf.cast(ind, tf.float32))
        return tf.stack(inds)
    keypoints_pridict = tf.map_fn(find_keypoints,
                                  end_points[output],
                                  back_prop=False)
    keypoints_labels = tf.map_fn(find_keypoints,
                                 hm_labels,
                                 back_prop=False)
    err = tf.losses.mean_squared_error(keypoints_labels, keypoints_pridict, labels_weight)

    if mode == tf.estimator.ModeKeys.EVAL:
        mean_err, mean_err_op = tf.metrics.mean(err)
        mean_loss, mean_loss_op = tf.metrics.mean(loss)
        evaluation_hook = tf.train.LoggingTensorHook(
            {'Global Steps': tf.train.get_global_step(),
             'Distance Error': mean_err_op,
             'Evaluation Loss': mean_loss_op,
             'Batch Size': tf.shape(hm_labels)[0],
             'Learning Rate': learning_rate},
            every_n_iter=1,
            every_n_secs=None,
            at_end=False
        )
        init_counter = tf.train.Scaffold(
            init_fn=tf.local_variables_initializer
        )
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          scaffold=init_counter,
                                          evaluation_hooks=[evaluation_hook])

    training_hook = tf.train.LoggingTensorHook(
        {'Global Steps': tf.train.get_global_step(),
         'Distance Error': err,
         'Training Loss': loss,
         'Learning Rate': learning_rate},
        every_n_iter=100,
        every_n_secs=None,
        at_end=False
    )
    if params['pretrained_model']:
        saver = get_pretrained_model_saver(params['pretrained_model_path'])
        load_fn = partial(load_for_estimator,
                          data_path=params['pretrained_model_path'],
                          saver=saver)
        init_load = tf.train.Scaffold(init_fn=load_fn)
    else:
        init_load = None
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train,
                                      scaffold=init_load,
                                      training_hooks=[training_hook])


def get_pretrained_model_saver(pretrained_model_path):
    reader = tf.train.NewCheckpointReader(pretrained_model_path)
    #reader = tf.train.NewCheckpointReader(pretrained_model_path + '/model.ckpt-13139116')
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                       if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    return tf.train.Saver(restore_vars)


def load_for_estimator(scaffold, session, data_path, saver):
    '''Load network weights.
    scaffold: tf.train.Scaffold object
    session: tf.Session()
    data_path: The path to the numpy-serialized network weights
    session: The current TensorFlow session
    '''
    print('Global steps:', session.run(tf.train.get_global_step()))
    if session.run(tf.train.get_global_step()) != 0:
        return
    saver.restore(session, data_path)
    # saver.restore(session, data_path + '/model.ckpt-13139116')
    session.graph._unsafe_unfinalize()
    session.run(tf.assign(tf.train.get_global_step(), 0))
    session.graph.finalize()


def LR(initial_learning_rate, global_step, decay_steps, decay_factor):
    return initial_learning_rate * decay_factor ** (global_step // decay_steps)

def main(_):
    pruning_scopes = ['SE1', 'SE2', 'SE3']
    for i in range(1,17):
        name = 'MobilenetV2/expanded_conv_' + str(i) + '/'
        pruning_scopes.append(name)
        pass

    output_dir = '/workspace/model_pruning/Testing/Pruning_Class_Test'
    tf_pruning = Pruning(
        input_ckpt_path=FLAGS.pretrained_model_path,
        scope_names=pruning_scopes,
        output_dir=output_dir
        )

    datasets = FLAGS.dataset_path.split(',')
    model_params = {
        'model_arch': 'SEMobilePose',
        'backbone': 'mobilenet_v2',
        'loss_fn': 'MSE_OHEM',
        'optimizer': FLAGS.optimizer,
        'initial_learning_rate': FLAGS.learning_rate,
        'decay_steps': FLAGS.decay_steps,
        'decay_factor': FLAGS.decay_factor,
        'layer_depth_multiplier': FLAGS.layer_depth_multiplier,
        'number_keypoints': FLAGS.number_keypoints,
        'pretrained_model': FLAGS.pretrained_model,
        'pretrained_model_path': FLAGS.pretrained_model_path,
        'ohem_top_k': FLAGS.ohem_top_k
    }
    pipeline_param = {
        'model_arch': 'SEMobilePose',
        'do_data_augmentation': FLAGS.data_augmentation,
        'loss_fn': 'MSE_OHEM',
        'number_keypoints': FLAGS.number_keypoints,
        'dataset_split_num': len(datasets),
    }

    th_steps = [0.8, 0.7, 0.6, 0.55, 0.5]
    for th in th_steps:
        try_count = 0
        failed_rate = 1.0
        tf_pruning.set_threshold(th)
        while failed_rate>0.1:
            if try_count == 0:
                pruning_output_dir = os.path.join(output_dir, 'TH'+str(th))
                tf_pruning.set_output_dir(pruning_output_dir)
                ckpt_to_retrain = tf_pruning.pruning_process(retrain=False)
            else:
                pruning_output_dir = os.path.join(output_dir, 'TH'+str(th)+'_Repruned'+str(try_count))
                tf_pruning.set_output_dir(pruning_output_dir)
                ckpt_to_retrain = tf_pruning.pruning_process(retrain=True)
            try_count = try_count + 1

            model_params['pretrained_model_path'] = ckpt_to_retrain
            #==== Training with ckpt_to_retrain ====
            task_graph = tf.Graph()
            with task_graph.as_default():
                global_step = tf.Variable(0, name='global_step', trainable=False)

                session_config = tf.ConfigProto()
                session_config.gpu_options.allow_growth = True
                config = (
                    tf.estimator
                    .RunConfig()
                    .replace(
                        session_config=session_config,
                        save_summary_steps=1000,
                        save_checkpoints_secs=None,
                        save_checkpoints_steps=FLAGS.validation_interval,
                        keep_checkpoint_max=1000,
                        log_step_count_steps=1000
                    )
                )

                model = tf.estimator.Estimator(model_fn=model_fn,
                                               model_dir=pruning_output_dir,
                                               config=config,
                                               params=model_params)

                print(
                    ('\n validation data number: {} \n').format(
                        len(list(tf.python_io.tf_record_iterator(FLAGS.validationset_path)))
                    )
                )

                pip = Pipeline()
                model.train(
                    input_fn=lambda: pip.data_pipeline(
                        datasets,
                        params=pipeline_param,
                        batch_size=FLAGS.batch_size
                    ),
                    steps=FLAGS.training_steps,
                    saving_listeners=[
                        EvalCheckpointSaverListener(
                            model,
                            lambda: pip.eval_data_pipeline(
                                FLAGS.validationset_path,
                                params=pipeline_param,
                                batch_size=FLAGS.validation_batch_size
                            )
                        )
                    ]
                )
                print('Training Process Finished.')
            #==== Training End ====
            # get retrained ckpt
            retrained_ckpt_path = tf_pruning.get_retrained_ckpt(pruning_output_dir)
            # analyze the failed rate
            failed_rate = tf_pruning.get_pruning_summary()
            pass




if __name__ == '__main__':
    if not os.path.exists('../logs'):
        os.mkdir('../logs')
    logging.basicConfig(
        filename='../logs/' + FLAGS.output_model_path.split('/')[-1] + '.log',
        level=logging.INFO
    )
    logging.info(
        (
            '--dataset_path={0} \\\n' +
            '--validationset_path={1} \\\n' +
            '--output_model_path={2} \\\n' +
            '--pretrained_model_path={3} \\\n' +
            '--model_type={4} \\\n' +
            '--backbone={5} \\\n' +
            '--loss_fn={6} \\\n' +
            '--layer_depth_multiplier={7} \\\n' +
            '--number_keypoints={8} \\\n' +    
            '--batch_size={9} \\\n' +
            '--pretrained_model={10} \\\n' +
            '--data_augmentation={11} \\\n' +
            '--optimizer={12} \\\n' +
            '--learning_rate={13} \\\n' +
            '--decay_steps={14} \\\n' +
            '--decay_factor={15} \\\n' +
            '--training_steps={16} \\\n' +
            '--validation_interval={17} \\\n' +
            '--validation_batch_size={18} \\\n' +
            '--ohem_top_k={19} \n'
        ).format(
            FLAGS.dataset_path,
            FLAGS.validationset_path,
            FLAGS.output_model_path,
            FLAGS.pretrained_model_path,
            'SEMobilePose',
            'mobilenet_v2',
            'MSE_OHEM',
            FLAGS.layer_depth_multiplier,
            FLAGS.number_keypoints,
            FLAGS.batch_size,
            FLAGS.pretrained_model,
            FLAGS.data_augmentation,
            FLAGS.optimizer,
            FLAGS.learning_rate,
            FLAGS.decay_steps,
            FLAGS.decay_factor,
            FLAGS.training_steps,
            FLAGS.validation_interval,
            FLAGS.validation_batch_size,
            FLAGS.ohem_top_k
        )
    )
    logging.info(
        ('\n validation data number: {} \n').format(
            len(list(tf.python_io.tf_record_iterator(FLAGS.validationset_path)))
        )
    )
    tf.app.run()
