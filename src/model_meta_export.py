"""Export Model Structure in Meta form
For the requirement of pruning filter
We need the meta file that only has model structure but not training node.
Using this meta for generating pruned pb.
"""

import argparse
import os

import tensorflow as tf

from se_mobilepose import SEMobilePose

def meta_export(output_dir):
    with tf.Session() as sess:

        print('Rebuild graph...')
        # The model you define.
        model_arch = SEMobilePose
        model = model_arch(backbone='mobilenet_v2',
                           is_training=False,
                           depth_multiplier=0.5,
                           number_keypoints=17)

        inputs = tf.placeholder(tf.float32,
                                shape=(None, 320, 256, 3),
                                name='input')
        end_points = model.build(inputs)
        sess.run(tf.global_variables_initializer()) 
        saver = tf.train.Saver()

        meta_name = 'model_meta'
        meta_path = os.path.join(output_dir, meta_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            pass
        saver.save(sess, meta_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        default=None, help='Output path.')

    args = parser.parse_args()

    meta_export(args.output_dir)
