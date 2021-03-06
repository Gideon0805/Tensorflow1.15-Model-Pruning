''' Rebuild the checkpoint with input placeholder
After retraining the checkpoint with soft-pruning, 
rebuilding the new checkpoint graph that is without training node for real pruing.
Refer: 
https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/import_meta_graph

'''
import argparse
import os

import tensorflow as tf


def rebuild_ckpt(input_ckpt, input_meta, output_dir):
    """Rebuild the checkpoint for structured pruning.
    Args:
        input_ckpt: Tensorflow 1.x checkpoint file. Retrained checkpoint.
        input_meta: Tensorflow 1.x model structure file. 
                    It is generated from model_meta_export.py.
        output_dir: Diretory that save file.
    Outputs:
        The new checkpoint will be saved in output_dir.
    """
    with tf.Session() as sess:

        print('Rebuild graph...')
        # Create the input tensor.
        # The shape of tensor must be the same as the input of training
        inputs = tf.placeholder(tf.float32,
                                shape=(None, 320, 256, 3),
                                name='input')

        # Load the meta graph, and replace the input to placeholder
        saver = tf.train.import_meta_graph(input_meta, input_map={'MobilenetV2/input': inputs})
        saver.restore(sess, input_ckpt)

        g = tf.get_default_graph()

        ckpt_name = input_ckpt.split('/')[-1]
        ckpt_path = os.path.join(output_dir, ckpt_name)
        saver.save(sess, ckpt_path)

        pbtxt_name = ckpt_name + '.pbtxt'
        tf.train.write_graph(g.as_graph_def(),
                         output_dir,
                         pbtxt_name,
                         True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_ckpt', type=str, required=True,
                        default=None, help='CKPT path.')
    parser.add_argument('-m', '--input_meta', type=str, required=True,
                        default=None, help='Meta path(Must be generated by model).')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        default=None, help='Output path.')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        pass

    rebuild_ckpt(args.input_ckpt, args.input_meta, args.output_dir)
