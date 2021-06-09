"""Load Dictionaries and Prune the variables.

Read Scope_Conv_Dict.pkl, Conv_Vars_Dict.pkl, and Bias_Vars_Dict.pkl, genearted from pruning_get_params.py.

The pruning process:

First, according to the scope name in Scope_Conv_Dict.pkl, 
calculating the sparsity of each filter in convloution layer under the scope.
(The filter sparsity comes from "Computation-Performance Optimization of Convolutional Neural Networks with Redundant Kernel Removal" Liu et al.)

Second, if the sparsity of filter  is greater than the threshold we set, 
then that filter in convolution layer will be add in a list as index of pruning.
The index of pruning is pruned_filters_dict in here.

Third, get the variable names from Conv_Vars_Dict.pkl and Bias_Vars_Dict.pkl, 
and base on the index of pruned_filters_dict, 
using sess.run() to update variable values of the specific filter.

Finally, save the checkpoint with new variable values as new checkpoint to retrain 
and pruned_filters_dict as Pruned_Filters.pkl.

After these steps, we are only doing soft prunig, 
still need to retrain and pruned the zero-value filter to achive the structured-pruning
(Soft pruning is He et al. "Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks".)

"""
import tensorflow as tf
import numpy as np
import argparse
import os
import pickle
from tensorflow.contrib import graph_editor as ge


def filters_sparsity_NHWC(W):
    """ Calculate the sparsity under NHWC format.
    Args:
        W: The all filter weights of convolution layer.
    Returns:
        s_list: The list of filter sparsity.
    """
    W = np.abs(W)
    temp_mean = np.mean(W)
    s_list = [] # (filter, sparsity) list
    for filter_i in range(W.shape[3]):
        s_count = 0 # account the nums of sparse weight in filter
        for ch_i in range(W.shape[2]):
            x = W[0,0,ch_i,filter_i]
            # |x| < Mean
            # sparse sum
            if x < temp_mean:
                s_count = s_count + 1
            pass
        # sparsity
        sparsity = s_count / W.shape[2]
        s_list.append((filter_i, sparsity))
    pass
    return s_list


def four_mul(x):
    """Ensure that the number of pruned filters is a multiple of 4"""
    modulus = x % 4
    if modulus == 0 :
        return x
    y = ((x // 4) + int(modulus / 2)) * 4
    return int(y)


def pruning_ckpt(ckpt, meta, pruned_dict
                 pkl_dir, threshold, output_dir):

    """Generate pruned filter dictionary and soft-pruning checkpoint.
    Args:
        ckpt: Tensorflow 1.x checkpoint file.
        meta: Tensorflow 1.x model structure file.
        pruned_dict: The pruned filter dictionary, if exist, will prune by this.
        pkl_dir: The directory, store Scope_Conv_Dict, Conv_Vars_Dict, and Bias_Vars_Dict.
        output_dir: Diretory that save dictionary and checkpoint.
        threshold: The sparsity threshold for pruning.
    Outputs:
        Pruned_Filters.pkl and checkpoint will be saved in output_dir.
    """
    scope_conv_path = os.path.join(pkl_dir, 'Scope_Conv_Dict.pkl')
    conv_var_path = os.path.join(pkl_dir, 'Conv_Vars_Dict.pkl')
    bias_var_path = os.path.join(pkl_dir, 'Bias_Vars_Dict.pkl')

    with tf.Session() as sess:

        # Load the meta graph and weights
        saver = tf.train.import_meta_graph(meta)
        saver.restore(sess, ckpt)

        variables = tf.get_collection(tf.GraphKeys.VARIABLES)
        variables_name = [x.name for x in variables]

        # Get vars dict
        with open(scope_conv_path, 'rb') as f:
            scope_conv_dict = pickle.load(f)
        with open(conv_var_path, 'rb') as f:
            conv_var_dict = pickle.load(f)
        with open(bias_var_path, 'rb') as f:
            bias_var_dict = pickle.load(f)

        pruned_filters_dict = {}
        if pruned_dict:
            with open(pruned_dict, 'rb') as f:
                pruned_filters_dict = pickle.load(f)
            pass
        else:
            # The first and second step
            # Make pruned_filters_dict by scope_conv_dict
            # Determine the filters that need to be pruned, by threshold
            for scope, convs in scope_conv_dict.items():
                conv_op = convs[0][0]
                conv_var_name = conv_var_dict[conv_op]
                conv_weights = sess.run(variables[variables_name.index(conv_var_name)])

                sparsity_list = []
                filter_list = []
                filter_pruned_num = 0
                filter_total_num = conv_weights.shape[3]

                # Make the list of filter sparsity
                sparsity_list = filters_sparsity_NHWC(conv_weights)
                sparsity_list.sort(key=lambda a:a[1], reverse=True)
                filter_list = [x[0] for x in sparsity_list if x[1]>threshold]
                filter_pruned_num = four_mul(len(filter_list))
                if filter_pruned_num < len(sparsity_list):
                    temp = sparsity_list[0:filter_pruned_num]
                    filter_list = [x[0] for x in temp]
                filter_list.sort()
                # Save the index of pruning filters in dict by scope
                if filter_list:
                    pruned_filters_dict[scope] = filter_list
                    pass
                pass
        print('{}Generated Pruned Filter Lists.{}'.format('='*10, '='*10))

        pruned_filters_summary = {}

        for scope, filter_list in pruned_filters_dict.items() :

            # Get Pruning summary 
            conv_set = scope_conv_dict[scope][0]
            conv_var_name = conv_var_dict[conv_set[0]]
            conv_weights = sess.run(variables[variables_name.index(conv_var_name)])

            filter_pruned_num = len(filter_list)
            filter_total_num = conv_weights.shape[conv_set[1]]
            filters_summary = 'Pruned filters/ Total filters: {}/{}'.format(filter_pruned_num, filter_total_num)
            pruned_filters_summary[scope] = filters_summary

            # Prune variables by Scope with soft-pruning
            for conv_op, pruned_dim in scope_conv_dict[scope]:
                # Set the weight of filter to be 0
                conv_var_name = conv_var_dict[conv_op]
                conv_weights = variables[variables_name.index(conv_var_name)]
                w_values = sess.run(conv_var_name)
                for filter_i in filter_list:
                    if pruned_dim == 2:
                        w_values[:, :, filter_i, :] = 0
                    elif pruned_dim == 3:
                        w_values[:, :, :, filter_i] = 0
                        pass
                conv_weights.load(w_values, sess)
                sess.run(conv_weights)

                # Set Bias or BN to be 0
                b_var_names = bias_var_dict[conv_op][0]
                b_pruned_dim = bias_var_dict[conv_op][1]
                if b_pruned_dim is not None:
                    for var_name in b_var_names:
                        b_weights = variables[variables_name.index(var_name)]
                        b_values = sess.run(var_name)
                        for filter_i in filter_list:
                            b_values[filter_i] = 0
                            pass
                        b_weights.load(b_values, sess)
                        sess.run(b_weights)
                    pass
            print('{}{} Done{}'.format('-'*5, scope, '-'*5))

        saver = tf.train.Saver()
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            pass
        ckpt_name = ckpt.split('/')[-1]
        ckpt_path = os.path.join(output_dir, ckpt_name)
        saver.save(sess, ckpt_path)
        print('{}Finished, Saved in {}{}'.format('='*10, ckpt_path, '='*10))

        pkl_path = os.path.join(output_dir, 'Pruned_Filters.pkl')
        with open(pkl_path,"wb") as f:
            pickle.dump(pruned_filters_dict,f)
        pass

        # Save the pruning info into Pruned_Filters.txt
        txt_path = pkl_path.replace('.pkl', '.txt')
        with open(txt_path,"w") as f:
            f.write('Pruning ckpt is {}\n'.format(ckpt))
            if pruned_dict:
                f.write('Pruning dict is {}\n'.format(pruned_dict))
            else:
                f.write('Pruning sparsity threshold: {}\n'.format(threshold))
            for key, value in pruned_filters_dict.items():
                f.write('{} {}\n'.format(key, pruned_filters_summary[key]))
            for key, value in pruned_filters_dict.items():
                f.write('{} : {}\n'.format(key, value))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_ckpt', type=str, required=True,
                        default=None, help='CKPT path.')
    parser.add_argument('--input_meta', type=str, 
                        default=None, help='meta path.')
    parser.add_argument('--pruned_dict', type=str, 
                        default=None, help='Pruned DICT path.')
    parser.add_argument('--pkl_dir', type=str, required=True,
                        default=None, help='The diretory store pkl.')
    parser.add_argument('-o','--output_dir', type=str, 
                        default=None, help='Output dir path.')
    parser.add_argument('--threshold', type=str, 
                        default=None, help='Sparsity threshold.')

    args = parser.parse_args()

    if args.input_meta is None:
        args.input_meta = args.input_ckpt + '.meta'
        pass

    pruning_ckpt(args.input_ckpt, args.input_meta, args.pruned_dict
                 args.pkl_dir, args.threshold, args.output_dir)
