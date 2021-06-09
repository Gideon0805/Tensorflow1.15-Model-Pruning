# *** Now, we just can pruning the following op type ***
# CONV type: 'Conv2D', 'DepthwiseConv2dNative'
# BIAS type: 'BiasAdd', 'FusedBatchNorm', 'FusedBatchNormV2', 'FusedBatchNormV3'
# =====================================================

"""Generate Pruning variable dictionary.
Get scope_conv_dict, conv_var_dict, and bias_var_dict by input scope_names 
then save into Scope_Conv_Dict.pkl, Conv_Vars_Dict.pkl, and Bias_Vars_Dict.pkl respectively.

Pruning OPs are determined by the above dictionary.

Key and value of dictionary.

scope_conv_dict:
{scope_name: [[conv1 name, pruning dim of conv1], [conv2 name, pruning dim of conv2], ...]}

conv_var_dict 
{conv name: variable name of conv filter weights} 

bias_var_dict 
{conv name: [[variable name of bias or BN], pruning dim]} 

p.s. the variable names could be found by VariableV2 OP in pbtxt.

Examples:
    scope_name = ['MobilenetV2/expanded_conv_1']
    scope_conv_dict = {
        'MobilenetV2/expanded_conv_1/': [
            ['MobilenetV2/expanded_conv_1/expand/Conv2D', 3], 
            ['MobilenetV2/expanded_conv_1/depthwise/depthwise', 2], 
            ['MobilenetV2/expanded_conv_1/project/Conv2D', 2]
        ]
    }

    conv_var_dict = {
        'MobilenetV2/expanded_conv_1/expand/Conv2D': 'MobilenetV2/expanded_conv_1/expand/weights:0',
        'MobilenetV2/expanded_conv_1/depthwise/depthwise': 'MobilenetV2/expanded_conv_1/depthwise/depthwise_weights:0',
        'MobilenetV2/expanded_conv_1/project/Conv2D': 'MobilenetV2/expanded_conv_1/project/weights:0'
    }

    bias_var_dict = {
        'MobilenetV2/expanded_conv_1/expand/Conv2D': [
            [
            'MobilenetV2/expanded_conv_1/expand/BatchNorm/gamma:0', 
            'MobilenetV2/expanded_conv_1/expand/BatchNorm/beta:0'
            ],
            0
        ]
    }
    # So, the pruning process is going to prune the Conv and Bias OP under MobilenetV2/expanded_conv_1.
"""

import numpy as np
import argparse
import os
import pickle

import tensorflow as tf
from tensorflow.contrib import graph_editor as ge


def pruning_get_params(input_ckpt, input_meta, output_dir, scope_names):

    """Generate relevant variable dictionary for pruning.
    Args:
        input_ckpt: Tensorflow 1.x checkpoint file.
        input_meta: Tensorflow 1.x model structure file.
        output_dir: Diretory that save variable dictionaries as .pkl.
        scope_names: The specific scope need to be pruned.
            Ex: scope_names = 'MobilenetV2/expanded_conv_1','MobilenetV2/expanded_conv_2'
            will be split by comma.
    Outputs:
        Scope_Conv_Dict.pkl, Conv_Vars_Dict.pkl, and Bias_Vars_Dict.pkl
        will be saved in output_dir.
    """

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        # Load the meta graph and weights
        saver = tf.train.import_meta_graph(input_meta)
        saver.restore(sess, input_ckpt)

        list_of_ops = ge.make_list_of_op(graph)
        # Find CONV OP
        list_of_conv = [x for x in list_of_ops if x.type == 'Conv2D' or x.type == 'DepthwiseConv2dNative']
        # Find Bias or BN OP
        list_of_bias = [x for x in list_of_ops if x.type == 'BiasAdd' or x.type == 'FusedBatchNorm' or x.type == 'FusedBatchNormV2' or x.type == 'FusedBatchNormV3']

        scope_conv_dict = {}

        # Find the corresponding CONV OP and the dimension to be pruning for the scope name
        for scope_name in scope_names:
            temp = []
            for op in list_of_conv:
                if scope_name in op.name:
                    if op.type == 'DepthwiseConv2dNative':
                        temp.append([op.name, 2])
                    else:
                        temp.append([op.name, 3])
            if temp:
                # Ensure that the output channel remains unchanged, 
                # the last op does not prune the channel dimension.
                if len(temp) >= 2:
                    temp[-1][1] = 2
                scope_conv_dict[scope_name] = temp

        conv_var_dict = {}
        bias_var_dict = {}

        for key, convs in scope_conv_dict.items():
            for conv_name, pruned_dim in convs:
                # Get the variable name by CONV OP that need pruning.
                op = graph.get_operation_by_name(conv_name)
                # Each CONV OP has only one variable.
                # Replace '/read' with ':0', it is for sess.run.
                conv_var = [x.replace('/read', ':0') for x in op.node_def.input if '/read' in x]
                conv_var_dict[conv_name] = conv_var[0]
                # Because CONV OP is the input of BN or Bias, 
                # use conv name to find the variable to be pruned.
                for bias_op in list_of_bias:
                    if conv_name in bias_op.node_def.input:
                        # if OP is BN, there are two variables (gamma, beta)
                        b_vars = [x.replace('/read', ':0') for x in bias_op.node_def.input if '/read' in x]
                        # if OP is CONV and pruning dimension is 3, 
                        # the pruning result doesn't affect bias shape. 
                        if op.type == 'Conv2D' and pruned_dim == 2:
                            # Pruning is not necessary.
                            pruned_b_dim = None
                        else:
                            pruned_b_dim = 0
                        bias_var_dict[conv_name] = [b_vars, pruned_b_dim]
            pass

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            pass

        scope_conv_pkl_path = os.path.join(output_dir, 'Scope_Conv_Dict.pkl')
        conv_pkl_path = os.path.join(output_dir, 'Conv_Vars_Dict.pkl')
        bias_pkl_path = os.path.join(output_dir, 'Bias_Vars_Dict.pkl')
        with open(scope_conv_pkl_path,"wb") as f:
            pickle.dump(scope_conv_dict,f)
        pass
        with open(conv_pkl_path,"wb") as f:
            pickle.dump(conv_var_dict,f)
        pass
        with open(bias_pkl_path,"wb") as f:
            pickle.dump(bias_var_dict,f)
        pass

        print('{}Finished, Var dict Saved in {}{}'.format('='*10, output_dir, '='*10))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ckpt", type=str, required=True,
                        default=None, help='CKPT path.')
    parser.add_argument("--input_meta", type=str, 
                        default=None, help='meta path.')
    parser.add_argument("-o", "--output_dir", type=str, 
                        default='./Pruning_Vars', help='Output dir path.')
    parser.add_argument("-s", "--scope_names", type=str, 
                        default=None, help='Scope that need pruning')

    args = parser.parse_args()

    if args.input_meta is None:
        args.input_meta = args.input_ckpt + '.meta'
        pass
    scope_names = args.scope_names.split(',')

    pruning_get_params(args.input_ckpt, args.input_meta, args.output_dir, scope_names)
