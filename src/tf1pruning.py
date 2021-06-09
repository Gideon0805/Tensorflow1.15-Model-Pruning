# *** Now, we just can pruning the following op type ***
# CONV type: 'Conv2D', 'DepthwiseConv2dNative'
# BIAS type: 'BiasAdd', 'FusedBatchNorm', 'FusedBatchNormV2', 'FusedBatchNormV3'
# =====================================================
"""
Tensorflow 1.15

Soft-Pruning Process:
(Soft pruning is He et al. "Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks".)
1. set_output_dir(self, dir_path=None)
2. get_pruning_params()
3. set_threshold(self, value)
4. get_pruning_filters()
5. get_pruning_ckpt(pruning_filters_path=None)
6. Using the ckpt that return from get_pruning_ckpt() to retrain
7. get_pruning_summary() calculate the failure rate to set retrain to be true or false.
8. pruning_process(retrain)
9. Repeat 2~8 to finish soft-pruning.

After retraining of soft-pruning
Structural Pruning Process:
1. get_rebuild_graph
2. get_strip_params()
3. get_strip_ckpt()
"""

import six
import copy
import tensorflow as tf
import numpy as np
import argparse
import pickle
import os
from tensorflow.contrib import graph_editor as ge


class Pruning(object):
    """Soft-Pruning and Structural Pruned
    Args:
        # For soft-pruning
        input_ckpt_path: The checkpoint for pruning.
        scope_names: The scope list that needs to be pruned.
        scope_conv_dict: The convolution node name under the scope_names.
        conv_vars_dict: The weight variable names of convolution.
        bias_vars_dict: The bias or batch-normalization variable name convolution.
        pruning_filters_dict: Index of the filter that need to be pruned in convolution.
        pruned_ckpt_path: The checkpoint that have done soft-pruning and ready to retrain.
        retrained_ckpt_path: Soft-pruning checkpoint that has be retrained.
        # For structural pruning and inference
        rebuild_meta_graph: Model structure without training node.
        input_tensors_map: Placeholder tensor for inference.
        ckpt_to_strip: The checkpoint from get_rebuild_graph().
        strip_nodes_dict: The node names of convolution that need to be pruned.
        strip_vars_dict: The variable names that need to be pruned and update the variable value.
        output_dir: The directory that save checkpoint.
        output_nodes: Output node is used to freeze graph and export inference model.
        export_ckpt_path: The checkpoint that has been pruned.
        export_pb_path: The inference file.
    Example:
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

        strip_nodes_dict = {
            'MobilenetV2/expanded_conv_1/expand/Conv2D': [
                [
                'MobilenetV2/expanded_conv_1/expand/BatchNorm/gamma:0', 
                'MobilenetV2/expanded_conv_1/expand/BatchNorm/beta:0'
                ], 
                0
            ]
            'MobilenetV2/expanded_conv_1/depthwise/depthwise': [
                [
                'MobilenetV2/expanded_conv_1/depthwise/BatchNorm/gamma:0',
                'MobilenetV2/expanded_conv_1/depthwise/BatchNorm/beta:0'
                ], 
                0
            ]
            'MobilenetV2/expanded_conv_1/project/Conv2D': [
                [
                'MobilenetV2/expanded_conv_1/project/BatchNorm/gamma:0', 
                'MobilenetV2/expanded_conv_1/project/BatchNorm/beta:0'
                ], 
                None
            ]
        }

        strip_vars_dict = {
            'MobilenetV2/expanded_conv_1/': [
                ['MobilenetV2/expanded_conv_1/expand/weights:0', 3], 
                ['MobilenetV2/expanded_conv_1/expand/BatchNorm/gamma:0', 0], 
                ['MobilenetV2/expanded_conv_1/expand/BatchNorm/beta:0', 0], 
                ['MobilenetV2/expanded_conv_1/expand/BatchNorm/moving_mean:0', 0], 
                ['MobilenetV2/expanded_conv_1/expand/BatchNorm/moving_variance:0', 0], 
                ['MobilenetV2/expanded_conv_1/depthwise/depthwise_weights:0', 2], 
                ['MobilenetV2/expanded_conv_1/depthwise/BatchNorm/gamma:0', 0], 
                ['MobilenetV2/expanded_conv_1/depthwise/BatchNorm/beta:0', 0], 
                ['MobilenetV2/expanded_conv_1/depthwise/BatchNorm/moving_mean:0', 0], 
                ['MobilenetV2/expanded_conv_1/depthwise/BatchNorm/moving_variance:0', 0], 
                ['MobilenetV2/expanded_conv_1/project/weights:0', 2]
            ]
        }

        input_tensors_map:
            inputs = tf.placeholder(tf.float32,
                                    shape=(None, 320, 256, 3),
                                    name='input')
            saver = tf.train.import_meta_graph(input_meta, input_map={'MobilenetV2/input': inputs})
            input_tensors_map = {'MobilenetV2/input': inputs}

    """
    def __init__(
        self,
        input_ckpt_path=None,
        scope_names=[],
        scope_conv_dict={},
        conv_vars_dict={}, 
        bias_vars_dict={}, 
        pruning_filters_dict={}, 
        pruned_ckpt_path='', 
        retrained_ckpt_path='', 
        rebuild_meta_graph=None,
        input_tensors_map={},
        ckpt_to_strip='',
        strip_nodes_dict={},
        strip_vars_dict={},
        output_dir=None,
        output_nodes='output',
        export_ckpt_path='', 
        export_pb_path='' ): 
        # Soft-Pruning Params
        if input_ckpt_path is None:
            raise TypeError("The input_ckpt_path is not string!!!")
        self._input_ckpt_path = input_ckpt_path
        self._input_meta_path = input_ckpt_path + '.meta'
        self._scope_names = scope_names
        self._scope_conv_dict = scope_conv_dict
        self._conv_vars_dict = conv_vars_dict
        self._bias_vars_dict = bias_vars_dict
        # Soft-Pruning Checkpoint
        self._threshold = 0
        self._pruning_filters_dict = pruning_filters_dict 
        self._pruned_ckpt_path = pruned_ckpt_path
        # Retraining 
        self._retrained_ckpt_path = retrained_ckpt_path
        self.failed_rate = 0
        # Rebuilding
        self._rebuild_meta_graph = rebuild_meta_graph
        self._input_tensors_map = input_tensors_map
        # Structural Prune
        self._ckpt_to_strip = ckpt_to_strip
        self._strip_nodes_dict = strip_nodes_dict
        self._strip_vars_dict = strip_vars_dict
        # Export
        if output_dir is None:
            self._output_dir = '/'.join(input_ckpt_path.split('/')[:-1]) 
        else:
            self._output_dir = output_dir
        if not os.path.exists(self._output_dir):
            os.mkdir(self._output_dir)
            pass
        self._output_nodes = output_nodes.split(',')
        self._export_ckpt_path = export_ckpt_path
        self._export_pb_path = export_pb_path

    def set_output_dir(self, dir_path=None):
        if dir_path is None:
            raise TypeError("The dir_path is not None!!!")
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self._output_dir = dir_path
        pass

    def get_pruning_params(self):
        """Generate Pruning variable dictionary.
        Get scope_conv_dict, conv_var_dict, and bias_var_dict by input scope_names 
        then save into Scope_Conv_Dict.pkl, Conv_Vars_Dict.pkl, and Bias_Vars_Dict.pkl respectively.

        This function is base on pruning_get_params.py.
        """
        input_ckpt = self._input_ckpt_path
        input_meta = self._input_meta_path
        scope_names = self._scope_names

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:

            # Load the meta graph and weights
            saver = tf.train.import_meta_graph(input_meta)
            saver.restore(sess, input_ckpt)

            list_of_ops = ge.make_list_of_op(graph)
            list_of_conv = [x for x in list_of_ops if x.type == 'Conv2D' or x.type == 'DepthwiseConv2dNative']
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
                    if len(temp) >= 2:
                        temp[-1][1] = 2
                    scope_conv_dict[scope_name] = temp
            conv_vars_dict = {}
            bias_vars_dict = {}

            for key, convs in scope_conv_dict.items():
                for conv_name, pruned_dim in convs:
                    op = graph.get_operation_by_name(conv_name)
                    conv_var = [x.replace('/read', ':0') for x in op.node_def.input if '/read' in x]
                    conv_vars_dict[conv_name] = conv_var[0]
                    for bias_op in list_of_bias:
                        if conv_name in bias_op.node_def.input:
                            b_vars = [x.replace('/read', ':0') for x in bias_op.node_def.input if '/read' in x]
                            if op.type == 'Conv2D' and pruned_dim == 2:
                                pruned_b_dim = None
                            else:
                                pruned_b_dim = 0
                            bias_vars_dict[conv_name] = [b_vars, pruned_b_dim]
                pass

            output_dir = self._output_dir
            dict_dir = os.path.join(output_dir, 'Pruning_Dict')
            if not os.path.exists(dict_dir):
                os.mkdir(dict_dir)
                pass

            scope_conv_pkl_path = os.path.join(dict_dir, 'Scope_Conv_Dict.pkl')
            conv_pkl_path = os.path.join(dict_dir, 'Conv_Vars_Dict.pkl')
            bias_pkl_path = os.path.join(dict_dir, 'Bias_Vars_Dict.pkl')
            self._scope_conv_dict = scope_conv_dict
            self._conv_vars_dict = conv_vars_dict
            self._bias_vars_dict = bias_vars_dict

            with open(scope_conv_pkl_path,"wb") as f:
                pickle.dump(scope_conv_dict,f)
            pass
            with open(conv_pkl_path,"wb") as f:
                pickle.dump(conv_vars_dict,f)
            pass
            with open(bias_pkl_path,"wb") as f:
                pickle.dump(bias_vars_dict,f)
            pass

            print('{}Finished, Var dict Saved in {}{}'.format('='*10, dict_dir, '='*10))
        pass

    def _filters_sparsity_NHWC(self, W):
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

    def _four_mul(self, x):
        """Ensure that the number of pruned filters is a multiple of 4"""
        modulus = x % 4
        if modulus == 0 :
            return x
        y = ((x // 4) + int(modulus / 2)) * 4
        return int(y)

    def set_threshold(self, value):
        """Set the sparsity threshold."""
        self._threshold = value 
        pass

    def get_pruning_filters(self):
        """Generate pruned filter dictionary
        This function is base on pruning_ckpt_byvars.py.
        Make pruned filters dict.
        """
        input_ckpt = self._input_ckpt_path
        input_meta = self._input_meta_path
        threshold = self._threshold
        output_dir = self._output_dir
        # Get vars dict
        scope_conv_dict = self._scope_conv_dict
        conv_vars_dict = self._conv_vars_dict
        bias_vars_dict = self._bias_vars_dict

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(input_meta)
            saver.restore(sess, input_ckpt)

            variables = tf.get_collection(tf.GraphKeys.VARIABLES)
            variables_name = [x.name for x in variables]

            pruning_filters_dict = {}
            for scope, convs in scope_conv_dict.items():
                conv_op = convs[0][0]
                conv_var_name = conv_vars_dict[conv_op]
                conv_weights = sess.run(variables[variables_name.index(conv_var_name)])

                sparsity_list = []
                filter_list = []
                filter_pruned_num = 0
                filter_total_num = conv_weights.shape[3]

                sparsity_list = self._filters_sparsity_NHWC(conv_weights)
                sparsity_list.sort(key=lambda a:a[1], reverse=True)
                filter_list = [x[0] for x in sparsity_list if x[1]>threshold]
                filter_pruned_num = self._four_mul(len(filter_list))
                if filter_pruned_num < len(sparsity_list):
                    temp = sparsity_list[0:filter_pruned_num]
                    filter_list = [x[0] for x in temp]
                filter_list.sort()
                if filter_list:
                    pruning_filters_dict[scope] = filter_list
                    pass
                pass
            print('{}Generated Pruned Filter Lists.{}'.format('='*10, '='*10))
            self._pruning_filters_dict = pruning_filters_dict

            output_dir = self._output_dir
            dict_dir = os.path.join(output_dir, 'Pruning_Dict')
            if not os.path.exists(dict_dir):
                os.mkdir(dict_dir)
                pass

            pkl_path = os.path.join(dict_dir, 'Pruned_Filters.pkl')
            with open(pkl_path,"wb") as f:
                pickle.dump(pruning_filters_dict,f)
            pass

        pass

    def get_pruning_ckpt(self, pruning_filters_path=None):
        """Generate soft-pruning checkpoint
        This function is base on pruning_ckpt_byvars.py.

        Args:
            pruning_filters_path: The pruned filter dictionary, if exist, will prune by this.
        Returns:
            ckpt_path: The soft-pruning checkpoint, and will be retrain.
        """
        input_ckpt = self._input_ckpt_path
        input_meta = input_ckpt + '.meta'
        output_dir = self._output_dir
        threshold = self._threshold
        # These is related to soft-pruning variables.
        scope_conv_dict = self._scope_conv_dict
        conv_vars_dict = self._conv_vars_dict
        bias_vars_dict = self._bias_vars_dict
        if pruning_filters_path is not None:
            with open(pruning_filters_path, 'rb') as f:
                pruning_filters_dict = pickle.load(f)
            pass
        else:
            pruning_filters_dict = self._pruning_filters_dict

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(input_meta)
            saver.restore(sess, input_ckpt)

            variables = tf.get_collection(tf.GraphKeys.VARIABLES)
            variables_name = [x.name for x in variables]

            pruned_filters_summary = {}

            for scope, filter_list in pruning_filters_dict.items() :

                conv_set = scope_conv_dict[scope][0] # conv_set: [conv_op, pruned_dim]
                conv_var_name = conv_vars_dict[conv_set[0]]
                conv_weights = sess.run(variables[variables_name.index(conv_var_name)])

                filter_pruned_num = len(filter_list)
                filter_total_num = conv_weights.shape[conv_set[1]]
                filters_summary = 'Pruned filters/ Total filters: {}/{}'.format(filter_pruned_num, filter_total_num)
                pruned_filters_summary[scope] = filters_summary

                # Prune variables by Scope with soft-pruning
                for conv_op, pruned_dim in scope_conv_dict[scope]:
                    conv_var_name = conv_vars_dict[conv_op]
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

                    b_var_names = bias_vars_dict[conv_op][0]
                    b_pruned_dim = bias_vars_dict[conv_op][1]
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

            pruned_ckpt_dir = os.path.join(output_dir, 'Pruned_Dir')
            if not os.path.exists(pruned_ckpt_dir):
                os.mkdir(pruned_ckpt_dir)
                pass
            ckpt_name = input_ckpt.split('/')[-1]
            ckpt_path = os.path.join(pruned_ckpt_dir, ckpt_name)
            self._pruned_ckpt_path = ckpt_path
            saver.save(sess, ckpt_path)
            print('{}Finished, Saved in {}{}'.format('='*10, ckpt_path, '='*10))

            txt_path = os.path.join(pruned_ckpt_dir, 'Pruned_Filters.txt')
            with open(txt_path,"w") as f:
                f.write('Input ckpt is {}\n'.format(input_ckpt))
                f.write('Pruning ckpt is {}\n'.format(ckpt_path))
                if pruning_filters_path:
                    f.write('Pruning dict is {}\n'.format(pruning_filters_path))
                else:
                    f.write('Pruning sparsity threshold: {}\n'.format(threshold))
                for key, value in pruning_filters_dict.items():
                    f.write('{} {}\n'.format(key, pruned_filters_summary[key]))
                for key, value in pruning_filters_dict.items():
                    f.write('{} : {}\n'.format(key, value))
        return ckpt_path
        pass

    def get_retrained_ckpt(self, retrain_path):
        self._retrained_ckpt_path = tf.train.latest_checkpoint(retrain_path)
        self._input_ckpt_path = self._retrained_ckpt_path
        return self._retrained_ckpt_path
        pass

    def get_pruning_summary(self):
        """Get the pruning result of retrained checkpoint.
        
        Calculate if all the weights in the pruning filter is zero 
        to define the success or fail of soft-pruning.  

        Returns:
            failed_rate: total_failed / total_pruned
        """
        input_ckpt = self._retrained_ckpt_path
        input_meta = input_ckpt + '.meta'
        # Get pruning filter and the related variable names.
        pruning_filters_dict = self._pruning_filters_dict
        scope_conv_dict = self._scope_conv_dict
        conv_vars_dict = self._conv_vars_dict
        bias_vars_dict = self._bias_vars_dict

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(input_meta)
            saver.restore(sess, input_ckpt)

            graph = tf.get_default_graph()

            variables = tf.get_collection(tf.GraphKeys.VARIABLES)
            variables_name = [x.name for x in variables]

            # Check pruned filters
            pruned_result = {}
            failed_result = {}
            total_failed = 0
            total_pruned = 0
            for scope, filter_list in pruning_filters_dict.items() :

                pruned_result[scope] = filter_list
                for conv_op, pruned_dim in scope_conv_dict[scope]:
                    # Conv Pruned Filter Account
                    conv_var_name = conv_vars_dict[conv_op]
                    # conv_weights = variables[variables_name.index(conv_var_name)]
                    w_values = sess.run(conv_var_name)
                    w_abs = np.abs(w_values)
                    conv_pruned_filter_list = []
                    conv_failed_count = 0
                    if pruned_dim == 2:
                        w_sum = np.sum(w_abs, axis=(0,1,3))
                    elif pruned_dim == 3:
                        w_sum = np.sum(w_abs, axis=(0,1,2))
                    else:
                        raise IndexError("The pruning dim is out of range, it should be 2 or 3!!!")
                    for filter_i in filter_list:
                        if w_sum[filter_i] < 0.000000001:
                            conv_pruned_filter_list.append('1')
                        else:
                            conv_pruned_filter_list.append('failed')
                            conv_failed_count = conv_failed_count + 1
                            pass
                    pruned_result[conv_var_name] = conv_pruned_filter_list
                    failed_result[conv_var_name] = '{}/{}'.format(conv_failed_count, len(filter_list))
                    total_failed = total_failed + conv_failed_count
                    total_pruned = total_pruned + len(filter_list)

                    # Bias/BN Pruned Filter Account
                    b_var_names = bias_vars_dict[conv_op][0]
                    b_pruned_dim = bias_vars_dict[conv_op][1]
                    if b_pruned_dim is not None:
                        for b_var_name in b_var_names:
                            b_values = sess.run(b_var_name)
                            b_abs = np.abs(b_values)
                            b_pruned_list = []
                            for i in filter_list:
                                if b_abs[i] < 0.000000001:
                                    b_pruned_list.append('1')
                                else:
                                    b_pruned_list.append('failed')
                                pass
                            pruned_result[b_var_name] = b_pruned_list
                            pass
            self.failed_rate = total_failed / total_pruned

            txt_path = '/'.join(input_ckpt.split('/')[:-1]) + '/Pruning_Summary.txt'
            with open(txt_path,"w") as f:
                for key, value in failed_result.items():
                    f.write('{} : {}\n'.format(key, value))
                for key, value in pruned_result.items():
                    f.write('{} : {}\n'.format(key, value))
        print('{} Pruning Summary Done. {}'.format('='*15, '='*15))
        return self.failed_rate
        pass

    def pruning_process(self, retrain=False):
        """Determine the pruninig process.
        If the failed_rate from get_pruning_summary is higher than we specify 
        then do the same soft-pruning again, otherwise update the pruning_filters_dict 
        and make more filter be soft-pruning.
        # Retraining, when the failure rate is higher than the specific value.
            Args:
                retrain: If false, regenerate Conv_Vars_Dict.pkl, Bias_Vars_Dict.pkl, and Pruned_Filters.pkl.
                Otherwise, those pkl will not chage.
            Returns:
                Return soft-pruning ckpt.
        """
        if retrain:
            return self.get_pruning_ckpt()
        else:
            self.get_pruning_params()
            self.get_pruning_filters()
            return self.get_pruning_ckpt()
            pass

    def get_rebuild_graph(self):
        """Rebuild the checkpoint for structured pruning.
        meta_graph: Tensorflow 1.x model structure file. 
                    It is generated from model_meta_export.py.
        """
        input_ckpt = self._retrained_ckpt_path
        meta_graph = self._rebuild_meta_graph
        input_tensors_map = self._input_tensors_map
        output_dir = self._output_dir

        with tf.Session() as sess:
            print('{} Rebuilding Graph... {}'.format('='*15, '='*15))

            saver = tf.train.import_meta_graph(meta_graph, input_map=input_tensors_map)
            saver.restore(sess, input_ckpt)
            ckpt_name = input_ckpt.split('/')[-1]
            rebuild_dir = os.path.join(output_dir, 'Rebuild_CKPT')
            if not os.path.exists(rebuild_dir):
                os.mkdir(rebuild_dir)
                pass
            ckpt_path = os.path.join(rebuild_dir, ckpt_name)
            saver.save(sess, ckpt_path)

            pbtxt_name = ckpt_name + '.pbtxt'
            tf.train.write_graph(sess.graph.as_graph_def(),
                             rebuild_dir,
                             pbtxt_name,
                             True)
        self._ckpt_to_strip = ckpt_path

    def get_strip_params(self):
        """Prepare the dictionary of operation nodes and variables for pruning.
        This function is base on strip_get_params.py.
        *** In tensorflow 1.x, the operation(OP) in node is include the info of node_def, OP type, 
        *** and inputs of node, that will help us to find the variables name.
        
        Generate Strip_Nodes_Dict.pkl and Strip_Vars_Dict.pkl.
        
        Examples:
            scope_name = ['MobilenetV2/expanded_conv_1']
            strip_nodes_dict = {
                'MobilenetV2/expanded_conv_1/expand/Conv2D': [
                    [
                    'MobilenetV2/expanded_conv_1/expand/BatchNorm/gamma:0', 
                    'MobilenetV2/expanded_conv_1/expand/BatchNorm/beta:0'
                    ], 
                    0
                ]
                'MobilenetV2/expanded_conv_1/depthwise/depthwise': [
                    [
                    'MobilenetV2/expanded_conv_1/depthwise/BatchNorm/gamma:0',
                    'MobilenetV2/expanded_conv_1/depthwise/BatchNorm/beta:0'
                    ], 
                    0
                ]
                'MobilenetV2/expanded_conv_1/project/Conv2D': [
                    [
                    'MobilenetV2/expanded_conv_1/project/BatchNorm/gamma:0', 
                    'MobilenetV2/expanded_conv_1/project/BatchNorm/beta:0'
                    ], 
                    None
                ]
            }

            strip_vars_dict = {
                'MobilenetV2/expanded_conv_1/': [
                    ['MobilenetV2/expanded_conv_1/expand/weights:0', 3], 
                    ['MobilenetV2/expanded_conv_1/expand/BatchNorm/gamma:0', 0], 
                    ['MobilenetV2/expanded_conv_1/expand/BatchNorm/beta:0', 0], 
                    ['MobilenetV2/expanded_conv_1/expand/BatchNorm/moving_mean:0', 0], 
                    ['MobilenetV2/expanded_conv_1/expand/BatchNorm/moving_variance:0', 0], 
                    ['MobilenetV2/expanded_conv_1/depthwise/depthwise_weights:0', 2], 
                    ['MobilenetV2/expanded_conv_1/depthwise/BatchNorm/gamma:0', 0], 
                    ['MobilenetV2/expanded_conv_1/depthwise/BatchNorm/beta:0', 0], 
                    ['MobilenetV2/expanded_conv_1/depthwise/BatchNorm/moving_mean:0', 0], 
                    ['MobilenetV2/expanded_conv_1/depthwise/BatchNorm/moving_variance:0', 0], 
                    ['MobilenetV2/expanded_conv_1/project/weights:0', 2]
                ]
            }

        """
        # Using the new checkpoint that is generated by get_rebuild_graph() and without training node.
        input_ckpt = self._ckpt_to_strip
        input_meta = input_ckpt + '.meta'
        output_dir = self._output_dir
        # Read the key and value in Scope_Conv_Dict.pkl to find the nodes of convolution,
        # bias, and batch-normalization and their variable names.
        scope_conv_dict = self._scope_conv_dict

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:

            saver = tf.train.import_meta_graph(input_meta)
            saver.restore(sess, input_ckpt)

            list_of_ops = ge.make_list_of_op(graph)
            list_of_bias = [x for x in list_of_ops if x.type == 'BiasAdd' or x.type == 'FusedBatchNorm' or x.type == 'FusedBatchNormV2' or x.type == 'FusedBatchNormV3']

            strip_nodes_dict = {}
            strip_vars_dict = {}
            for scope, conv_sets in scope_conv_dict.items():
                node_name_dim = []
                var_name_dim = []
                for conv_name, conv_dim in conv_sets:
                    conv_op = graph.get_operation_by_name(conv_name)
                    conv_var = [x.replace('/read', ':0') for x in conv_op.node_def.input if '/read' in x]
                    var_name_dim.append([conv_var[0], conv_dim])

                    conv_assign = [x.replace('/read', '/Assign') for x in conv_op.node_def.input if '/read' in x]
                    assign_name = conv_assign[0]
                    assign_op = graph.get_operation_by_name(assign_name)
                    for i_name in assign_op.node_def.input:
                        op = graph.get_operation_by_name(i_name)
                        conv_var_name = i_name + '/shape' if op.type == 'Add' else i_name
                        node_name_dim.append([conv_var_name, conv_dim])

                    if conv_op.type == 'Conv2D' and conv_dim == 2:
                        continue
                    for bias_op in list_of_bias:
                        if conv_name in bias_op.node_def.input:
                            b_assign = [x.replace('/read', '/Assign') for x in bias_op.node_def.input if '/read' in x]
                            for name in b_assign:
                                assign_op = graph.get_operation_by_name(name)
                                for b_i_name in assign_op.node_def.input:
                                    node_name_dim.append([b_i_name, 0])

                            b_vars = [x.replace('/read', ':0') for x in bias_op.node_def.input if '/read' in x]
                            pruned_b_dim = 0
                            for b_var in b_vars:
                                var_name_dim.append([b_var, pruned_b_dim])
                                pass

                strip_nodes_dict[scope] = node_name_dim
                strip_vars_dict[scope] = var_name_dim

            dict_dir = os.path.join(output_dir, 'Pruning_Dict')
            if not os.path.exists(dict_dir):
                os.mkdir(dict_dir)
                pass

            strip_nodes_dict_path = os.path.join(dict_dir, 'Strip_Nodes_Dict.pkl')
            strip_vars_dict_path = os.path.join(dict_dir, 'Strip_Vars_Dict.pkl')
            with open(strip_nodes_dict_path,"wb") as f:
                pickle.dump(strip_nodes_dict,f)
                self._strip_nodes_dict = strip_nodes_dict
                print('{} Strip Nodes Dict Dumped. {}'.format('='*15, '='*15))
            with open(strip_vars_dict_path,"wb") as f:
                pickle.dump(strip_vars_dict,f)
                self._strip_vars_dict = strip_vars_dict
                print('{} Strip Vars Dict Dumped. {}'.format('='*15, '='*15))
            pass
        pass

    def _constDimSize(self, op, filter_nums, dim_i):
        """Modify the dim of node, which the op type is "Const".
        This function is to modify the dimension size in node_def of tf.Operation.

        Example:
        node {
          name: "MobilenetV2/expanded_conv_1/expand/BatchNorm/gamma/Initializer/ones"
          op: "Const"
          ...
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_FLOAT
                tensor_shape {
                  dim {
                    size: 48
                  }
                }
                float_val: 1.0
              }
            }
          }
        }

        Args:
            op: tf.Operation in tensorflow 1.x.
            filter_nums: The amount of pruning filter.
            dim_i: The pruning dimension. The NCHW format is used here.
        """
        node = op.node_def
        node_value = node.attr['value']
        update_value = node_value.tensor.tensor_shape.dim[dim_i].size - filter_nums
        node_value.tensor.tensor_shape.dim[dim_i].size = update_value
        op._set_attr('value', node.attr['value'])    

    def _constTensorContent(self, op, filter_nums, dim_i):
        """Modify the tensor_content of node, which the op type is "Const".
        This function is to modify the tensor_content value of node_def in tf.Operation.
        The tensor_content usually represents the convolution shape and is buffer type.

        Example:
        node {
          name: "MobilenetV2/expanded_conv_1/expand/weights/Initializer/truncated_normal/shape"
          op: "Const"
          ...
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_INT32
                tensor_shape {
                  dim {
                    size: 4
                  }
                }
                tensor_content: "\001\000\000\000\001\000\000\000\010\000\000\0000\000\000\000"
              }
            }
          }
        }

        Args:
            op: tf.Operation in tensorflow 1.x.
            filter_nums: The amount of pruning filter.
            dim_i: The pruning dimension. The NCHW format is used here.
        """
        node = op.node_def
        T = node.attr['value'].tensor
        tempDtype = tf.dtypes.DType(T.dtype).as_numpy_dtype
        # T_content = tf.decode_raw(T.tensor_content, T.dtype)
        temp_shape = np.frombuffer(T.tensor_content, dtype=tempDtype)
        new_shape = np.array(temp_shape)
        # print(new_shape)
        new_shape[dim_i] = new_shape[dim_i] - filter_nums
        temp_content = np.array(new_shape, dtype=np.dtype(tf.dtypes.DType(T.dtype).as_numpy_dtype))
        T.tensor_content = temp_content.tobytes()
        op._set_attr('value', node.attr['value'])

    def _varDimSize(self, op, filter_nums, dim_i):
        """Modify the dim of node, which the op type is "VariableV2".
        This function is to modify the shape of filter weight in node_def.

        Example:
        node {
          name: "MobilenetV2/expanded_conv_1/expand/weights"
          op: "VariableV2"
          ...
          attr {
            key: "shape"
            value {
              shape {
                dim {
                  size: 1
                }
                dim {
                  size: 1
                }
                dim {
                  size: 8
                }
                dim {
                  size: 48
                }
              }
            }
          }
          ...
        }

        Args:
            op: tf.Operation in tensorflow 1.x.
            filter_nums: The amount of pruning filter.
            dim_i: The pruning dimension. The NCHW format is used here.
        """
        node = op.node_def
        node_value = node.attr['shape']
        update_value = node_value.shape.dim[dim_i].size - filter_nums
        node_value.shape.dim[dim_i].size = update_value
        op._set_attr('shape', node.attr['shape'])

    def get_strip_ckpt(self):
        """Generate pruned checkpoint and inference file.
        This function is base on strip_ckpt_byvars.py.
        """
        # Using the new checkpoint that is generated by get_rebuild_graph() and without training node.
        input_ckpt = self._ckpt_to_strip
        input_meta = input_ckpt + '.meta'
        output_dir = self._output_dir
        # Get the related pruning dictionary
        scope_conv_dict = self._scope_conv_dict
        scope_pruned_filters= self._pruning_filters_dict
        strip_nodes_dict = self._strip_nodes_dict
        strip_vars_dict = self._strip_vars_dict

        graph = tf.Graph()
        new_graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            saver = tf.train.import_meta_graph(input_meta)
            saver.restore(sess, input_ckpt)

            print('{} Storing variables... {}'.format('='*15, '='*15))
            weights_dict = {}
            for key in graph.get_all_collection_keys():
                if 'variable' in key:
                    for variable in graph.get_collection(key):
                        weights_dict[variable.name] = variable.eval(sess)
                        pass
                    pass
            pkl_path = '/'.join(input_ckpt.split('/')[:-1]) + '/' +'Model_weights.pkl'
            print('{} Dumping weights... {}'.format('='*15, '='*15))
            with open(pkl_path,"wb") as f:
                pickle.dump(weights_dict,f)

            print('{} Modifying Graph and Exporting GraphDef... {}'.format('='*15, '='*15))
            for scope, filters in scope_pruned_filters.items():
                pruned_nums = len(filters)
                for strip_node, pruned_dim in strip_nodes_dict[scope]:
                    op = graph.get_operation_by_name(strip_node)
                    if op.type == 'VariableV2':
                        self._varDimSize(op, pruned_nums, pruned_dim)
                        pass
                    else:
                        if 'shape' in op.name:
                            self._constTensorContent(op, pruned_nums, pruned_dim)
                        else:
                            self._constDimSize(op, pruned_nums, pruned_dim)
                            pass

            print('{} Graph Modified. {}'.format('='*15, '='*15))
            new_graph_def = graph.as_graph_def()
            print('{} Export Graph to New GraphDef. {}'.format('='*15, '='*15))

        # Import New GraphDef to New Graph ===
        print('{} Importing New GraphDef to New Graph and Updating weights... {}'.format('='*15, '='*15))
        tf.reset_default_graph()
        with tf.Session(graph=new_graph) as new_sess:
            tf.import_graph_def(new_graph_def, name='')
            # ==== Add Variables key in New Graph ====
            g = tf.get_default_graph()
            for key in graph.get_all_collection_keys():
                if 'variable' in key:
                    for variable in graph.get_collection(key):
                        temp = tf.Variable(initial_value=variable.value())
                        temp_v = temp.from_proto(variable.to_proto())
                        g.add_to_collection(key, temp_v)
                else:
                    for variable in graph.get_collection(key):
                        g.add_to_collection(key, variable)
                        pass
                    pass

            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            variables_name = [x.name for x in variables]

            for scope, filters in scope_pruned_filters.items():
                for var_name, dim in strip_vars_dict[scope]:
                    variable = variables[variables_name.index(var_name)]
                    prev_weights = weights_dict.pop(var_name)
                    pruned_weights = np.delete(prev_weights, filters, axis=dim)
                    variable.load(pruned_weights, new_sess)
                    new_sess.run(variable)
                pass

            for key, value in weights_dict.items():
                temp = value
                variable = variables[variables_name.index(key)]
                variable.load(temp, new_sess)
                new_sess.run(variable)
                pass

            # After updating, we can save the nwe graph as checkpoint.
            strip_dir = os.path.join(output_dir, 'Stripped_Dir')
            if not os.path.exists(strip_dir):
                os.mkdir(strip_dir)
                pass
            ckpt_name = input_ckpt.split('/')[-1]
            ckpt_path = os.path.join(strip_dir, ckpt_name)
            saver.save(new_sess, ckpt_path)
            # Save inference model
            print('{} Export .pb {}'.format('='*15, '='*15))
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                new_sess, new_graph.as_graph_def(), self._output_nodes)
            tf.train.write_graph(frozen_graph,
                                 strip_dir,
                                 'modify.pb',
                                 False)
            print('{} modify.pb has been saved in {} {}'.format('='*15, strip_dir, '='*15))
            print('{} Fininshed. {}'.format('='*15, '='*15))
        pass


        