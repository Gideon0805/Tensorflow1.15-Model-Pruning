"""Implement the structural pruning

After retraining the soft-pruning model, we need to trim the filter that is full of zero-value.
Cause the filter number is defined by the value of node in pbtxt, we need to modify the value to achieve pruning.

First, using the rebuild_ckpt_export to remove the training node in the retrained checkpoint
and save as a new checkpoint, called strip_ckpt.
Second, read Scope_Conv_Dict.pkl, Pruned_Filters.pkl, Strip_Nodes_Dict.pkl, and Strip_Vars_Dict.pkl.
Using these to trim the filters that we set to be zero in retraining.
Third, define a new graph with modified node and update the variable, include tensors shape and weight.
Then the graph will be a pruned model, and then export the inference model(pb file).


Stripping process:
1. Restore checkpoint to tf.Graph
2. variable.eval(sess) for storing weights as DICT
3. Modify node in tf.Graph
4. Export modified graph to tf.GraphDef as new graphdef
5. Import new graphdef into new graph
6. Update new graph with the weigths_dict, we store.
7. Export pb file

"""
import six
import copy
import tensorflow as tf
import numpy as np
import argparse
import pickle
import os
from tensorflow.contrib import graph_editor as ge

# Method 1
def constDimSize(op, filter_nums, dim_i):
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

def constTensorContent(op, filter_nums, dim_i):
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
    temp_shape = np.frombuffer(T.tensor_content, dtype=tempDtype)
    new_shape = np.array(temp_shape)
    new_shape[dim_i] = new_shape[dim_i] - filter_nums
    temp_content = np.array(new_shape, dtype=np.dtype(tf.dtypes.DType(T.dtype).as_numpy_dtype))
    T.tensor_content = temp_content.tobytes()
    op._set_attr('value', node.attr['value'])


def varDimSize(op, filter_nums, dim_i):
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

# Method 2
# def constDimSize(node, filter_nums, dim_i):
#     node_value = node.attr['value']
#     update_value = node_value.tensor.tensor_shape.dim[dim_i].size - filter_nums
#     node_value.tensor.tensor_shape.dim[dim_i].size = update_value

# def constTensorContent(node, filter_nums, dim_i):
#     T = node.attr['value'].tensor
#     tempDtype = tf.dtypes.DType(T.dtype).as_numpy_dtype
#     # T_content = tf.decode_raw(T.tensor_content, T.dtype)
#     temp_shape = np.frombuffer(T.tensor_content, dtype=tempDtype)
#     new_shape = np.array(temp_shape)
#     # print(new_shape)
#     new_shape[dim_i] = new_shape[dim_i] - filter_nums
#     temp_content = np.array(new_shape, dtype=np.dtype(tf.dtypes.DType(T.dtype).as_numpy_dtype))
#     T.tensor_content = temp_content.tobytes()

# def varDimSize(node, filter_nums, dim_i):
#     node_value = node.attr['shape']
#     update_value = node_value.shape.dim[dim_i].size - filter_nums
#     node_value.shape.dim[dim_i].size = update_value


def strip_ckpt(input_ckpt, input_meta, pkl_dir, pruned_dict_path):
    """Generate pruned checkpoint and inference file.
    Args:
        input_ckpt: Tensorflow 1.x checkpoint file.
        input_meta: Tensorflow 1.x model structure file.
        pkl_dir: The directory, store Scope_Conv_Dict, Strip_Nodes_Dict, and Strip_Vars_Dict.
        pruned_dict_path: The Pruned_Filters path.
    Outputs:
        The pruned checkpoint and inference model will be save in the directory, named pruned_ckpt.
        The directory is in the path of input_ckpt.
    """
    scope_conv_path = os.path.join(pkl_dir, 'Scope_Conv_Dict.pkl')
    strip_nodes_dict_path = os.path.join(pkl_dir, 'Strip_Nodes_Dict.pkl')
    strip_vars_dict_path = os.path.join(pkl_dir, 'Strip_Vars_Dict.pkl')

    graph = tf.Graph()
    # The new graph is used to add pruned node.
    new_graph = tf.Graph()
    with tf.Session(graph=graph) as sess:

        # Load the meta graph and weights.
        saver = tf.train.import_meta_graph(input_meta)
        saver.restore(sess, input_ckpt)

        # Save all variables of weight.
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

        # Load the filter index and node name for prunig.
        with open(pruned_dict_path, "rb") as f: 
            scope_pruned_filters = pickle.load(f)
        with open(strip_nodes_dict_path, 'rb') as f:
            strip_nodes_dict = pickle.load(f)

        print('{} Modifying Graph and Exporting GraphDef... {}'.format('='*15, '='*15))
        # Modify the node in grpah.
        for scope, filters in scope_pruned_filters.items():
            pruned_nums = len(filters)
            for strip_node, pruned_dim in strip_nodes_dict[scope]:
                # Get tf.Operation by node_name.
                op = graph.get_operation_by_name(strip_node)
                if op.type == 'VariableV2':
                    varDimSize(op, pruned_nums, pruned_dim)
                    pass
                else:
                    if 'shape' in op.name:
                        constTensorContent(op, pruned_nums, pruned_dim)
                    else:
                        constDimSize(op, pruned_nums, pruned_dim)
                        pass

        print('{} Graph Modified. {}'.format('='*15, '='*15))
        # Export modified graph to tf.GraphDef
        # The new tf.GraphDef will be import into new_graph to generate inference model.
        new_graph_def = graph.as_graph_def()
        print('{} Export Graph to New GraphDef. {}'.format('='*15, '='*15))


    # Import New GraphDef to New Graph
    print('{} Importing New GraphDef to New Graph and Updating weights... {}'.format('='*15, '='*15))
    tf.reset_default_graph()
    with tf.Session(graph=new_graph) as new_sess:
        tf.import_graph_def(new_graph_def, name='')

        # Add Variables key in New Graph
        g = tf.get_default_graph()
        for key in graph.get_all_collection_keys():
            if 'variable' in key:
                for variable in graph.get_collection(key):
                    temp = tf.Variable(initial_value=variable.value())
                    temp_v = temp.from_proto(variable.to_proto())
                    g.add_to_collection(key, temp_v)
                pass
            else:
                for variable in graph.get_collection(key):
                    g.add_to_collection(key, variable)
                    pass

        # Get the related dictionary for updating variables.
        with open(scope_conv_path, 'rb') as f:
            scope_conv_dict = pickle.load(f)
        with open(strip_vars_dict_path, 'rb') as f:
            strip_vars_dict = pickle.load(f)

        # Get variables list.
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variables_name = [x.name for x in variables]

        # Update the pruninig variable according to the scope we specified.
        for scope, filters in scope_pruned_filters.items():
            for var_name, dim in strip_vars_dict[scope]:
                variable = variables[variables_name.index(var_name)]
                prev_weights = weights_dict.pop(var_name)
                pruned_weights = np.delete(prev_weights, filters, axis=dim)
                variable.load(pruned_weights, new_sess)
                new_sess.run(variable)
            pass

        # Update the remained variables.
        for key, value in weights_dict.items():
            temp = value
            variable = variables[variables_name.index(key)]
            variable.load(temp, new_sess)
            new_sess.run(variable)
            pass

        pb_path = '/'.join(input_ckpt.split('/')[:-1])
        # After updating, we can save the nwe graph as checkpoint.
        output_dir = pb_path + '/' + 'pruned_ckpt'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            pass
        ckpt_name = input_ckpt.split('/')[-1]
        ckpt_path = os.path.join(output_dir, ckpt_name)
        saver.save(new_sess, ckpt_path)
        # Save inference model
        print('{} Export .pb {}'.format('='*15, '='*15))
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            new_sess, new_graph.as_graph_def(), ['output/BiasAdd'])
        tf.train.write_graph(frozen_graph,
                             pb_path,
                             'modify.pb',
                             False)
        print('{} modify.pb has been saved in {} {}'.format('='*15, pb_path, '='*15))

        print('{} Fininshed. {}'.format('='*15, '='*15))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_ckpt', type=str, required=True,
                        default=None, help='CKPT path.')
    parser.add_argument('--input_meta', type=str, 
                        default=None, help='meta path.')
    parser.add_argument('--pkl_dir', type=str, required=True,
                        default=None, help='The diretory store pkl.')
    parser.add_argument('--pruned_dict', type=str, required=True,
                        default=None, help='Pruned_Filters.pkl path.')

    args = parser.parse_args()

    if args.input_meta is None:
        args.input_meta = args.input_ckpt + '.meta'
        pass

    strip_ckpt(args.input_ckpt, args.input_meta, args.pkl_dir, args.pruned_dict)
