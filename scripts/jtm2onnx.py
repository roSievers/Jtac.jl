
import os
import msgpack

import numpy as np
import onnx
import onnx.helper as ox

from operator import mul
from functools import reduce

# Params use float32
FLOAT = onnx.TensorProto.FLOAT

def product(vals):
    return reduce(mul, vals)

# Need unique names for outputs
def get_name(state, prefix):
  name = prefix + str(state["count"])
  state["count"] += 1
  return name

def convert_param(state, d, prefix = "generic", flat = False, flip = None):

    if flat: dims = [product(d['dims'])]
    else: dims = list(reversed(d['dims']))

    # Since ONNX does not take the word 'convolution'
    # seriously (and uses cross-correlation instead),
    # we have to manually flip some axes :/
    if flip: 
        data = d['bytes']
        data = np.frombuffer(data, dtype = np.float32)
        data = data.reshape(dims)
        data = np.flip(data, axis=flip)
        vals = data.tobytes()
    else:
        vals = d['bytes']

    name = get_name(state, prefix)
    tensor = ox.make_tensor(
            name = name,
            data_type = FLOAT,
            dims = dims,
            vals = vals,
            raw = True)

    state["tensors"].append(tensor)
    return name

# TODO: should add several other activation functions
def convert_activation(state, d, input_name):
    fun = d['name']
    if fun == 'id':
        return input_name

    elif fun == 'relu':
        name = get_name(state, 'relu')
        op_name = 'Relu'

    elif fun == 'tanh':
        name = get_name(state, 'tanh')
        op_name = 'Tanh'

    elif fun == 'softmax':
        name = get_name(state, 'softmax')
        op_name = 'Softmax'

    else:
        assert(False)

    node = ox.make_node(
            op_name,
            inputs = [input_name],
            outputs = [name])

    state["nodes"].append(node)
    return name


# PRIMITIVE LAYERS

def convert_conv(state, d, input_name):
    assert(d['type']['name'] == 'conv')
    w = convert_param(state, d['w'], prefix = 'w', flip = [2,3])
    b = convert_param(state, d['b'], prefix = 'b', flat = True)
    strides = list(reversed(d['s']))
    pads = list(reversed(d['p']))
    pads.extend(pads)
    name = get_name(state, "conv")
    node = ox.make_node(
            'Conv',
            pads = pads,
            strides = strides,
            inputs = [input_name, w, b],
            outputs = [name])

    state["nodes"].append(node)
    out_name = convert_activation(state, d['a'], name)
    return out_name

def convert_dense(state, d, input_name):
    assert(d['type']['name'] == 'dense')

    flat_name = get_name(state, 'flat')
    node = ox.make_node(
            'Flatten',
            inputs = [input_name],
            outputs = [flat_name])

    state["nodes"].append(node)

    w = convert_param(state, d['w'], prefix = 'w')
    b = convert_param(state, d['b'], prefix = 'b', flat = True)
    name = get_name(state, "dense")
    node = ox.make_node(
            'Gemm',
            inputs = [flat_name, w, b],
            outputs = [name])

    state["nodes"].append(node)
    out_name = convert_activation(state, d['a'], name)
    return out_name

def convert_batchnorm(state, d, input_name):
    assert(d['type']['name'] == 'batchnorm')

    scale, bias = convert_bn_scale_bias(state, d['params'])
    mean = convert_param(state, d['moments']['mean'], prefix = 'bnmean', flat = True)
    var = convert_param(state, d['moments']['var'], prefix = 'bnvar', flat = True)

    name = get_name(state, "batchnorm")
    node = ox.make_node(
            'BatchNormalization',
            inputs = [input_name, scale, bias, mean, var],
            outputs = [name])

    state["nodes"].append(node)
    out_name = convert_activation(state, d['a'], name)
    return out_name


def convert_bn_scale_bias(state, d):
    assert(not d['param'])

    scale_name = get_name(state, "bnscale")
    bias_name = get_name(state, "bnbias")

    assert(len(d['dims']) == 1)
    assert(d['dims'][0] % 2 == 0)
    length = d['dims'][0] // 2

    scale_tensor = ox.make_tensor(
            name = scale_name,
            data_type = FLOAT,
            dims = [length],
            vals = d['bytes'][0:4*length],
            raw = True)

    bias_tensor = ox.make_tensor(
            name = bias_name,
            data_type = FLOAT,
            dims = [length],
            vals = d['bytes'][4*length:],
            raw = True)

    state["tensors"].append(scale_tensor)
    state["tensors"].append(bias_tensor)

    return scale_name, bias_name

# COMPOUND LAYERS

def convert_chain(state, d, input_name):
    assert(d['type']['name'] == 'chain')
    for l in d['layers']:
      input_name = convert_layer(state, l, input_name)
    return input_name

def convert_residual(state, d, input_name):
    assert(d['type']['name'] == 'residual')
    chain_name = input_name
    for l in d['chain']['layers']:
      chain_name = convert_layer(state, l, chain_name)

    name = get_name(state, "residual")
    node = ox.make_node(
            "Add",
            inputs = [input_name, chain_name],
            outputs = [name])

    state["nodes"].append(node)

    out_name = convert_activation(state, d['a'], name)
    return out_name

def convert_layer(state, d, input_name):
    typ = d['type']['name']
    if typ == 'conv': return convert_conv(state, d, input_name)
    elif typ == 'dense': return convert_dense(state, d, input_name)
    elif typ == 'batchnorm': return convert_batchnorm(state, d, input_name)
    elif typ == 'chain': return convert_chain(state, d, input_name)
    elif typ == 'residual': return convert_residual(state, d, input_name)
    else: assert(False)


# MODEL

def convert_model(state, d):
    trunk = convert_layer(state, d['trunk'], 'INPUT')
    vhead = convert_layer(state, d['heads'][0], trunk)
    phead = convert_layer(state, d['heads'][1], trunk)

    value = convert_activation(state, {'name' : 'tanh'}, vhead)
    policy = convert_activation(state, {'name' : 'softmax'}, phead)

    node = ox.make_node(
            'Concat',
            axis = -1,
            inputs = [value, policy],
            outputs = ['OUTPUT'])

    state['nodes'].append(node)


def get_game_shapes(d):
    if d['name'] == 'm_n_k_game':
        inshape = [1, *list(reversed(d['params'][0:2]))]
        outshape = [product(inshape) + 1]
    elif d['name'] == 'meta_tac':
        inshape = [1, 9, 9]
        outshape = [9*9 + 1]
    elif d['name'] == 'paco_sako':
        inshape = [30, 8, 8]
        outshape = [132 + 1]

    return inshape, outshape


def convert_onnx(file):
    state = {"tensors" : [], "nodes" : [], "count" : 1}
    with open(file, "rb") as mp:
        model = msgpack.unpack(mp)

    assert(model['type']['name'] == 'neural_model')
    convert_model(state, model)

    game = model['type']['params'][0]
    inshape, outshape = get_game_shapes(game)

    x = ox.make_tensor_value_info(
            name = 'INPUT',
            elem_type = FLOAT,
            shape = [None, *inshape])

    y = ox.make_tensor_value_info(
            name = 'OUTPUT',
            elem_type = FLOAT,
            shape = [None, *outshape])

    graph = ox.make_graph(
            nodes = state['nodes'],
            name = os.path.basename(file),
            inputs = [x],
            outputs = [y],
            initializer = state['tensors'])

    onnx.checker.check_graph(graph)
    onnx_model = onnx.helper.make_model(graph, producer_name="jtac")
    onnx_model.opset_import[0].version = 15
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)

    return onnx_model

import sys

# Conversion script
if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    path = sys.argv[1]
    model = convert_onnx(path)
    print('jtac model loaded:', path)
    base, _ = os.path.splitext(path)
    out = base + '.onnx'
    onnx.save(model, base + '.onnx')
    print('onnx model saved:', out)

# Testing the model
#import onnxruntime as rt
#sess = rt.InferenceSession("ludwig.onnx")
#data = (...)
#res = sess.run(["OUTPUT"], {"INPUT":data})
