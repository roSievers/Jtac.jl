
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

def add_param(state, d, prefix = "generic", flat = False, flip = None):

    if flat: dims = [product(d['size'])]
    else: dims = list(reversed(d['size']))

    # Since ONNX does not take the word 'convolution'
    # seriously (and uses cross-correlation instead),
    # we have to manually flip some axes :/
    if flip: 
        data = d['data']
        data = np.frombuffer(data, dtype = np.float32)
        data = data.reshape(dims)
        data = np.flip(data, axis=flip)
        vals = data.tobytes()
    else:
        vals = d['data']

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
def add_activation_layer(state, f, input_name):
    if f in ['id', 'identity']:
        return input_name

    elif f == 'relu':
        name = get_name(state, 'relu')
        op_name = 'Relu'

    elif f == 'tanh':
        name = get_name(state, 'tanh')
        op_name = 'Tanh'

    elif f == 'softmax':
        name = get_name(state, 'softmax')
        op_name = 'Softmax'

    else:
        raise ValueError("Expected valid activation layer name, got", f)

    node = ox.make_node(
            op_name,
            inputs = [input_name],
            outputs = [name])

    state["nodes"].append(node)
    return name


## Primitive layers

def add_dense_layer(state, d, input_name):
    assert(d['type']['name'] == 'Dense')

    flat_name = get_name(state, 'flat')
    node = ox.make_node(
            'Flatten',
            inputs = [input_name],
            outputs = [flat_name])

    state["nodes"].append(node)

    w = add_param(state, d['value']['w'], prefix = 'w')
    b = add_param(state, d['value']['b'], prefix = 'b', flat = True)

    name = get_name(state, 'dense')
    node = ox.make_node(
            'Gemm',
            inputs = [flat_name, w, b],
            outputs = [name])

    state["nodes"].append(node)
    out_name = add_activation_layer(state, d['value']['f'], name)
    return out_name

def add_conv_layer(state, d, input_name):
    assert(d['type']['name'] == 'Conv')

    w = add_param(state, d['value']['w'], prefix = 'w', flip = [2,3])
    b = add_param(state, d['value']['b'], prefix = 'b', flat = True)

    strides = list(reversed(d['value']['s']))
    pads = list(reversed(d['value']['p']))
    pads.extend(pads)
    name = get_name(state, 'conv')
    node = ox.make_node(
            'Conv',
            pads = pads,
            strides = strides,
            inputs = [input_name, w, b],
            outputs = [name])

    state["nodes"].append(node)
    out_name = add_activation_layer(state, d['value']['f'], name)
    return out_name

def add_batchnorm_layer(state, d, input_name):
    assert(d['type']['name'] == 'Batchnorm')

    bias = add_param(state, d['value']['bias'], prefix = 'bnbias', flat = True)
    scale = add_param(state, d['value']['scale'], prefix = 'bnscale', flat = True)
    mean = add_param(state, d['value']['mean'], prefix = 'bnmean', flat = True)
    var = add_param(state, d['value']['var'], prefix = 'bnvar', flat = True)

    name = get_name(state, "batchnorm")
    node = ox.make_node(
            'BatchNormalization',
            inputs = [input_name, scale, bias, mean, var],
            outputs = [name])

    state["nodes"].append(node)
    out_name = add_activation_layer(state, d['value']['f'], name)
    return out_name


## Compound layers

def add_chain(state, d, input_name):
    assert(d['type']['name'] == 'Chain')
    for l in d['value']['layers']:
      input_name = add_layer(state, l, input_name)
    return input_name

def add_residual(state, d, input_name):
    assert(d['type']['name'] == 'Residual')
    chain_name = add_chain(state, d['value']['chain'], input_name)
    name = get_name(state, "residual")
    node = ox.make_node(
            "Add",
            inputs = [input_name, chain_name],
            outputs = [name])

    state["nodes"].append(node)

    out_name = add_activation_layer(state, d['value']['f'], name)
    return out_name

def add_layer(state, d, input_name):
    type = d['type']['name']
    if type == 'Conv': return add_conv_layer(state, d, input_name)
    elif type == 'Dense': return add_dense_layer(state, d, input_name)
    elif type == 'Batchnorm': return add_batchnorm_layer(state, d, input_name)
    elif type == 'Chain': return add_chain(state, d, input_name)
    elif type == 'Residual': return add_residual(state, d, input_name)
    else: raise ValueError("Expected Jtac layer, got", type)


## Full model

def add_model(state, d):
    assert(d['type']['name'] == 'NeuralModel')
    trunk = add_layer(state, d['value']['trunk'], 'INPUT')
    vhead = add_layer(state, d['value']['target_heads'][0], trunk)
    phead = add_layer(state, d['value']['target_heads'][1], trunk)

    value = add_activation_layer(state, 'tanh', vhead)
    policy = add_activation_layer(state, 'softmax', phead)

    node = ox.make_node(
            'Concat',
            axis = -1,
            inputs = [value, policy],
            outputs = ['OUTPUT'])

    state['nodes'].append(node)


def get_game_shapes(d):
    if d['name'] == 'MNKGame':
        inshape = [1, *list(reversed(d['params'][0:2]))]
        outshape = [product(inshape) + 1]
    elif d['name'] == 'MetaTac':
        inshape = [1, 9, 9]
        outshape = [9*9 + 1]
    elif d['name'] == 'PacoSako':
        inshape = [30, 8, 8]
        outshape = [132 + 1]

    return inshape, outshape


def convert_onnx(file):
    state = {"tensors" : [], "nodes" : [], "count" : 1}
    with open(file, "rb") as mp:
        model = msgpack.unpack(mp)

    assert(model['type']['name'] == 'NeuralModel')
    add_model(state, model)

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
import onnxruntime as rt
sess = rt.InferenceSession("hedwig-0.8.onnx")
data = np.float32(np.random.rand(1, 30, 8, 8))
res = sess.run(["OUTPUT"], {"INPUT": data})
