import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from collections import OrderedDict
import tensorflow as tf
from torch.autograd import Variable
from onnx_tf.backend import prepare

class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model= nn.Sequential(layers)
        print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)

print("%s" % sys.argv[1])
print("%s" % sys.argv[2])


# Load the trained model from file
trained_dict = torch.load(sys.argv[1], map_location={'cuda:0': 'cpu'})

trained_model = MLP(784, [256, 256], 10)
trained_model.load_state_dict(trained_dict)

if not os.path.exists("%s" % sys.argv[2]):
    os.makedirs("%s" % sys.argv[2])

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 1, 28, 28)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(trained_model, dummy_input, "%s/mnist.onnx" % sys.argv[2])

# Load the ONNX file
model = onnx.load("%s/mnist.onnx" % sys.argv[2])

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)

# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')
print(tf_rep.tensor_dict)

tf_rep.export_graph("%s/mnist.pb" % sys.argv[2])

converter = tf.lite.TFLiteConverter.from_frozen_graph(
        "%s/mnist.pb" % sys.argv[2], tf_rep.inputs, tf_rep.outputs)
tflite_model = converter.convert()
open("%s/mnist.tflite" % sys.argv[2], "wb").write(tflite_model)
