from functools import reduce
import numpy as np
import lightning as L
import torch
from torch import nn

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.backend import conv2d, equal
from tensorflow.python.keras.utils import conv_utils
from tensorflow.compat.v1.nn import convolution
from tensorflow.keras import activations


L.seed_everything(100)

batch = 1
input_width = 3
input_height = 3
filters = 1
sequence_num = 1
channels = 1

kernel_size = (1,3)

tf_input_shape = [batch,sequence_num,input_width,input_height,channels] # batch, time_step, width, height, channel
torch_input_shape = [batch,sequence_num,channels,input_width,input_height] # batch, time_step, channel, width, height

tf_inputs_arr = np.random.rand(*tf_input_shape).astype(np.float32).round(3)
torch_inputs_arr = np.transpose(tf_inputs_arr, (0,1,4,2,3))

tf_hidden_shape = (batch, input_width, input_height, filters) # (samples, new_rows(base on input width and kernels), new_cols (base on input height and kernels), filters)
tf_h0 = np.random.rand(*tf_hidden_shape).astype(np.float32).round(3)
tf_c0 = np.random.rand(*tf_hidden_shape).astype(np.float32).round(3) # carry memory shape equal to hidden state shape

tf_inputs_weights = (kernel_size[0], kernel_size[1], channels, 4*filters)
tf_kernel_weights = (kernel_size[0], kernel_size[1], filters, 4*filters)
inputs_weights_arr = np.random.rand(*tf_inputs_weights).astype(np.float32).round(3)
kernel_weights_arr = np.random.rand(*tf_kernel_weights).astype(np.float32).round(3)

##########################################################################################
def test_tf_convlstm():
    print("#"*100)
    # Tensorflow-Keras part
    inputs = tf.cast(tf.convert_to_tensor(tf_inputs_arr), dtype=tf.float32)
    layer = ConvLSTM2D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same", # in the compgan code, this padding is 'valid'
        use_bias=False, 
        recurrent_activation = "sigmoid",
        return_state=True, 
        return_sequences=False)
    
    initial_state = (tf.cast(tf.convert_to_tensor(tf_h0), dtype=tf.float32), tf.cast(tf.convert_to_tensor(tf_c0), dtype=tf.float32))
    print("len(layer(inputs))", len(layer(inputs)))

    weights = (tf.cast(tf.convert_to_tensor(inputs_weights_arr), dtype=tf.float32), 
                tf.cast(tf.convert_to_tensor(kernel_weights_arr), dtype=tf.float32))
    layer.set_weights(weights)

    outputs, ht, ct = layer(inputs,initial_state=initial_state)
    
    weights = layer.get_weights()
    for w in weights:
        print("tf weight", w)

    # (kernel_i, kernel_f, kernel_c, kernel_o) = tf.split(
    #     layer.cell.kernel, 4, axis=2 + 1
    # )
    # print("4 kernel shape:")
    # for k in (kernel_i, kernel_f, kernel_c, kernel_o):
    #     print(k.shape) # width, height, channel, filter_num

    # (rnn_kernel_i, rnn_kernel_f, rnn_kernel_c, rnn_kernel_o) = tf.split(
    #     layer.cell.recurrent_kernel, 4, axis=2 + 1
    # )
    # print("4 rnn_kernel shape:")
    # for k in (rnn_kernel_i, rnn_kernel_f, rnn_kernel_c, rnn_kernel_o):
    #     print(k.shape) # width, height, channel, filter_num
    return ht, ct

tf_output = test_tf_convlstm()
##########################################################################################
## Pytorch ConvLSTM

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
            
        Input: ()
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim # filter num

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


def test_pytorch_convLSTM():
    h0 = torch.tensor(np.transpose(tf_h0, (0,3,1,2)))
    c0 = torch.tensor(np.transpose(tf_c0, (0,3,1,2)))
    weights = torch.cat((torch.from_numpy(inputs_weights_arr), 
                         torch.from_numpy(kernel_weights_arr)), axis=2).permute(3,2,0,1)
    input_tensor = torch.tensor(torch_inputs_arr)
    
    conv_LSTM = ConvLSTMCell(
        input_dim=torch_input_shape[2],
        hidden_dim=filters,
        kernel_size=kernel_size,
        bias=False,)
    with torch.no_grad():
        conv_LSTM.conv.weight.copy_(weights)
    
    weights = conv_LSTM.conv.weight
    print("pytorch_weight", weights)
    
    output_inner = []
    layer_output_list = []
    last_state_list = []

    seq_len = input_tensor.size(1)
    cur_layer_input = input_tensor
    h, c = h0, c0

    for t in range(seq_len):
        h, c = conv_LSTM(input_tensor=cur_layer_input[:, t, :, :, :],
                            cur_state=[h, c])
        output_inner.append(h)

    layer_output = torch.stack(output_inner, dim=1)
    cur_layer_input = layer_output

    layer_output_list.append(layer_output)
    last_state_list.append([h, c])
    
    print("ht", last_state_list[-1][0].shape)
    print("ct", last_state_list[-1][1].shape)
    
    return last_state_list[-1]
        
torch_output = test_pytorch_convLSTM()

def test_both_output(tf_output, torch_output):
    print("#"*100)
    tf_ht, tf_ct = tf_output
    tf_ht = tf_ht.numpy()
    tf_ct = tf_ct.numpy()
    torch_ht, torch_ct = torch_output
    torch_ht = torch_ht.permute(0,2,3,1).detach().numpy()
    torch_ct = torch_ct.permute(0,2,3,1).detach().numpy()
    
    print("tf_ht", tf_ht)
    print("torch_ht", torch_ht)
    
    print("tf_ht.dtype", tf_ht.dtype)
    print("torch_ht.dtype", torch_ht.dtype)
    
test_both_output(tf_output, torch_output)

def test_convolution():
    ##########################################################################################
    print("#"*100)
    print("Test Conv")
    ## Test convolution
    conv_input_shape = (batch, 3, 4, 2)
    conv_inputs_arr = np.arange(reduce(lambda x,y: x*y, conv_input_shape)).reshape(*conv_input_shape).astype(np.float32)
    conv_kernel_array = np.array([[[
        [1., 1.],
        [0., 0.]],
        [[1., 0.],
        [0., 1.]],
    ]]).astype(np.float32)

    # Tensorflow & Keras
    print("#"*100)
    print("TEST TENSORFLOW CONV")
    conv_inputs = tf.cast(tf.convert_to_tensor(conv_inputs_arr), dtype=tf.float32)
    conv_kernel = tf.cast(tf.convert_to_tensor(conv_kernel_array), dtype=tf.float32)
    data_format = conv_utils.normalize_data_format(None)
    data_format = "NHWC"
    conv_output = convolution(
        conv_inputs,
        conv_kernel,
        padding="VALID",
        data_format=data_format)

    print("conv_output.shape", conv_output.shape)
    print("conv_inputs", tf.transpose(conv_inputs, perm=[0, 3, 1, 2]))
    print("conv_kernel", tf.transpose(conv_kernel, perm=[3, 2, 0, 1]))
    print("conv_output", tf.transpose(conv_output, perm=[0, 3, 1, 2]))

    ## Pytorch
    print("#"*100)
    print("TEST TORCH CONV")
    conv_inputs = torch.Tensor(conv_inputs_arr.transpose(0, 3, 1, 2))
    print("conv_inputs.shape", conv_inputs.shape)
    conv_kernel_transposed = conv_kernel_array.transpose(3, 2, 0, 1)
    print("conv_kernel_transposed.shape", conv_kernel_transposed.shape)

    conv_kernel = torch.Tensor(conv_kernel_transposed)
    conv_output = nn.functional.conv2d(conv_inputs, conv_kernel, stride=1)

    print("conv_output.shape", conv_output.shape)
    print("conv_output", conv_output)
    
    ## Pytorch Conv2d
    print("#"*100)
    print("TEST TORCH CONV2D")
    conv_inputs = torch.Tensor(conv_inputs_arr.transpose(0, 3, 1, 2))
    print("conv_inputs.shape", conv_inputs.shape)
    conv_kernel_transposed = conv_kernel_array.transpose(3, 2, 0, 1)
    print("conv_kernel_transposed.shape", conv_kernel_transposed.shape)

    conv_kernel = torch.Tensor(conv_kernel_transposed)
    
    conv2d_model = nn.Conv2d(
        in_channels=conv_inputs.shape[1],
        out_channels=conv_input_shape[-1],
        kernel_size=(1,2),
        padding=0,
        bias=False)
    
    with torch.no_grad():
        conv2d_model.weight.copy_(conv_kernel)
        
    conv2d_output = conv2d_model(conv_inputs)

    print("conv_output.shape", conv2d_output.shape)
    print("conv_output", conv2d_output)

# test_convolution()
