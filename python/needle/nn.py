"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, dtype=dtype, device=device))
        if bias == True:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, dtype=dtype, device=device).reshape((1, out_features)))
        else:
            self.bias = init.zeros(out_features, 1, dtype=dtype, device=device).reshape((1, out_features))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X @ self.weight + self.bias.broadcast_to((X.shape[0], self.out_features))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        X.grad = X
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        exp_x = ops.exp(-x)
        one_mat = init.ones(*x.shape, device=x.device, dtype=x.dtype)
        return one_mat / (1 + exp_x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        res_device = logits.device
        # print(f"softmax loss device: {res_device}")

        one_hot_tuple = tuple([init.one_hot(logits.shape[1], Tensor(y.numpy()[i]), device=res_device) for i in range(y.shape[0])])
        one_hot = ops.stack(one_hot_tuple, axis=0)
        return ops.summation(ops.logsumexp(logits, axes=1)-ops.summation(logits*one_hot, axes=1)) / y.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(dim, dtype=dtype, device=device))
        self.running_mean = init.zeros(dim, dtype=dtype, device=device)
        self.running_var = init.ones(dim, dtype=dtype, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training == True:
            mean_pre = (x.sum(axes=0) / x.shape[0])
            mean = mean_pre.reshape((1, x.shape[1])).broadcast_to(x.shape)
            var_pre = (((x-mean)**2).sum(axes=0) / x.shape[0])
            var = var_pre.reshape((1, x.shape[1])).broadcast_to(x.shape)
            # print(mean, var)
            w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
            b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
            y = w * (x - mean) / ((var + self.eps) ** 0.5) + b

            self.running_mean = self.running_mean.data * (1 - self.momentum) + mean_pre * self.momentum
            self.running_var = self.running_var.data * (1 - self.momentum) + var_pre * self.momentum
        else:
            x_normalize = (x - self.running_mean.data) / ((self.running_var.data + self.eps) ** 0.5) 
            w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
            b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
            y = w * x_normalize + b
        return y
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, dtype=dtype, device=device))
        self.bias = Parameter(init.zeros(dim, dtype=dtype, device=device))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = np.zeros(x.shape)
        mean = (x.sum(axes=1) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var = (((x-mean)**2).sum(axes=1) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        y = w * (x - mean) / ((var + self.eps) ** 0.5) + b
        return y
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training == True:
            mask = Tensor(init.randb(*x.shape, p=1-self.p)) / (1 - self.p)
            return x * mask
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        interval = 1.0 / ((in_channels * kernel_size ** 2) ** 0.5)
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_channels * kernel_size**2, fan_out=out_channels * kernel_size**2, \
                 shape = (kernel_size, kernel_size, in_channels, out_channels), device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.rand(out_channels, low=-interval, high=interval, requires_grad=True, device=device, dtype=dtype))
        else:
            self.bias = init.zeros(out_channels, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_fit = x.transpose((1, 2)).transpose((2, 3))
        padding = self.kernel_size // 2
        
        assert self.kernel_size % 2 == 1

        conv_out = ops.conv(x_fit, self.weight, stride=self.stride, padding=padding)
        res = (conv_out + self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(conv_out.shape)).transpose((2, 3)).transpose((1, 2))
        return res
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = (1.0 / hidden_size) ** 0.5
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = init.zeros(hidden_size, device=device, dtype=dtype)
            self.bias_hh = init.zeros(hidden_size, device=device, dtype=dtype)

        if nonlinearity == 'tanh':
            self.activation = ops.tanh
        else:
            self.activation = ops.relu
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bias_ih = self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size))
        bias_hh = self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((X.shape[0], self.hidden_size))
        if not h:
            h = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=self.dtype)
        h_next = self.activation(X @ self.W_ih + bias_ih + h @ self.W_hh + bias_hh)
        return h_next
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        self.rnn_cells = [RNNCell(input_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype)] + \
                        [RNNCell(hidden_size, hidden_size, bias=bias, nonlinearity=nonlinearity, device=device, dtype=dtype) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = X.shape[0], X.shape[1]
        output = []

        inp = list(ops.split(X, axis=0))
        if not h0:
            h = [init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        else:
            h = list(ops.split(h0, axis=0))

        for t in range(seq_len):
            for layer in range(self.num_layers):
                h[layer] = self.rnn_cells[layer](inp[t], h[layer])
                inp[t] = h[layer]

            # 收集最终层的输出
            output.append(h[-1])

        output = ops.stack(tuple(output), axis=0)
        h_n = ops.stack(tuple(h), axis=0)

        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = (1.0 / hidden_size) ** 0.5
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True))
        else:
            self.bias_ih = init.zeros(4*hidden_size, device=device, dtype=dtype)
            self.bias_hh = init.zeros(4*hidden_size, device=device, dtype=dtype)

        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bias_ih = self.bias_ih.reshape((1, 4*self.hidden_size)).broadcast_to((X.shape[0], 4*self.hidden_size))
        bias_hh = self.bias_hh.reshape((1, 4*self.hidden_size)).broadcast_to((X.shape[0], 4*self.hidden_size))
        if not h:
            h0 = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(X.shape[0], self.hidden_size, device=self.device, dtype=self.dtype)
            h = (h0, c0)

        
        # 一次性计算三个门 与 g_t
        ifgo = X @ self.W_ih + bias_ih + h[0] @ self.W_hh + bias_hh
        i, f, g, o = ops.split(ops.reshape(ifgo, (X.shape[0], 4, self.hidden_size)), axis=1)

        c_next = Sigmoid()(f) * h[1] + Sigmoid()(i) * Tanh()(g)

        h_next = Sigmoid()(o) * Tanh()(c_next)


        return (h_next, c_next)
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias=bias, device=device, dtype=dtype)] + \
                        [LSTMCell(hidden_size, hidden_size, bias=bias, device=device, dtype=dtype) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = X.shape[0], X.shape[1]
        output = []

        inp = list(ops.split(X, axis=0))
        if not h:
            h_lstm = [init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
            c_lstm = [init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        else:
            h_lstm = list(ops.split(h[0], axis=0))
            c_lstm = list(ops.split(h[1], axis=0))

        for t in range(seq_len):
            for layer in range(self.num_layers):
                h_lstm[layer], c_lstm[layer] = self.lstm_cells[layer](inp[t], (h_lstm[layer], c_lstm[layer]))
                inp[t] = h_lstm[layer]

            # 收集最终层的输出
            output.append(h_lstm[-1])

        output = ops.stack(tuple(output), axis=0)
        h_n = ops.stack(tuple(h_lstm), axis=0)
        c_n = ops.stack(tuple(c_lstm), axis=0)

        return output, (h_n, c_n)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        x_one_hot = init.one_hot(self.num_embeddings, x.realize_cached_data().flat, device=self.device, dtype=self.dtype)
        res = x_one_hot @ self.weight
        return res.reshape((*x.shape, self.embedding_dim))
        ### END YOUR SOLUTION
