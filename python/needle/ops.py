"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return out_grad * self.scalar * (x ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad/rhs, -(out_grad*lhs)/(rhs**2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        res_device = a.device
        # print(f"transpose compute device: {res_device}")

        a = NDArray(a.numpy(), device=res_device)
        if self.axes is not None:
            return swapaxes(a, self.axes)
        else:
            size = a.ndim
        return swapaxes(a, (size-2, size-1))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION

def swapaxes(tensor, axes):
    axes_list = numpy.arange(tensor.ndim)
    axes_list[axes[0]], axes_list[axes[1]] = axes_list[axes[1]], axes_list[axes[0]]
    return tensor.permute(tuple(axes_list))


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        #支持shape中带-1的值
        self.shape = a.numpy().reshape(self.shape).shape

        res = a.compact().reshape(self.shape)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return out_grad.reshape(x.shape)
        ### END YOUR SOLUTION



def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return unbroadcast(out_grad, x.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

def unbroadcast(out_grad, shape):
    ndims_added = out_grad.numpy().ndim - len(shape)
    add_nums = 0
    for _ in range(ndims_added):
         out_grad = out_grad.sum(axes=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            out_grad = out_grad.sum(axes=(i-add_nums))
            add_nums += 1
            
    return out_grad.reshape(shape)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.sumShape = a.sum(self.axes, keepdims=True).shape

        return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return out_grad.reshape(self.sumShape).broadcast_to(x.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # X: M * N Y: M * K W: N * K
        # dW = matmul(X.T, dY), dX = matmul(dY, W.T)
        return unbroadcast(out_grad.matmul(rhs.transpose()), lhs.shape), unbroadcast(lhs.transpose().matmul(out_grad), rhs.shape)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return out_grad / x
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return out_grad * exp(x)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.device = a.device
        self.mask = (a > 0)

        return a * self.mask
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return  out_grad * Tensor(self.mask.numpy(), device=self.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=self.axes, keepdims=True)
        sum = (Z-max_z.broadcast_to(Z.shape)).exp().sum(axis=self.axes)
        self.oldShape = max_z.shape
        res = sum.log() + Z.max(axis=self.axes)
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z, = node.inputs
        Z_process = logsumexp(Z, axes=self.axes).reshape(self.oldShape).broadcast_to(Z.shape)
        res = out_grad.reshape(self.oldShape).broadcast_to(Z.shape) * exp(Z - Z_process)
        return res
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        return out_grad * (1-tanh(x) ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        in_shape = args[0].shape
        out_shape = [len(args)] + list(in_shape)
        out = NDArray.make(out_shape, device=args[0].device)
        idxs = [slice(None, None, None) for j in range(len(in_shape))]
        for i, arg in enumerate(args):
            assert arg.shape == in_shape
            idxs_i = tuple([i] + idxs)
            out[idxs_i] = arg.compact()
        out_axes = list(range(1, len(out_shape)))
        out_axes.insert(self.axis, 0)
        return out.permute(tuple(out_axes)).compact()
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        in_shape = A.shape
        idx = [slice(None, None, None) for j in range(len(in_shape))]
        results = []
        for i in range(in_shape[self.axis]):
            idx_i = idx.copy()
            idx_i[self.axis] = i
            idx_i = tuple(idx_i)
            out = A[idx_i]
            # TODO: since it's a fake reduction, can I just drop the 1-dimension?
            out = out.sum(axis=self.axis)
            results.append(out)
        return tuple(results)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)



class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.device = a.device
        self.dtype = a.dtype
        # print(f"flip compute device: {res_device}")

        a = NDArray(a.numpy())
        res = a.flip(self.axes)
        return NDArray(res.numpy(), device=self.device)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Tensor(numpy.flip(out_grad.numpy(), self.axes), device=self.device, dtype=self.dtype)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.device = a.device
        self.dtype = a.dtype
        # print(f"dilate compute device: {res_device}")

        new_shape = tuple([(self.dilation+1) * s if i in self.axes else s for i, s in enumerate(a.shape)])
        out = numpy.zeros(new_shape)
        self.dilate_slice = tuple([slice(None, None, (self.dilation+1)) if i in self.axes else slice(None) for i, s in enumerate(a.shape)])
        out[self.dilate_slice] = a.numpy()
        return NDArray(out, device=self.device)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Tensor(out_grad.numpy()[self.dilate_slice], device=self.device, dtype=self.dtype)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        self.device = a.device
        self.dtype = a.dtype
        # print(f"undilate compute device: {res_device}")

        new_shape = tuple([s // (self.dilation+1) if i in self.axes else s for i, s in enumerate(a.shape)])
        out = numpy.zeros(new_shape)
        self.dilate_slice = tuple([slice(None, None, (self.dilation+1)) if i in self.axes else slice(None) for i, s in enumerate(a.shape)])
        out = (a.numpy())[self.dilate_slice]
        return NDArray(out, device=self.device)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, = node.inputs
        out = numpy.zeros(x.shape)
        out[self.dilate_slice] = out_grad.numpy()
        return Tensor(out, device=self.device, dtype=self.dtype)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        self.device = A.device
        self.dtype = A.dtype
        # print(f"conv compute device: {self.device}")

        padding_A = A.pad(axes=((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))).numpy()
        N,H,W,C_in = padding_A.shape
        K,_,_,C_out = B.shape
        Ns, Hs, Ws, Cs = padding_A.strides
    
        inner_dim = K * K * C_in
        C = numpy.lib.stride_tricks.as_strided(padding_A, shape = (N, (H-K+1) // self.stride, (W-K+1) // self.stride, K, K, C_in),
                                        strides = (Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).reshape(-1,inner_dim)
        out = C @ B.numpy().reshape(-1, C_out)

        return NDArray(out.reshape(N, (H-K+1) // self.stride, (W-K+1) // self.stride, C_out), device=self.device)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x, w = node.inputs

        x_dw = Tensor(NDArray(x.numpy()).permute((3, 1, 2, 0)).numpy())
        out_grad_dilate = Tensor(dilate(out_grad, axes=(1, 2), dilation=self.stride-1).numpy())
        out_grad_dw = Tensor(NDArray(out_grad_dilate.numpy()).permute((1, 2, 0, 3)).numpy())
        # print(f"x_dw shape: {x_dw.shape},  out_grad_dw shape: {out_grad_dw.shape}")
        w_flip = Tensor(numpy.flip(w.transpose((2, 3)).numpy(), (0, 1)))


        if self.padding == 0:
            dx = Tensor(conv(out_grad_dilate, w_flip, padding=w.shape[0]-1).numpy(), device=self.device, dtype=self.dtype)
        elif self.padding > 0:
            dx = Tensor(conv(out_grad_dilate, w_flip, padding=w.shape[0]-1).numpy()[:, self.padding:-self.padding, self.padding:-self.padding, :], device=self.device, dtype=self.dtype)
        dw = Tensor(NDArray(conv(x_dw, out_grad_dw, padding=self.padding).numpy()).permute((1,2,0,3)).numpy(), device=self.device, dtype=self.dtype)
        return dx, dw
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


