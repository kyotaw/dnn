import numpy as np

from dnn.utils import *


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング
    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad
    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class AffineLayer:
    def __init__(self, name, W, b):
        self.name = name
        self.W = W
        self.b = b
        self.x = None
        self.y = None
        self.dW = None
        self.db = None
        self.dx = None
        self.original_x_shape = None

    @property
    def params(self):
        return {key_W(self.name): self.W, key_b(self.name): self.b}

    @property
    def grads(self):
        return {key_W(self.name): self.dW, key_b(self.name): self.db}
    
    def forward(self, x, is_train):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        self.x = x
        self.y = np.dot(x, self.W) + self.b
        return self.y 

    def backward(self, dy):
        self.dx = np.dot(dy, self.W.T)
        self.dW = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0)
        self.dx = self.dx.reshape(*self.original_x_shape)
        return self.dx


class ReluLayer:
    def __init__(self, name):
        self.name = name
        self.mask = None
        self.y = None
    
    @property
    def params(self):
        return {}

    @property
    def grads(self):
        return {}
    
    def forward(self, x, is_train):
        self.mask = (x <= 0)
        self.y = x.copy()
        self.y[self.mask] = 0
        return self.y

    def backward(self, dy):
        dy[self.mask] = 0
        dx = dy
        return dx


class SoftmaxLayer:
    def __init__(self, name, loss_calculator):
        self.name = name
        self.y = None
        self.t = None
        self.loss_calculator = loss_calculator
    
    @property
    def params(self):
        return {}
    
    @property
    def grads(self):
        return {}

    def forward(self, x, t):
        self.t = t
        self.y = self._softmax(x)
        self.loss = self.loss_calculator.calc(self.y, t)
        return self.y, self.loss

    def backward(self, dy=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

    def _softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, name, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.name = name
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None
    
    @property
    def params(self):
        return {key_gamma(self.name): self.gamma, key_beta(self.name): self.beta}

    @property
    def grads(self):
        return {key_gamma(self.name): self.dgamma, key_beta(self.name): self.dbeta}
   
    def forward(self, x, is_train=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, name, W, b, stride=1, pad=0):
        self.name = name
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    @property
    def params(self):
        return {key_W(self.name): self.W, key_b(self.name): self.b}
    
    @property
    def grads(self):
        return {key_W(self.name): self.dW, key_b(self.name): self.db}
    
    def output_size(self, input_height, input_width):
        FN, C, FH, FW = self.W.shape
        out_h = 1 + int((input_height + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((input_width + 2 * self.pad - FW) / self.stride)
        return out_h, out_w


    def forward(self, x, is_train):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h, out_w = self.output_size(H, W)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx



class Pooling:
    def __init__(self, name, pool_h, pool_w, stride=1, pad=0):
        self.name = name
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None
    
    @property
    def params(self):
        return {}

    @property
    def grads(self):
        return {}
    
    def output_size(self, input_height, input_size, input_num):
        out_w = int(input_num * (input_height / 2) * (input_height / 2))
        out_h = int(input_num * (input_width / 2) * (input_width / 2))
        return out_w, out_w
        

    def forward(self, x, is_train):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
