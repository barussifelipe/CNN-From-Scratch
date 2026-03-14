import cupy as np 
import components 
import math


class BatchNorm2D:
    def __init__(self, channels, gamma, beta, alpha=0.1, sgd_momentum=0.9):
        self.running_mean = np.zeros(channels)
        self.running_variance = np.ones(channels)
        
        # alpha in your formula
        self.momentum = alpha
        self.gamma = gamma 
        self.beta = beta
        self.sgd_momentum = sgd_momentum
        self.v_gamma = np.zeros_like(self.gamma)
        self.v_beta = np.zeros_like(self.beta)
        self.batch_cache = None
        self.batch_norm_cache = None
        self.mean = None
        self.variance = None 
        self.epsilon = None

    def forward(self, batch, training=True, e=1e-5):
        if training:

            mean = np.mean(batch, axis=(0, 1, 2))
            variance = np.var(batch, axis=(0, 1, 2))

            self.mean = mean 
            self.variance = variance 
            self.epsilon = e 

            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_variance = self.momentum * variance + (1 - self.momentum) * self.running_variance

            batch_normalized = (batch - mean) / np.sqrt(variance + e)
            
        else:
            batch_normalized = (batch - self.running_mean) / np.sqrt(self.running_variance + e)

        self.batch_norm_cache = batch_normalized
        

        y = self.gamma * batch_normalized + self.beta


        
        return y
    
    def backward(self, dy, learning_rate): 
        x_hat = self.batch_norm_cache
        gamma = self.gamma
        variance = self.variance
        epsilon = self.epsilon

        # Reduce over N, H, W and keep channel dimension for broadcasting.
        batch_size = math.prod(dy.shape[:3])
        inv_std = 1.0 / np.sqrt(variance + epsilon)

        d_gamma = np.sum(dy * x_hat, axis=(0, 1, 2))
        d_beta = np.sum(dy, axis=(0, 1, 2))

        sum_dy = d_beta.reshape(1, 1, 1, -1)
        sum_dy_xhat = d_gamma.reshape(1, 1, 1, -1)


        dX = (gamma * inv_std / batch_size) * (
            batch_size * dy - sum_dy - x_hat * sum_dy_xhat
        )

        self.v_gamma = self.sgd_momentum * self.v_gamma - learning_rate * d_gamma
        self.v_beta = self.sgd_momentum * self.v_beta - learning_rate * d_beta

        self.gamma += self.v_gamma
        self.beta += self.v_beta

        # ==========================================
        # THE GARBAGE COLLECTOR
        # ==========================================
        # 1. Unbind the matrices from the class instance
        self.batch_norm_cache = None
        self.mean = None
        self.variance = None
        self.epsilon = None
        
        # 2. Delete local intermediate matrices
        del x_hat, inv_std, d_gamma, d_beta, sum_dy, sum_dy_xhat
        
        # 3. Force CuPy to instantly return the VRAM to the pool
        mempool = np.get_default_memory_pool()
        mempool.free_all_blocks()


        return dX

    

class Convolution:
    def __init__(self, filter_h, filter_w, filters, bias, depth_wise, sgd_momentum=0.9):
        self.bias = bias 
        self.depth_wise = depth_wise
        self.sgd_momentum = sgd_momentum
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.filters = filters
        self.kernel = None
        self.v_kernel = None
        self.v_bias = None
        self.cache_input = None
        self.cache_weight = None 
        self.padding = None
        self.stride = None
        
    def forward(self, x, stride, padding): 
        self.input_data = x

        C = self.input_data.shape[3]
        if self.kernel is None:
            if self.depth_wise:
                kernel_shape = (self.filter_h, self.filter_w, C, 1)
                bias_depth = C
            else:
                kernel_shape = (self.filter_h, self.filter_w, C, self.filters)
                bias_depth = self.filters
            self.kernel = components.initialization(kernel_shape=kernel_shape, type="He")
            self.bias = np.zeros(bias_depth)
        
        if self.v_kernel is None or self.v_bias is None:
            self.v_kernel = np.zeros_like(self.kernel)
            self.v_bias = np.zeros_like(self.bias)

        self.padding = padding
        self.stride = stride
        filters = self.kernel.shape[3]
        filter_h = self.kernel.shape[0]
        filter_w = self.kernel.shape[1]
        
        kernel_passes_h = (self.input_data.shape[1] - filter_h + 2 * padding) // stride + 1 
        kernel_passes_w = (self.input_data.shape[2] - filter_w + 2 * padding) // stride + 1

        x_flat = components.im2col(self.input_data, self.kernel, padding=padding, stride=stride, depth_wise=self.depth_wise)

        if not self.depth_wise:
            weight_flat = self.kernel.reshape(-1, filters)

            output = x_flat @ weight_flat
            final_output = output.reshape(-1, kernel_passes_h, kernel_passes_w, filters) #(N, H_out, W_out, F)
            self.cache_input = x_flat #[patches * batch_size, pixels per batch]
            self.cache_weight = weight_flat #[height * width * channels, filters]
            
        else: 
            x_batched = x_flat.transpose(2, 0, 1)
            w_batched = self.kernel.transpose(2, 0, 1, 3).reshape(C, -1, 1)

            output = x_batched @ w_batched
            output = output.reshape(C, self.input_data.shape[0], kernel_passes_h, kernel_passes_w)
            self.cache_input = x_batched #[C, patches * batch_size, pixels per channel per batch]
            self.cache_weight = w_batched #[C, height * width, 1]
            

            final_output = output.transpose(1, 2, 3, 0) #(N, H_out, W_out, C)

        
        final_output += self.bias 
        return final_output

    def backward(self, dy, learning_rate, alpha=1e-5):
        x = self.cache_input
        w = self.cache_weight 
        filter_h = self.kernel.shape[0]
        filter_w = self.kernel.shape[1]
        C = self.input_data.shape[3]
        
        if not self.depth_wise: 
            d_out_flat = dy.reshape(-1, w.shape[1])
            dW = x.T @ d_out_flat #[Pixels per batch, F]
            dX = d_out_flat @ w.T #[Patches * Batch Size, pixels per batch]

            dW = dW.reshape(self.kernel.shape)
            

        if self.depth_wise:
            d_out_flat = dy.transpose(3, 0, 1, 2).reshape(C, -1, 1) #[Channels, N * Patches, 1(Filter)]
            dW = x.transpose(0, 2, 1) @ d_out_flat #[Channels, Pixels per Channel per Batch, 1]
            dX = d_out_flat @ w.transpose(0, 2, 1) #[Channels, Batch * Patches, Pixels per Channel per batch]

            # 1. Swap Channels back to the end: (Batch * Patches, Pixels, C) and then fuse it. 
            dX = dX.transpose(1, 2, 0).reshape(-1, filter_h * filter_w * C)

            dW = dW.reshape(C, filter_h, filter_w, 1).transpose(1, 2, 0, 3)
        
        dX = components.col2im(
            dX,
            self.input_data,
            self.kernel,
            padding=self.padding,
            stride=self.stride
            )   
        
        dW += alpha * dW
        
        dB = np.sum(dy, axis=(0, 1, 2))

        self.v_bias = self.sgd_momentum * self.v_bias - learning_rate * dB
        self.v_kernel = self.sgd_momentum * self.v_kernel - learning_rate * dW

        self.bias += self.v_bias
        self.kernel += self.v_kernel
            
        self.cache_input = None
        self.cache_weight = None
        self.input_data = None
        
        # 2. Delete local intermediate matrices
        del x, w, d_out_flat, dW, dB
        
        # 3. Force CuPy to instantly return the VRAM to the pool
        mempool = np.get_default_memory_pool()
        mempool.free_all_blocks()
        
        return dX
    
class Pooling:
    def __init__(self, filter_size, stride=1, padding=0, ptype="max"): 
        
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.type = ptype 
    

    def forward(self, x): 
        self.x = x
        N, H, W, C = self.x.shape

        out_h = (H - self.filter_size + 2 * self.padding) // self.stride + 1
        out_w = (W - self.filter_size + 2 * self.padding) // self.stride + 1

        dummy_kernel = np.empty((self.filter_size, self.filter_size, C, 1))

        cols = components.im2col(
        self.x, 
        dummy_kernel, 
        padding=self.padding, 
        stride=self.stride, 
        depth_wise=True
        )
        # cols shape: (B * Patches, Pixels, C)

        if self.type == 'max':
            pooled_flat = np.max(cols, axis=1) #(N * patches, C)
        elif self.type == 'avg':
            pooled_flat = np.mean(cols, axis=1)
        
        final_output = pooled_flat.reshape(N, out_h, out_w, C)

        self.cols_cache = cols 

        return final_output
    
    def backward(self, dy): 
        N, H, W, C = self.x.shape
        pixels_per_patch = self.filter_size * self.filter_size 

        dY_flat = dy.reshape(-1, 1, C)
        dX_cols = np.zeros_like(self.cols_cache)

        if self.type == 'max':
            # (B * Patches, C)
            max_idx = np.argmax(self.cols_cache, axis=1) 
            
            
            # (B * Patches, 1, C)
            max_idx = np.expand_dims(max_idx, axis=1) 
            

            np.put_along_axis(dX_cols, max_idx, 1.0, axis=1)
            

            dX_cols = dX_cols * dY_flat
        
        elif self.type == 'avg':
            dX_cols = np.ones_like(self.cols_cache) * (dY_flat / pixels_per_patch)
            
        # 4. Format for your universal col2im function
        # Crush Pixels and Channels together to match the col2im expectations
        # Shape: (B * Patches, Pixels_per_Patch * C)
        dX_cols_flat = dX_cols.reshape(-1, pixels_per_patch * C)
        
        # 5. Build the dummy kernel to pass the shape requirements
        dummy_kernel = np.empty((self.filter_size, self.filter_size, C, 1))
        
        # 6. Reassemble the image!
        dX = components.col2im(
            dX_cols_flat, 
            self.x, 
            dummy_kernel, 
            padding=self.padding, 
            stride=self.stride
        )
        
        # ==========================================
        # THE GARBAGE COLLECTOR
        # ==========================================
        # 1. Unbind the matrices from the class instance
        self.x = None
        self.cols_cache = None
        
        # 2. Delete local intermediate matrices
        if self.type == 'max':
            del max_idx
        del dY_flat, dX_cols, dX_cols_flat, dummy_kernel
        
        # 3. Force CuPy to instantly return the VRAM to the pool
        mempool = np.get_default_memory_pool()
        mempool.free_all_blocks()

        return dX

        
class FullyConnected: 
    def __init__(self, d_in, d_out, keep_prob=1.0, sgd_momentum=0.9): 

        self.d_in =  d_in 
        self.d_out = d_out 
        self.keep_prob = keep_prob

        self.bias = np.zeros((1, d_out)) 
        self.weight = components.initialization((d_in, d_out), type="He")
        self.sgd_momentum = sgd_momentum
        self.v_weight = np.zeros_like(self.weight)
        self.v_bias = np.zeros_like(self.bias)

        self.dropout_mask = None

    def forward(self, x, training=True): 
        self.x = x 
        #x.shape = [Batch, Height, Width, Channels] @ [din, D_out]

        self.x_flat = self.x.reshape(self.x.shape[0], -1)
        

        #[Batch, din] @ [din, dOut]
        output = self.x_flat @ self.weight + self.bias

        if training and self.keep_prob < 1.0: 
            mask = np.random.rand(*output.shape) < self.keep_prob

            self.dropout_mask = mask / self.keep_prob

            output *= self.dropout_mask
        else: 
            self.dropout_mask = np.ones_like(output)

        return output #[Batch, dOut]
    
    def backwards(self, dy, learning_rate, alpha=1e-5): 
        dy = dy * self.dropout_mask

        dW = self.x_flat.T @ dy 
        dX = dy @ self.weight.T 
        dB = np.sum(dy, axis=0, keepdims=True)

        dW += alpha * dW
        
        self.v_weight = self.sgd_momentum * self.v_weight - learning_rate * dW
        self.v_bias = self.sgd_momentum * self.v_bias - learning_rate * dB

        self.weight += self.v_weight
        self.bias += self.v_bias

        dX = dX.reshape(self.x.shape)

        # ==========================================
        # THE GARBAGE COLLECTOR
        # ==========================================
        # 1. Unbind the matrices from the class instance
        self.x = None
        self.x_flat = None
        self.dropout_mask = None
        
        # 2. Delete local intermediate matrices
        del dW, dB, dy
        
        # 3. Force CuPy to instantly return the VRAM to the pool
        mempool = np.get_default_memory_pool()
        mempool.free_all_blocks()

        return dX 

class ReLu: 
    def __init__(self, alpha=0):
        self.alpha = alpha 
        self.mask = None
    def forward(self, x): 
        self.mask = (x > 0)
        output = np.where(self.mask, x, x*self.alpha) 
        return output
    
    def backward(self, dy):
        dX = np.where(self.mask, 1, self.alpha)
        dX = dy * dX
        return dX
    
class SoftmaxCrossEntropy:
    def __init__(self):
        self.probabilities = None
        self.y_true = None
    def forward(self, logits, y_true): 
        self.y_true = y_true 

        self.probabilities = components.softmax(logits)

        loss = components.cross_entropy(probabilities=self.probabilities, target_classes=y_true, epsilon=1e-15)
        
        return loss, self.probabilities
    
    def backwards(self):
        batch_size = self.probabilities.shape[0]

        dy = (self.probabilities - self.y_true) / batch_size 

        return dy 


class GlobalAveragePool:
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        # x: (N, H, W, C) -> (N, C)
        self.input_shape = x.shape
        return np.mean(x, axis=(1, 2))

    def backward(self, dy):
        # dy: (N, C) -> (N, H, W, C)
        N, H, W, C = self.input_shape
        dX = np.broadcast_to(dy.reshape(N, 1, 1, C) / (H * W), self.input_shape).copy()
        return dX


class InceptionModule:
    def __init__(self, b1_filters=128, b2_reduce=64, b2_filters=192,
                 b3_reduce=64, b3_filters=96, b4_filters=64, sgd_momentum=0.9):
        self.b1_filters = b1_filters
        self.b2_filters = b2_filters
        self.b3_filters = b3_filters
        self.b4_filters = b4_filters
        self.out_channels = b1_filters + b2_filters + b3_filters + b4_filters

        m = sgd_momentum

        # Branch 1: 1x1 conv -> BN -> ReLU
        self.b1_conv = Convolution(1, 1, b1_filters, True, False, m)
        self.b1_bn = BatchNorm2D(b1_filters, np.ones(b1_filters), np.zeros(b1_filters), sgd_momentum=m)
        self.b1_relu = ReLu()

        # Branch 2: 1x1 reduce -> BN -> ReLU -> 3x3 conv -> BN -> ReLU
        self.b2_reduce_conv = Convolution(1, 1, b2_reduce, True, False, m)
        self.b2_reduce_bn = BatchNorm2D(b2_reduce, np.ones(b2_reduce), np.zeros(b2_reduce), sgd_momentum=m)
        self.b2_reduce_relu = ReLu()
        self.b2_conv = Convolution(3, 3, b2_filters, True, False, m)
        self.b2_bn = BatchNorm2D(b2_filters, np.ones(b2_filters), np.zeros(b2_filters), sgd_momentum=m)
        self.b2_relu = ReLu()

        # Branch 3: 1x1 reduce -> BN -> ReLU -> 5x5 conv -> BN -> ReLU
        self.b3_reduce_conv = Convolution(1, 1, b3_reduce, True, False, m)
        self.b3_reduce_bn = BatchNorm2D(b3_reduce, np.ones(b3_reduce), np.zeros(b3_reduce), sgd_momentum=m)
        self.b3_reduce_relu = ReLu()
        self.b3_conv = Convolution(5, 5, b3_filters, True, False, m)
        self.b3_bn = BatchNorm2D(b3_filters, np.ones(b3_filters), np.zeros(b3_filters), sgd_momentum=m)
        self.b3_relu = ReLu()

        # Branch 4: 3x3 max pool -> 1x1 conv -> BN -> ReLU
        self.b4_pool = Pooling(3, stride=1, padding=1, ptype="max")
        self.b4_conv = Convolution(1, 1, b4_filters, True, False, m)
        self.b4_bn = BatchNorm2D(b4_filters, np.ones(b4_filters), np.zeros(b4_filters), sgd_momentum=m)
        self.b4_relu = ReLu()

    def forward(self, x, training=True):
        # Branch 1
        b1 = self.b1_conv.forward(x, stride=1, padding=0)
        b1 = self.b1_bn.forward(b1, training=training)
        b1 = self.b1_relu.forward(b1)

        # Branch 2
        b2 = self.b2_reduce_conv.forward(x, stride=1, padding=0)
        b2 = self.b2_reduce_bn.forward(b2, training=training)
        b2 = self.b2_reduce_relu.forward(b2)
        b2 = self.b2_conv.forward(b2, stride=1, padding=1)
        b2 = self.b2_bn.forward(b2, training=training)
        b2 = self.b2_relu.forward(b2)

        # Branch 3
        b3 = self.b3_reduce_conv.forward(x, stride=1, padding=0)
        b3 = self.b3_reduce_bn.forward(b3, training=training)
        b3 = self.b3_reduce_relu.forward(b3)
        b3 = self.b3_conv.forward(b3, stride=1, padding=2)
        b3 = self.b3_bn.forward(b3, training=training)
        b3 = self.b3_relu.forward(b3)

        # Branch 4
        b4 = self.b4_pool.forward(x)
        b4 = self.b4_conv.forward(b4, stride=1, padding=0)
        b4 = self.b4_bn.forward(b4, training=training)
        b4 = self.b4_relu.forward(b4)

        # Concatenate along channel axis
        output = np.concatenate([b1, b2, b3, b4], axis=3)
        return output

    def backward(self, dy, learning_rate):
        lr = learning_rate
        c1 = self.b1_filters
        c2 = c1 + self.b2_filters
        c3 = c2 + self.b3_filters

        dy_b1 = dy[:, :, :, :c1]
        dy_b2 = dy[:, :, :, c1:c2]
        dy_b3 = dy[:, :, :, c2:c3]
        dy_b4 = dy[:, :, :, c3:]

        # Branch 1: ReLU -> BN -> Conv
        dy_b1 = self.b1_relu.backward(dy_b1)
        dy_b1 = self.b1_bn.backward(dy_b1, lr)
        dX_b1 = self.b1_conv.backward(dy_b1, lr)

        # Branch 2: ReLU -> BN -> Conv3x3 -> ReLU -> BN -> Conv1x1
        dy_b2 = self.b2_relu.backward(dy_b2)
        dy_b2 = self.b2_bn.backward(dy_b2, lr)
        dy_b2 = self.b2_conv.backward(dy_b2, lr)
        dy_b2 = self.b2_reduce_relu.backward(dy_b2)
        dy_b2 = self.b2_reduce_bn.backward(dy_b2, lr)
        dX_b2 = self.b2_reduce_conv.backward(dy_b2, lr)

        # Branch 3: ReLU -> BN -> Conv5x5 -> ReLU -> BN -> Conv1x1
        dy_b3 = self.b3_relu.backward(dy_b3)
        dy_b3 = self.b3_bn.backward(dy_b3, lr)
        dy_b3 = self.b3_conv.backward(dy_b3, lr)
        dy_b3 = self.b3_reduce_relu.backward(dy_b3)
        dy_b3 = self.b3_reduce_bn.backward(dy_b3, lr)
        dX_b3 = self.b3_reduce_conv.backward(dy_b3, lr)

        # Branch 4: ReLU -> BN -> Conv1x1 -> Pool
        dy_b4 = self.b4_relu.backward(dy_b4)
        dy_b4 = self.b4_bn.backward(dy_b4, lr)
        dy_b4 = self.b4_conv.backward(dy_b4, lr)
        dX_b4 = self.b4_pool.backward(dy_b4)

        # All branches share the same input
        dX = dX_b1 + dX_b2 + dX_b3 + dX_b4
        return dX
    
