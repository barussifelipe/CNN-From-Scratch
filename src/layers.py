import numpy as np
import components 
class BatchNorm2D:
    def __init__(self, channels, gamma, beta, alpha=0.1):
        self.running_mean = np.zeros(channels)
        self.running_variance = np.ones(channels)
        
        # alpha in your formula
        self.momentum = alpha
        self.gamma = gamma 
        self.beta = beta
        self.batch_cache = None
        self.gamma_cache = None
        self.beta_cache = None

    def forward(self, batch, training=True, e=1e-5):
        if training:

            mean = np.mean(batch, axis=(0, 1, 2))
            variance = np.var(batch, axis=(0, 1, 2))

            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_variance = self.momentum * variance + (1 - self.momentum) * self.running_variance

            batch_normalized = (batch - mean) / np.sqrt(variance + e)
            
        else:
            batch_normalized = (batch - self.running_mean) / np.sqrt(self.running_variance + e)

        self.batch_cache = batch_normalized
        self.gamma_cache = self.gamma
        self.beta_cache = self.beta

        y = self.gamma * batch_normalized + self.beta


        
        return y
    
    def backward(self, d_out, learning_rate): 

        batch = self.batch_cache
        gamma = self.gamma_cache
        beta = self.beta_cache

        dX = gamma 
        dG = batch
        dB = np.sum(d_out, axis=(0, 1, 2))

        self.gamma -= dG * learning_rate
        self.beta -= dB * learning_rate
        
        return dX

    

class Convolution:
    def __init__(self, x, filter_h, filter_w, filters, bias, depth_wise):
        self.input_data = x
        self.bias = bias 
        self.depth_wise = depth_wise
        
        channels = self.input_data[3]

        if self.depth_wise: 
            kernel_shape = (filter_h, filter_w, channels, 1)
        else: 
            kernel_shape = (filter_h, filter_w, channels, filters)

        self.kernel = components.initialization(kernel_shape=kernel_shape, type="He") #We using ReLu on convs. 
        bias_depth = channels if self.depth_wise else filters 
        self.bias = np.zeros(bias_depth)
        self.cache_input = None
        self.cache_weight = None 
        self.padding = None
        self.stride = None
        
    def forward(self, stride, padding): 

        N = self.input_data.shape[0]
        C = self.input_data.shape[3]
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
            self.cache_input = self.x_flat #[patches * batch_size, pixels per batch]
            self.cache_weight = weight_flat #[height * width * channels, filters]
            
        else: 
            x_batched = x_flat.transpose(2, 0, 1)
            w_batched = self.kernel.transpose(2, 0, 1, 3).reshape(C, -1, 1)

            output = x_batched @ w_batched
            output = output.reshape(C, N, kernel_passes_h, kernel_passes_w)
            self.cache_input = self.x_batched #[C, patches * batch_size, pixels per channel per batch]
            self.cache_weight = w_batched #[C, height * width, 1]
            

            final_output = output.transpose(1, 2, 3, 0) #(N, H_out, W_out, C)

        
        final_output += self.bias 
        return final_output

    def backward(self, d_out, learning_rate):
        x = self.cache_input
        w = self.cache_weight 
        filter_h = self.kernel.shape[0]
        filter_w = self.kernel.shape[1]
        C = self.input_data.shape[3]
        
        if not self.depth_wise: 
            d_out_flat = d_out.reshape(-1, w.shape[1])
            dW = x.T @ d_out_flat #[Pixels per batch, F]
            dX = d_out_flat @ w.T #[Patches * Batch Size, pixels per batch]

            dW = dW.reshape(self.kernel.shape)
            

        if self.depth_wise:
            d_out_flat = d_out.transpose(3, 0, 1, 2).reshape[C, -1, 1] #[Channels, N * Patches, 1(Filter)]
            dW = x.transpose(0, 2, 1) @ d_out_flat #[Channels, Pixels per Channel per Batch, 1]
            dX = d_out_flat @ w.tranpose(0, 2, 1) #[Channels, Batch * Patches, Pixels per Channel per batch]

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
        
        dB = np.sum(d_out, axis=(0, 1, 2))

        self.bias -= learning_rate * dB 
        self.kernel -= learning_rate * dW
            
        
        return dX

        
  
