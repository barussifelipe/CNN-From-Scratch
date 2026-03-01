import numpy as np
import components 
class BatchNorm2D:
    def __init__(self, channels, alpha=0.1):
        self.running_mean = np.zeros(channels)
        self.running_variance = np.ones(channels)
        
        # alpha in your formula
        self.momentum = alpha

    def forward(self, batch, gamma, beta, training=True, e=1e-5):
        if training:

            mean = np.mean(batch, axis=(0, 1, 2))
            variance = np.var(batch, axis=(0, 1, 2))

            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_variance = self.momentum * variance + (1 - self.momentum) * self.running_variance

            batch_normalized = (batch - mean) / np.sqrt(variance + e)
            
        else:
            batch_normalized = (batch - self.running_mean) / np.sqrt(self.running_variance + e)

        y = gamma * batch_normalized + beta
        
        return y
    

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
        self.cache = None
        
    def forward(self, stride, padding): 

        N = self.input_data.shape[0]
        C = self.input_data.shape[3]

        filters = self.kernel.shape[3]
        filter_h = self.kernel.shape[0]
        filter_w = self.kernel.shape[1]
        
        kernel_passes_h = (self.input_data.shape[1] - filter_h + 2 * padding) // stride + 1 
        kernel_passes_w = (self.input_data.shape[2] - filter_w + 2 * padding) // stride + 1

        x_flat = components.im2col(self.input_data, self.kernel, padding=padding, stride=stride, depth_wise=self.depth_wise)

        if not self.depth_wise:
            weight_flat = self.kernel.reshape(-1, filters)

            output = x_flat @ weight_flat
            final_output = output.reshape(-1, kernel_passes_h, kernel_passes_w, filters)
        else: 
            x_batched = x_flat.transpose(2, 0, 1)
            w_batched = self.kernel.transpose(2, 0, 1, 3).reshape(C, -1, 1)

            output = x_batched @ w_batched

            output = output.reshape(C, N, kernel_passes_h, kernel_passes_w)

            final_output = output.transpose(1, 2, 3, 0)

        self.cache = self.input_data
        final_output += self.bias 
        return final_output

    def backward(self, d_out, learning_rate):
        x = self.cache 

    
