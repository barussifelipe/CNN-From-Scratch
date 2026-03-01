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
    def __init__(self, x, kernel, kernel_depth, bias):
        self.input_data = x
        self.kernel = kernel 
        self.kernel_depth = kernel_depth
        self.bias = bias 
        self.cache = None
        
    def forward(self, stride, padding, depth_wise): 
        output = components.convolution(x=self.input_data, kernel=self.kernel, bias=self.bias, stride=stride,   padding=padding, depth_wise=depth_wise)
        if depth_wise: 
            output = components.convolution(x=output, kernel=self.kernel_depth, bias=self.bias, stride=stride,   padding=padding, depth_wise=False)

        self.cache = self.input_data

    def backward(self, d_out, learning_rate):
        x = self.cache 

        d_weights = 
