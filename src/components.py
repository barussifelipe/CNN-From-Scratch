import numpy as np 
import matplotlib.pyplot as plt

def convolution(x, kernel, stride=1, padding=0, depth_wise=False): 
    #kernel = [height, width, channels, filters]
    kernel_passes_x = (x.shape[0] - kernel.shape[0] + 2 * padding) // stride + 1 
    kernel_passes_y = (x.shape[1] - kernel.shape[1] + 2 * padding) // stride + 1
    channels = x.shape[2]
    filters = kernel.shape[3] if len(kernel.shape) == 4 else 1
    padded_x = padding_image(x, padding)

    if depth_wise: 
        output = np.zeros((kernel_passes_x, kernel_passes_y, channels))
        filters = 1
    else: 
        output = np.zeros((kernel_passes_x, kernel_passes_y, filters))

    for filter in range(filters):
        current_kernel = kernel[:, :, :, filter] if len(kernel.shape) == 4 else kernel
        try: 
            for i in range(kernel_passes_x):
                for j in range(kernel_passes_y):
                    

                    h_start = i * stride
                    h_end = h_start + current_kernel.shape[0]

                    w_start = j * stride
                    w_end = w_start + current_kernel.shape[1]

                    image_patch = padded_x[h_start:h_end, w_start:w_end, :]

                    if depth_wise: 
                        output[i, j, :] = np.sum(image_patch * current_kernel, axis=(0,1))
                    else:
                        output[i, j, filter] = np.sum(image_patch * current_kernel)

        except Exception as e: 
            print(f"Error on performing convolution on filter: {filter}, error:{e}")
    
    if depth_wise: 
        parameters = kernel.shape[0] * kernel.shape[1] * channels
        memory = output.shape[0] * output.shape[1] * channels

    else: 
        parameters = kernel.shape[0] * kernel.shape[1] * channels * filters
        memory = output.shape[0] * output.shape[1] * filters
    
    return output, parameters/1000, memory/1000 #remember to set memory to inputsize = x.shape[0] * x.shape[1] * x.shape[2]. parameters = 0. 


def pooling(x, filter_size=2, stride=1, type="max"):
    filter_passes_x = (x.shape[0] - filter_size) // stride + 1 
    filter_passes_y = (x.shape[1] - filter_size) // stride + 1 
    channels = x.shape[2]
    try: 
        output = np.zeros((filter_passes_x, filter_passes_y, channels))

        for i in range(filter_passes_x):
            for j in range(filter_passes_y):
                h_start = i * stride
                h_end = h_start + filter_size

                w_start = j * stride
                w_end = w_start + filter_size

                image_patch = x[h_start:h_end, w_start:w_end, :]

                if type == "max": 
                    output[i, j, :] = np.max(image_patch, axis=(0,1))
                elif type == "avg": 
                    output[i, j, :] = np.mean(image_patch, axis=(0,1))

    except Exception as e: 
        print(f"Error on performing pooling, error: {e}")
    
    memory = output.shape[0] * output.shape[1] * channels

    return output, memory 




def padding_image(x, pad_width):
    np.pad(x, pad_width= ((pad_width, pad_width), (pad_width, pad_width), (0, 0)))
    return 

def relu(x): 
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01): 
    return np.where(x > 0, x, x * alpha)

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

    