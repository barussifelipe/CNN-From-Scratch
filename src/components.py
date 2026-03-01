import numpy as np 

def get_conv_indices(x, kernel, padding=1, stride=1): 
    x_shape = x.shape
    N, H, W, C = x_shape

    filter_h = kernel.shape[0]
    filter_w = kernel.shape[1]

    kernel_passes_h = (H - filter_h + 2 * padding) // stride + 1 
    kernel_passes_w = (W - filter_w + 2 * padding) // stride + 1

    i0 = np.repeat(np.arange(filter_h), filter_w)
    i0 = np.tile(i0, C)
    j0 = np.tile(np.arange(filter_w), filter_h * C)
    i1 = stride * np.repeat(np.arange(kernel_passes_h), kernel_passes_w)
    j1 = stride * np.tile(np.arange(kernel_passes_w), kernel_passes_h)

def convolution(x, kernel, bias, stride=1, padding=0, depth_wise=False): 
    #x = [batch, height, width, channels]
    #kernel = [height, width, channels, filters]
    kernel_passes_x = (x.shape[1] - kernel.shape[0] + 2 * padding) // stride + 1 
    kernel_passes_y = (x.shape[2] - kernel.shape[1] + 2 * padding) // stride + 1
    channels = x.shape[3]
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
                        output[i, j, :] = np.sum(image_patch * current_kernel, axis=(0,1)) + bias
                    else:
                        output[i, j, filter] = np.sum(image_patch * current_kernel) + bias

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
    
def initialization(kernel, type="Xa"): 
    kernel_shape = kernel.shape
    if len(kernel_shape) == 4:
        fan_in = kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
    else: 
        fan_in = kernel_shape[0] #In_features
    
    if type == "Xa": 
        weights = np.random.randn(*kernel_shape) / np.sqrt(1 / fan_in)
    elif type == "He":
        weights = np.random.randn(*kernel_shape) * np.sqrt(2 / fan_in) #If ReLU
                                                
    return weights 

def softmax(scores): 
    #scores_shape = (Batch_size, Num_classes)
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)

    exp_scores = np.exp(shifted_scores)

    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probabilities

def cross_entropy(probabilities, target_classes, epsilon=1e-5, alpha=1e-2): 
    batch_size = probabilities.shape[0]

    correct_class_probs = probabilities[np.arange(batch_size), target_classes]

    correct_class_probs = np.clip(correct_class_probs, epsilon, 1 - epsilon)

    loss_vector = -np.log(correct_class_probs) 

    average_loss = np.mean(loss_vector) 

    return average_loss

def fully_connected(x, weights, bias):
    x_flattened = np.reshape(x.shape[0], -1)

    output = x_flattened @ weights + bias
    return output

def res_con(f_x, x): 
    return f_x + x 

