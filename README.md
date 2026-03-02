# CNN From Scratch

## 1. Purpose of the Project
The primary purpose of this project is to build a Convolutional Neural Network (CNN) entirely from scratch using only Python and NumPy. This project is aimed at gaining a deep, under-the-hood understanding of deep learning concepts—including forward propagation, backpropagation, and mathematical optimization—without relying on high-level frameworks like PyTorch or TensorFlow.

## 2. Project Structure
To maintain a clean and modular codebase, the project is divided into three main files. Currently, the mathematical blocks and layer definitions exist, and a third file (`cnn.py`) will be created to house the main neural network architecture and training loop.

- `src/components.py`: Core mathematical operations, activation functions, and helper algorithms.
- `src/layers.py`: Object-oriented layer definitions that will comprise the neural network.
- `src/cnn.py`: (To be created) The CNN architecture implementation, dataset handling, and training execution.

## 3. File Contents and Implementation Details

### `src/components.py`
This file implements the essential, low-level mathematical operations that power the deep learning layers.
- **Indexing and Convolutions:** Convolutions are built leveraging the `im2col` (image-to-column) and `col2im` algorithms. By reshaping images and filters into 2D matrices using indexing, convolution operations are transformed into simple Matrix Multiplications (GEMM). This massively speeds up the computation by making full use of NumPy's highly optimized C backend.
- **Activations and Losses:** Implements functions like ReLU, Leaky ReLU, Softmax, and Cross-Entropy loss.
- **Hyperparameters:** Defines operations involving hyperparameters like `alpha` (momentum), `beta`, and `gamma` (such as in Batch Normalization).

### `src/layers.py`
This file contains the structural definitions of the network layers that will be used to construct the CNN.
- **Layer Implementation:** Implements layers like `Convolution` and `BatchNorm2D` with explicit forward and backward passes.
- **Extensibility:** The CNN will be built using the layers defined here, and this will include the implementation of Fully Connected (FC) layers used for final classification.

### `src/cnn.py` (Planned)
This file will define the overarching CNN architecture and orchestrate the training process.
- **Dataset:** We will train our CNN architecture using the **ImageNet** dataset.
- **Training and Regularization:** The training process will utilize a Dropout rate of `0.5` to aggressively regularize the network and prevent overfitting.
- **Visualization:** We will systematically plot the training process per epoch. This includes visually checking the training loss, validation set loss, and validation accuracy over time.

## 4. How to Use This Repository

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd CNN_from_scratch
   ```

2. **Install the dependencies:**
   The project primarily depends on NumPy (for matrix math) and Matplotlib (for visualization).
   ```bash
   pip install numpy matplotlib
   ```

3. **Run the training script:**
   Once `cnn.py` is implemented and the ImageNet dataset is configured, you can start the training loop by running:
   ```bash
   python src/cnn.py
   ```

## 5. Further Work
While the current scope focuses heavily on Computer Vision and Convolutional Neural Networks, future plans for this repository include expanding its capabilities to handle sequential data and modern NLP architectures:
- Implementing **Recurrent Neural Networks (RNNs)** from scratch.
- Implementing **Transformers** (incorporating Multi-Head Self-Attention mechanisms) entirely from scratch using NumPy.
