import cupy as np 
import matplotlib.pyplot as plt 
import layers
import components
import os 
import pickle 

class CNN:
    def __init__(self, input_channels=3, num_classes=100, sgd_momentum=0.9):
        m = sgd_momentum

        # ================ ResBlock ================
        # conv3x3x64 -> BN -> ReLU
        self.res_conv1 = layers.Convolution(3, 3, 64, True, False, m)
        self.res_bn1   = layers.BatchNorm2D(64, np.ones(64), np.zeros(64), sgd_momentum=m)
        self.res_relu1 = layers.ReLu()
        # conv3x3x128 -> BN -> ReLU
        self.res_conv2 = layers.Convolution(3, 3, 128, True, False, m)
        self.res_bn2   = layers.BatchNorm2D(128, np.ones(128), np.zeros(128), sgd_momentum=m)
        self.res_relu2 = layers.ReLu()
        # conv3x3x256 -> BN -> ReLU
        self.res_conv3 = layers.Convolution(3, 3, 256, True, False, m)
        self.res_bn3   = layers.BatchNorm2D(256, np.ones(256), np.zeros(256), sgd_momentum=m)
        self.res_relu3 = layers.ReLu()
        # maxpool 2x2
        self.res_pool  = layers.Pooling(2, stride=2, ptype="max")
        # Shortcut: project + downsample
        self.res_shortcut      = layers.Convolution(1, 1, 256, True, False, m)
        self.res_shortcut_pool = layers.Pooling(2, stride=2, ptype="avg")

        # Post-ResBlock BN -> ReLU
        self.post_res_bn   = layers.BatchNorm2D(256, np.ones(256), np.zeros(256), sgd_momentum=m)
        self.post_res_relu = layers.ReLu()

        # ================ Inception 1 (256 -> 480) ================
        self.inception1 = layers.InceptionModule(sgd_momentum=m)       # out 480
        self.inc1_proj  = layers.Convolution(1, 1, 480, True, False, m) # shortcut 256->480

        # ================ ResBlock transition 1->2 (480 -> 480) ================
        #self.trans12_conv = layers.Convolution(3, 3, 480, True, False, m)
        #self.trans12_bn   = layers.BatchNorm2D(480, np.ones(480), np.zeros(480), sgd_momentum=m)
        #self.trans12_relu = layers.ReLu()

        # ================ Inception 2 (480 -> 480) ================
        self.inception2 = layers.InceptionModule(sgd_momentum=m)

        # ================ ResBlock transition 2->3 (480 -> 480) ================
        #self.trans23_conv = layers.Convolution(3, 3, 480, True, False, m)
        #self.trans23_bn   = layers.BatchNorm2D(480, np.ones(480), np.zeros(480), sgd_momentum=m)
        #self.trans23_relu = layers.ReLu()

        # ================ Inception 3 (480 -> 480) ================
        self.inception3 = layers.InceptionModule(sgd_momentum=m)

        # ================ Global Average Pool ================
        self.gap = layers.GlobalAveragePool()

        # ================ Post-GAP BN -> ReLU ================
        self.post_gap_bn   = layers.BatchNorm2D(480, np.ones(480), np.zeros(480), sgd_momentum=m)
        self.post_gap_relu = layers.ReLu()

        # ================ FC head ================
        self.fc1 = layers.FullyConnected(480, 256, sgd_momentum=m)
        self.fc2 = layers.FullyConnected(256, num_classes, sgd_momentum=m)

        # ================ Loss ================
        self.criterion = layers.SoftmaxCrossEntropy()

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, x, y_true=None, training=True):
        # --- ResBlock main path: conv->BN->ReLU x3 + pool ---
        out = self.res_conv1.forward(x, stride=1, padding=1)
        out = self.res_bn1.forward(out, training=training)
        out = self.res_relu1.forward(out)

        out = self.res_conv2.forward(out, stride=1, padding=1)
        out = self.res_bn2.forward(out, training=training)
        out = self.res_relu2.forward(out)

        out = self.res_conv3.forward(out, stride=1, padding=1)
        out = self.res_bn3.forward(out, training=training)
        out = self.res_relu3.forward(out)

        out = self.res_pool.forward(out)

        # --- ResBlock shortcut ---
        shortcut = self.res_shortcut.forward(x, stride=1, padding=0)
        shortcut = self.res_shortcut_pool.forward(shortcut)

        out = components.res_con(out, shortcut)

        # --- Post-ResBlock BN -> ReLU ---
        out = self.post_res_bn.forward(out, training=training)
        out = self.post_res_relu.forward(out)

        # --- Inception 1 (projection residual: 256 -> 480) ---
        inc1_out  = self.inception1.forward(out, training=training)
        inc1_proj = self.inc1_proj.forward(out, stride=1, padding=0)
        out = components.res_con(inc1_out, inc1_proj)

        # --- ResBlock transition 1->2: conv->BN->ReLU + identity ---
        #self.trans12_identity = out
        #trans_out = self.trans12_conv.forward(out, stride=1, padding=1)
        #trans_out = self.trans12_bn.forward(trans_out, training=training)
        #trans_out = self.trans12_relu.forward(trans_out)
        #out = components.res_con(trans_out, self.trans12_identity)

        # --- Inception 2 (identity residual: 480 -> 480) ---
        self.inc2_identity = out
        inc2_out = self.inception2.forward(out, training=training)
        out = components.res_con(inc2_out, self.inc2_identity)

        # --- ResBlock transition 2->3: conv->BN->ReLU + identity ---
        #self.trans23_identity = out
        #trans_out = self.trans23_conv.forward(out, stride=1, padding=1)
        #trans_out = self.trans23_bn.forward(trans_out, training=training)
        #trans_out = self.trans23_relu.forward(trans_out)
        #out = components.res_con(trans_out, self.trans23_identity)

        # --- Inception 3 (identity residual: 480 -> 480) ---
        self.inc3_identity = out
        inc3_out = self.inception3.forward(out, training=training)
        out = components.res_con(inc3_out, self.inc3_identity)

        # --- Global Average Pool -> (N, 480) ---
        out = self.gap.forward(out)

        # --- Post-GAP BN -> ReLU (reshape to 4-D for BN2D) ---
        N = out.shape[0]
        out = out.reshape(N, 1, 1, -1)
        out = self.post_gap_bn.forward(out, training=training)
        out = out.reshape(N, -1)
        out = self.post_gap_relu.forward(out)

        # --- FC head ---
        out = self.fc1.forward(out, training=training)
        out = self.fc2.forward(out, training=training)

        # --- Loss ---
        if y_true is not None:
            loss, probs = self.criterion.forward(out, y_true)
            return loss, probs
        return components.softmax(out)

    # ------------------------------------------------------------------ #
    #  Backward                                                           #
    # ------------------------------------------------------------------ #
    def backward(self, learning_rate):
        lr = learning_rate
        dy = self.criterion.backwards()

        # FC head
        dy = self.fc2.backwards(dy, lr)
        dy = self.fc1.backwards(dy, lr)

        # Post-GAP ReLU -> BN
        dy = self.post_gap_relu.backward(dy)
        N = dy.shape[0]
        dy = dy.reshape(N, 1, 1, -1)
        dy = self.post_gap_bn.backward(dy, lr)
        dy = dy.reshape(N, -1)

        # GAP
        dy = self.gap.backward(dy)

        # Inception 3 (identity residual)
        dy_inc3 = self.inception3.backward(dy, lr)
        dy = dy_inc3 + dy

        # ResBlock transition 2->3 (identity residual)
        #dy_trans23 = self.trans23_relu.backward(dy)
        #dy_trans23 = self.trans23_bn.backward(dy_trans23, lr)
        #dy_trans23 = self.trans23_conv.backward(dy_trans23, lr)
        #dy = dy_trans23 + dy

        # Inception 2 (identity residual)
        dy_inc2 = self.inception2.backward(dy, lr)
        dy = dy_inc2 + dy

        # ResBlock transition 1->2 (identity residual)
        #dy_trans12 = self.trans12_relu.backward(dy)
        #dy_trans12 = self.trans12_bn.backward(dy_trans12, lr)
        #dy_trans12 = self.trans12_conv.backward(dy_trans12, lr)
        #dy = dy_trans12 + dy

        # Inception 1 (projection residual)
        dy_inc1 = self.inception1.backward(dy, lr)
        dy_proj  = self.inc1_proj.backward(dy, lr)
        dy = dy_inc1 + dy_proj

        # Post-ResBlock ReLU -> BN
        dy = self.post_res_relu.backward(dy)
        dy = self.post_res_bn.backward(dy, lr)

        # ResBlock main path: pool -> (ReLU->BN->conv) x3
        dy_main = self.res_pool.backward(dy)
        dy_main = self.res_relu3.backward(dy_main)
        dy_main = self.res_bn3.backward(dy_main, lr)
        dy_main = self.res_conv3.backward(dy_main, lr)
        dy_main = self.res_relu2.backward(dy_main)
        dy_main = self.res_bn2.backward(dy_main, lr)
        dy_main = self.res_conv2.backward(dy_main, lr)
        dy_main = self.res_relu1.backward(dy_main)
        dy_main = self.res_bn1.backward(dy_main, lr)
        dy_main = self.res_conv1.backward(dy_main, lr)

        # ResBlock shortcut path
        dy_short = self.res_shortcut_pool.backward(dy)
        dy_short = self.res_shortcut.backward(dy_short, lr)

        return dy_main + dy_short

    # ------------------------------------------------------------------ #
    #  Parameter & memory counting                                        #
    # ------------------------------------------------------------------ #
    def _conv_params(self, conv):
        if conv.kernel is None:
            return 0
        return conv.kernel.size + conv.bias.size

    def _bn_params(self, bn):
        return bn.gamma.size + bn.beta.size

    def _fc_params(self, fc):
        return fc.weight.size + fc.bias.size

    def _inception_params(self, inc):
        total = 0
        for name in dir(inc):
            attr = getattr(inc, name)
            if isinstance(attr, layers.Convolution):
                total += self._conv_params(attr)
            elif isinstance(attr, layers.BatchNorm2D):
                total += self._bn_params(attr)
        return total

    def count_parameters(self):
        total = 0
        # ResBlock convs + BNs
        for conv in [self.res_conv1, self.res_conv2, self.res_conv3, self.res_shortcut]:
            total += self._conv_params(conv)
        for bn in [self.res_bn1, self.res_bn2, self.res_bn3, self.post_res_bn]:
            total += self._bn_params(bn)
        # Inception modules
        total += self._conv_params(self.inc1_proj)
        for inc in [self.inception1, self.inception2, self.inception3]:
            total += self._inception_params(inc)
        # Transition ResBlocks
        #for conv in [self.trans12_conv, self.trans23_conv]:
            #total += self._conv_params(conv)
        #for bn in [self.trans12_bn, self.trans23_bn]:
            #total += self._bn_params(bn)
        # Post-GAP BN
        total += self._bn_params(self.post_gap_bn)
        # FC
        total += self._fc_params(self.fc1)
        total += self._fc_params(self.fc2)
        return total

    def estimate_activation_memory(self, input_shape):
        """Estimate peak activation memory in bytes (float64) for one forward pass."""
        N, H, W, C = input_shape
        elems = 0
        # ResBlock main path
        elems += N * H * W * 64       # conv1
        elems += N * H * W * 128      # conv2
        elems += N * H * W * 256      # conv3
        H2, W2 = H // 2, W // 2
        elems += N * H2 * W2 * 256    # pool
        elems += N * H2 * W2 * 256    # shortcut
        elems += N * H2 * W2 * 256    # res add + post BN/ReLU
        # 3x Inception + 2x transition (all at H2, W2, 480)
        for _ in range(3):
            elems += N * H2 * W2 * 480 # inception output
            elems += N * H2 * W2 * 480 # residual add
        #for _ in range(2):
            #elems += N * H2 * W2 * 480 # transition conv+BN+ReLU
            #elems += N * H2 * W2 * 480 # transition residual add
        # GAP + post BN/ReLU + FC
        elems += N * 480
        elems += N * 256
        elems += N * 100
        bytes_per_elem = 2  # float32
        return elems * bytes_per_elem

    def print_summary(self, input_shape):
        params = self.count_parameters()
        mem = self.estimate_activation_memory(input_shape)
        print(f"Total parameters:     {params:>12,}")
        print(f"  ({params / 1e6:.2f} M)")
        print(f"Activation memory:    {mem:>12,} bytes")
        print(f"  ({mem / (1024**2):.1f} MB for batch={input_shape[0]})")

    # ------------------------------------------------------------------ #
    #  Training loop                                                      #
    # ------------------------------------------------------------------ #
    def process_cifar_labels(self, labels, num_classes=100):
        N = len(labels)
            
        # 2. Create a massive blank canvas of zeros: Shape (N, 100)
        one_hot = np.zeros((N, num_classes))
        
        # 3. The Vectorized Trick: Drop a '1.0' into the correct column for every row
        one_hot[np.arange(N), labels] = 1.0
    
        return one_hot

    def train_step(self, x, y_true, learning_rate):
        loss, probs = self.forward(x, y_true, training=True)
        self.backward(learning_rate)
        return loss, probs

    def predict(self, x):
        return self.forward(x, y_true=None, training=False)

    def predict_batched(self, X, batch_size=32):
        all_probs = []
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size]
            probs = self.predict(xb)
            all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)


    @staticmethod
    def accuracy(probs, y_true):
        # 1. Get integer predictions: Shape (N,)
        preds = np.argmax(probs, axis=1) 
        
        # 2. If y_true is a 2D one-hot matrix, crush it back to 1D integers
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)
            
        # 3. Compare them cleanly
        return float(np.mean(preds == y_true))
    
    def augment_batch(self, xb):
        B, H, W, C = xb.shape

        # ==========================================
        # 1. Random Horizontal Flip (50% chance)
        # ==========================================
        flip_mask = np.random.rand(B) > 0.5
        xb[flip_mask] = xb[flip_mask, :, ::-1, :]

        # ==========================================
        # 2. Pad & Random Crop (Translation & Scale)
        # ==========================================
        # Pad 4 pixels on height and width dimensions with zeros (black)
        padded = np.pad(xb, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='constant', constant_values=0)
        
        cropped_xb = np.zeros_like(xb)
        
        # Generate random start coordinates for the crop (from 0 to 8)
        y_starts = np.random.randint(0, 9, size=B) 
        x_starts = np.random.randint(0, 9, size=B)
        
       
        for i in range(B):
            y = int(y_starts[i])
            x = int(x_starts[i])
            cropped_xb[i] = padded[i, y:y+H, x:x+W, :]
            
        xb = cropped_xb

        return xb

    def train(self, X_train, y_train, X_val, y_val,
              epochs, batch_size, learning_rate, patience=10):
        history = {"train_loss": [], "train_acc": [], "val_acc": []}
        N = X_train.shape[0]
        batch = X_train[[batch_size], :, :, :]
        
        self.print_summary(batch.shape)

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))

        best_val_acc = 0.0
        epochs_without_improvement = 0 

        for epoch in range(epochs):
            # Shuffle
            epoch_loss = 0.0
            epoch_correct = 0

            for i in range(0, N, batch_size):
               


                xb = X_train[i:i + batch_size].copy()
                yb = y_train[i:i + batch_size]

                xb = self.augment_batch(xb)

                yb_onehot = self.process_cifar_labels(yb)

                loss, probs = self.train_step(xb, yb_onehot, learning_rate)
                epoch_loss += loss * xb.shape[0]
                epoch_correct += np.sum(np.argmax(probs, axis=1) == yb)

                print(f"Batch {i + 1}/{N} of epoch {epoch + 1}/{epochs} ")

            train_loss = epoch_loss / N
            train_acc  = epoch_correct / N

            # Validation (batched to avoid OOM)
            val_probs = self.predict_batched(X_val, batch_size=batch_size)
            val_acc   = self.accuracy(val_probs, y_val)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                # Bundle the hyperparams used for this specific run
                current_hyperparams = {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                }
                
                self.save_checkpoint(val_acc, current_hyperparams, epoch + 1)
                epochs_without_improvement = 0
            else: 
                epochs_without_improvement += 1 
                print(f"No improvement for {epochs_without_improvement} epochs.")

            if epochs_without_improvement >= patience:
                print(f"\n[EARLY STOPPING] Validation accuracy hasn't improved in {patience} epochs.")
                print(f"Stopping training at Epoch {epoch + 1}. Best Acc: {best_val_acc:.4f}")
                break


            # If train_loss and train_acc are still CuPy arrays, leave their .get()
            history["train_loss"].append(float(train_loss.get() if hasattr(train_loss, 'get') else train_loss))
            history["train_acc"].append(float(train_acc.get() if hasattr(train_acc, 'get') else train_acc))
            
            # val_acc is already a standard float now!
            history["val_acc"].append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs} "
                  f"- loss: {train_loss:.4f} "
                  f"- train_acc: {train_acc:.4f} "
                  f"- val_acc: {val_acc:.4f}")
            # ==========================================
            # 2. LIVE PLOT REFRESH
            # ==========================================
            current_epochs = range(1, len(history["train_loss"]) + 1)
            
            # WIPE the canvases clean
            ax1.clear()
            ax2.clear()

            # REDRAW Loss
            ax1.plot(current_epochs, history["train_loss"], label="Train Loss", color="blue")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training Loss per Epoch")
            ax1.legend()
            ax1.grid(True)

            # REDRAW Accuracy
            ax2.plot(current_epochs, history["train_acc"], label="Train Accuracy", color="green")
            ax2.plot(current_epochs, history["val_acc"], label="Val Accuracy", color="orange")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.set_title("Accuracy per Epoch")
            ax2.legend()
            ax2.grid(True)

            # ==========================================
            # Force the GUI to flush and redraw
            # ==========================================
            # 1. Force the canvas to physically draw the new lines
            fig.canvas.draw()
            
            # 2. Force Windows to process all pending GUI events (unfreezes the window)
            fig.canvas.flush_events()
            
            # 3. Increase the pause slightly (0.1 seconds instead of 0.01)
            plt.pause(0.1)
        lr = current_hyperparams.get("learning_rate", 0)
        bs = current_hyperparams.get("batch_size", 0)

        os.makedirs("models_image", exist_ok=True)
        fig.savefig(f"models_image/live_training_history_lr_{lr}_bs{bs}_epochs{epochs}.png", bbox_inches='tight')

        plt.ioff()
        plt.show() 
        return history

    @staticmethod
    def plot_history(history, save_path=None):
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(epochs, history["train_loss"], label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss per Epoch")
        ax1.legend()
        ax1.grid(True)

        # Accuracy
        ax2.plot(epochs, history["train_acc"], label="Train Accuracy")
        ax2.plot(epochs, history["val_acc"],   label="Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy per Epoch")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_checkpoint(self, val_acc, hyperparams, epoch):
        # 1. Create the master dictionary
        checkpoint = {
            "hyperparameters": hyperparams,
            "val_acc": val_acc,
            "epoch": epoch,
            "layer_weights": {} # Changed to a dictionary!
        }
        all_layers = {}

        for name, layer in vars(self).items():
            all_layers[name] = layer
            # If the layer is an Inception block, open it and grab its inner layers!
            if isinstance(layer, layers.InceptionModule):
                for sub_name, sub_layer in vars(layer).items():
                    all_layers[f"{name}.{sub_name}"] = sub_layer
        
        # 2. Dynamically scan every attribute in the Roma class
        for layer_name, layer in all_layers.items():
            state = {}

            def to_cpu(tensor):
                return tensor.get() if hasattr(tensor, 'get') else tensor
            
            if hasattr(layer, 'kernel'):
                state['kernel'] = to_cpu(layer.kernel)
                if hasattr(layer, 'v_kernel'): 
                    state['v_kernel'] = to_cpu(layer.v_kernel)
                    
            if hasattr(layer, 'weight'):
                state['weight'] = to_cpu(layer.weight)
                if hasattr(layer, 'v_weight'): 
                    state['v_weight'] = to_cpu(layer.v_weight)
                    
            if hasattr(layer, 'bias'):
                state['bias'] = to_cpu(layer.bias)
                if hasattr(layer, 'v_bias'): 
                    state['v_bias'] = to_cpu(layer.v_bias)
                    
            if hasattr(layer, 'gamma'):
                state['gamma'] = to_cpu(layer.gamma)
                state['beta'] = to_cpu(layer.beta)
                state['running_mean'] = to_cpu(layer.running_mean)
                state['running_variance'] = to_cpu(layer.running_variance)
                
            # If we actually found weights in this attribute, save it
            if state:
                checkpoint["layer_weights"][layer_name] = state
                
        # 3. Create the automated filename
        lr = hyperparams.get("learning_rate", 0)
        bs = hyperparams.get("batch_size", 0)
        
        os.makedirs("checkpoints", exist_ok=True) 
        filename = f"checkpoints/roma_lr{lr}_bs{bs}_epoch{epoch}_acc{val_acc:.4f}.pkl"
        
        # 4. Save to disk
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        old_file = getattr(self, "last_saved_checkpoint", None)
        if old_file and os.path.exists(old_file) and old_file != filename:
            os.remove(old_file)
            
        # Remember the current file for the next time we need to delete
        self.last_saved_checkpoint = filename
        
        print(f"\n*** New High Score! Checkpoint saved: {filename} ***")
        

    def load_checkpoint(self, filepath):

        # 1. Safety check
        if not os.path.exists(filepath):
            print(f"Error: Checkpoint file '{filepath}' not found.")
            return

        # 2. Open the file and extract the dictionary
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        loaded_weights = checkpoint.get("layer_weights", {})

        all_layers = {}
        for name, layer in vars(self).items():
            all_layers[name] = layer
            if isinstance(layer, layers.InceptionModule):
                for sub_name, sub_layer in vars(layer).items():
                    all_layers[f"{name}.{sub_name}"] = sub_layer


        # 3. Dynamically map the saved matrices back into the model
        for layer_name, state in loaded_weights.items():
            if layer_name in all_layers:
                layer = all_layers[layer_name]

                # Restore Convolution matrices & their momentum
                if 'kernel' in state and hasattr(layer, 'kernel'):
                    layer.kernel = np.array(state['kernel'])
                if 'v_kernel' in state and hasattr(layer, 'v_kernel'):
                    layer.v_kernel = np.array(state['v_kernel'])
                
                # Restore Fully Connected matrices & their momentum
                if 'weight' in state and hasattr(layer, 'weight'):
                    layer.weight = np.array(state['weight'])
                if 'v_weight' in state and hasattr(layer, 'v_weight'):
                    layer.v_weight = np.array(state['v_weight'])
                
                # Restore Biases & their momentum
                if 'bias' in state and hasattr(layer, 'bias'):
                    layer.bias = np.array(state['bias'])
                if 'v_bias' in state and hasattr(layer, 'v_bias'):
                    layer.v_bias = np.array(state['v_bias'])
            else:
                print(f"Warning: Layer '{layer_name}' found in checkpoint but not in current model architecture.")

        # 4. Print the success metric
        val_acc = checkpoint.get("val_acc", "Unknown")
        epoch = checkpoint.get("epoch", "Unknown")
        print(f"\n*** Successfully loaded weights from Epoch {epoch}. Previous Val Acc: {val_acc:.4f} ***\n")
