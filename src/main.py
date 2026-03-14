import model 
import cupy as np 
import pickle

def unpickle(file): 
    with open(file, "rb") as fo: 
        lookup = pickle.load(fo, encoding='bytes')
    return lookup

def load_cifar100_data(): 
    train = unpickle("./src/data/cifar-100-python/train")
    test = unpickle("./src/data/cifar-100-python/test")
    meta = unpickle("./src/data/cifar-100-python/meta")

    return train, test, meta 

def process_cifar_data(data):
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float16) / 255.0
    return data 

def process_cifar_labels(labels, num_classes=100):
    N = len(labels)
        
    # 2. Create a massive blank canvas of zeros: Shape (N, 100)
    one_hot = np.zeros((N, num_classes))
    
    # 3. The Vectorized Trick: Drop a '1.0' into the correct column for every row
    one_hot[np.arange(N), labels] = 1.0
    
    return one_hot

def train_val_data(train): 
    X_all = np.array(train[b"data"])
    Y_all = np.array(train[b"fine_labels"])
    N = X_all.shape[0] 

    X_all = process_cifar_data(X_all)
    

    # 1. Create a randomized array of indices (e.g., [412, 18, 49991, 2, ...])
    shuffled_indices = np.random.permutation(N)

    # 2. Apply the exact same shuffled map to both images and labels
    X_shuffled = X_all[shuffled_indices]
    y_shuffled = Y_all[shuffled_indices]

    # 3. Calculate the strict 90% cutoff index (45,000)
    split_idx = int(N * 0.90)

    # 4. Slice the arrays cleanly in half at the cutoff point
    X_train, X_val = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_val = y_shuffled[:split_idx], y_shuffled[split_idx:]

    return X_train, X_val, y_train, y_val 

def test_data(test): 
    X = np.array(test[b"data"])
    Y = np.array(test[b"fine_labels"])

    X_test  = process_cifar_data(X)
    

    return X_test, Y

if __name__ == "__main__":
    cnn = model.CNN() 

    train, test, meta = load_cifar100_data()
    X_train, X_val, y_train, y_val = train_val_data(train)
    X_test, y_test = test_data(test)
 
    epochs = 100
    batch_size = 32
    learning_rate = 1e-3

   
    cnn.load_checkpoint("checkpoints/roma_lr0.001_bs32_epoch41_acc0.5280.pkl")

    

    predictions = cnn.predict_batched(X_test, batch_size)
    test_accuracy = cnn.accuracy(predictions, y_test)

    print(f'Test accuracy {test_accuracy:.4f}')
    
