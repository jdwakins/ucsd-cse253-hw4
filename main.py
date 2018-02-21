# input data
with open('input.txt', 'r') as f:
    data = f.read()

# check for GPU
use_gpu = torch.cuda.is_available()

seq_len = 30
batch_size = 10
num_epochs = 2

train_model(data, seq_len, batch_size, num_epochs)
