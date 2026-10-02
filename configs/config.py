import torch

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# Dataloader Settings
dataset_path = "./datasets"
batch_size = 4
num_workers = 4
pin_memory = True
split_ratio = [0.8, 0.2]

# Model Settings
image_channel = 1
channel_expansion = 32
lo = 6
ld = 3
stage_num = 6

# Loss Function
sigma = 0.01

# Train
learning_rate = 5e-4
epochs = 400

# Infer
weight_file = "./weight/best.pth"
thre = 0.5
