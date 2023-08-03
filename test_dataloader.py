from datamodule import CTDataModule
import numpy as np

# Initialize the dataset
data_dir = '/home/aviad/Desktop/src/learning/resnet-liver/train_volume'
label_dir = '/home/aviad/Desktop/src/learning/resnet-liver/train_seg'
batch_size = 32
num_workers = 4
data_module = CTDataModule(data_dir=data_dir, label_dir=label_dir, batch_size=batch_size, num_workers=num_workers)
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# Inspect training data
for batch in train_loader:
    # Access input and target tensors
    inputs, targets = batch
    # Print or log the shape and type of the input tensor
    print("Input tensor shape:", inputs.shape)
    print("Input tensor type:", inputs.dtype)