# Initialize the dataset
data_dir = '/home/aviadp/work/project/resnet-liver/train_volume'
seg_dir = '/home/aviadp/work/project/resnet-liver/train_seg'
dataset = CTScanDataset(data_dir, seg_dir, train=True)

# Get a sample from the dataset
sample = dataset[0]  # get the first sample

# Print the shapes and some stats about the sample to check if everything is ok
print(f"Input shape: {sample['input'].shape}, Output shape: {sample['output'].shape}")
print(f"Input min: {sample['input'].min()}, max: {sample['input'].max()}")
print(f"Output unique values: {np.unique(sample['output'])}")
