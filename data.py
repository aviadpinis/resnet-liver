# data.py
import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import re

class CTScanDataset(Dataset):
    def __init__(self, data_dir, label_dir, train=True, split_ratio=0.8, intensity_range=(-200, 200), new_resolution=(224,224)):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.intensity_range = intensity_range
        self.new_resolution = new_resolution

        # Load all file names and sort them by extracting the numbers from the filenames
        self.data_samples = sorted([file for file in os.listdir(data_dir) if file.endswith(".nii")], 
                                    key=lambda name: int(re.search(r'\d+', name).group()))
        self.label_samples = sorted([file for file in os.listdir(label_dir) if file.endswith(".nii")],
                                    key=lambda name: int(re.search(r'\d+', name).group()))

	print(len(self.data_samples))
	print(len(self.label_samples))
        # Check if data_samples and seg_samples match
        assert self.data_samples == self.label_samples, "Data samples and segmentation samples do not match."

        # Split data
        train_data, test_data = train_test_split(self.data_samples, test_size=1-split_ratio, random_state=42)
        train_labels, test_labels = train_test_split(self.label_samples, test_size=1-split_ratio, random_state=42)
        self.data_samples = train_data if train else test_data
        self.label_samples = train_labels if train else test_labels

    def __len__(self):
        # Return the total number of data samples
        return len(self.data_samples)

    def __getitem__(self, idx):
        # Load your data here and return a sample
        # Remember to apply your preprocessing steps here or during data loading

        # Loading and preprocessing for the volume
        volume_filepath = os.path.join(self.data_dir, self.data_samples[idx])
        volume = nib.load(volume_filepath).get_fdata()

        # Clip intensity
        np.clip(volume, self.intensity_range[0], self.intensity_range[1], out=volume)

        # Normalize to [0, 1]
        min = volume.min()
        max = volume.max()
        volume = (volume - min) / (max - min)

        # Resample to a coarser resolution
        zoom_factors = [old_dim / new_dim for old_dim, new_dim in zip(volume.shape, self.new_resolution)]
        volume = zoom(volume, zoom_factors, order=1)  # using bilinear interpolation

        # Loading and preprocessing for the label
        label_filepath = os.path.join(self.label_dir, self.label_samples[idx])
        label = nib.load(label_filepath).get_fdata()

        # Resample to the same resolution as the volume
        label = zoom(label, zoom_factors, order=1)  # using bilinear interpolation

        # Convert all labels that are greater than 1 to 1
        label[label > 1] = 1

        return {"input": volume, "output": label}
