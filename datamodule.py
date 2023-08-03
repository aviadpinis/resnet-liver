# datamodule.py
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from data import CTScanDataset

class CTDataModule(LightningDataModule):
    def __init__(self, data_dir, label_dir, batch_size=32, split_ratio=0.8, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_workers = num_workers


    def setup(self, stage=None):
        print("Setting up the data module...")
        # Prepare the full dataset
        full_dataset = CTScanDataset(self.data_dir, self.label_dir)

        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size

        # Split the dataset
        self.ct_scan_train, self.ct_scan_val = random_split(full_dataset, [train_size, val_size])

        if stage in ('test', None):
            self.ct_scan_test = CTScanDataset(self.data_dir, self.label_dir, train=False)  # For testing, you can replace this with your actual test dataset

    def train_dataloader(self):
        return DataLoader(self.ct_scan_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ct_scan_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.ct_scan_test, batch_size=self.batch_size, num_workers=self.num_workers)
