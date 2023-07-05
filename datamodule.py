# datamodule.py
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from data import CTScanDataset

class CTDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, split_ratio=0.8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio

    def setup(self, stage=None):
        # Prepare the full dataset
        full_dataset = CTScanDataset(self.data_dir)

        train_size = int(self.split_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size

        # Split the dataset
        self.ct_scan_train, self.ct_scan_val = random_split(full_dataset, [train_size, val_size])

        if stage in ('test', None):
            self.ct_scan_test = CTScanDataset(self.data_dir, train=False)  # For testing, you can replace this with your actual test dataset

    def train_dataloader(self):
        return DataLoader(self.ct_scan_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ct_scan_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.ct_scan_test, batch_size=self.batch_size)
