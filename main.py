import pytorch_lightning as pl
from datamodule import CTDataModule
from model import ResNetLiverModel
import wandb

def main():
    # Create a PyTorch Lightning trainer with the desired options
    wandb.login(key="bf4c93c7adbd15bce5b41147d566071d7b9581b8")
    wandb.init(project="resnet-liver")

    # Create a WandbLogger
    wandb_logger = pl.loggers.WandbLogger()
    # Define the data module and model
    data_dir = '/home/aviad/Desktop/src/learning/resnet-liver/train_volume'
    label_dir = '/home/aviad/Desktop/src/learning/resnet-liver/train_seg'

    # Parameters
    batch_size = 1
    num_workers = 4

    data_module = CTDataModule(data_dir=data_dir, label_dir=label_dir, batch_size=batch_size, num_workers=num_workers)
    model = ResNetLiverModel()

    # Initialize wandb
    wandb.watch(model)

    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, accelerator='cpu')  # Adjust as necessary
    # Fit the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test()


if __name__ == "__main__":
    main()
