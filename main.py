import pytorch_lightning as pl
from data import CTDataModule
from model import ResNetModel

def main():
    # Create a PyTorch Lightning trainer with the desired options
    trainer = pl.Trainer(max_epochs=10, gpus=1) # Adjust as necessary

    # Define the data module and model
    data_dir = "/path/to/your/data"  # Replace with the path to your data
    label_dir = "/path/to/your/labels"  # Replace with the path to your labels

    # Parameters
    batch_size = 8
    num_workers = 4

    data_module = CTDataModule(data_dir=data_dir, label_dir=label_dir, batch_size=batch_size, num_workers=num_workers)
    model = ResNetModel()

    # Fit the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test()

if __name__ == "__main__":
    main()
