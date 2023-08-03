import torch
from torchvision.models import resnet18
from torch.nn import functional as F
import pytorch_lightning as pl


class ResNetLiverModel(pl.LightningModule):

    def __init__(self, learning_rate=1e-3):
        super().__init__()

        # Initialize ResNet18
        self.model = resnet18(pretrained=False)  # not using pretrained weights
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Update the final layer to be a single output for binary classification (liver/not liver)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 1)

        self.learning_rate = learning_rate

    def forward(self, x):
        print("Input type:", type(x))  # Add this line to check the type of the input
        print("Input tensor shape:", x.shape)  # Add this line to check the shape of the input tensor

        batch_size, channels, height, width, depth = x.size()

        if x.dtype != torch.float32:
            x = x.float()
        # Reshape the tensor such that it is compatible with the conv2d layers.
        # We basically treat each slice as an independent example in the batch.
        x = x.view(-1, channels, height, width)

        print("x shape", x.shape)
        x = self.model(x)
        print("It's working")
        print("x shape", x.shape)
        return x

    def training_step(self, batch, batch_idx):
        print("train")
        x, y = batch['input'], batch['output']
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        print("val")
        x, y = batch['input'], batch['output']
        print("Input type:", type(x))  # Add this line to check the type of the input
        y_hat = self(x)
        y = y.view(-1)  # or target = target.reshape(-1)
        print("Target type:", type(y))  # Add this line to check the type of the input
        print("Target shape:", y.shape)  # Add this line to check the type of the input
        y_hat = y_hat.view(-1)  # or output = output.reshape(-1)
        print("y_hat shape:", y_hat.shape)  # Add this line to check the type of the input

        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
