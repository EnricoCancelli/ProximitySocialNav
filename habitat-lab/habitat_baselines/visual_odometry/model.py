import torch
import pytorch_lightning as pl
from torch import nn
from habitat_baselines.rl.ddppo.policy import resnet
import yaml

HEIGHT = 224
WIDTH = 224
ACTION_DIM = 2


class VOModel(nn.Module):
    @staticmethod
    def compute_encoder_output_size(encoder, compression, encoder_params):
        input_size = (1, encoder_params["in_channels"], HEIGHT,
                      WIDTH)

        encoder_input = torch.randn(*input_size)
        with torch.no_grad():
            output = encoder(encoder_input)
            output = compression(output)

        return output[-1].view(-1).size(0)

    def __init__(self, config):
        super().__init__()
        self.no_depth = config["no_depth"]
        self.no_rgb = config["no_rgb"]

        encoder_params = config["encoder"]
        self.encoder = getattr(resnet, encoder_params["type"])(
            encoder_params["in_channels"],
            encoder_params["base_planes"],
            encoder_params["ngroups"]
        )

        self.compression = nn.Sequential(
            nn.Conv2d(
                self.encoder.final_channels,
                encoder_params["num_compression_channels"],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, encoder_params["num_compression_channels"]),
            nn.ReLU(True),
        )

        self.action_embedding = nn.Linear(ACTION_DIM, config["action_embedding_size"])

        with torch.no_grad():
            fc_in = VOModel.compute_encoder_output_size(self.encoder, self.compression, encoder_params)

        fc_params = config["fc"]
        self.fc = nn.Sequential(
            nn.Linear(fc_in+config["action_embedding_size"], fc_params["hidden_size"]),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(fc_params["hidden_size"], fc_params["output_size"])
        )

    def forward(self, x, action):
        x = self.encoder(x)
        x = self.compression(x)
        x = torch.cat([nn.Flatten()(x), self.action_embedding(action)], dim=1)
        x = self.fc(x)
        return x

    @classmethod
    def from_config_path(cls, f_path):
        with open(f_path, "r") as f:
            config = yaml.load(f, yaml.loader.FullLoader)

        return cls(config["model"])


class VOPLModule(pl.LightningModule):
    def __init__(self, mod):
        self.mod = mod
        self.save_hyperparameters()

    def forward(self, x):
        return self.mod(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__ == "__main__":
    input = torch.zeros((1, 8, 224, 224))
    action = torch.zeros((1, 2))
    mod = VOModel.from_config_path("./config_base.yaml")
    out = mod(input, action)
    print(out.shape)
