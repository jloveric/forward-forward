import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra
from high_order_layers_torch.layers import high_order_fc_layers


def MNIST_loaders(train_batch_size=5000, test_batch_size=1000, data_dir: str = "data"):

    transform = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x)),
        ]
    )

    train_loader = DataLoader(
        MNIST(data_dir, train=True, download=True, transform=transform),
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        MNIST(data_dir, train=False, download=True, transform=transform),
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):
    def __init__(self, dims, network_layer, cfg: DictConfig):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [network_layer(dims[d], dims[d + 1], cfg=cfg).cuda()]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print("training layer", i, "...")
            h_pos, h_neg = layer.train(h_pos, h_neg)


class TrainMixin:
    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(
                1
                + torch.exp(
                    torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
                )
            ).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class Layer(TrainMixin, nn.Linear):
    def __init__(
        self, in_features, out_features, cfg, bias=True, device=None, dtype=None
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.cfg = cfg
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))


class HighOrderLayer(TrainMixin, nn.Module):
    def __init__(self, in_features, out_features, cfg, device=None):
        super().__init__()

        self.cfg = cfg
        self.model = high_order_fc_layers(
            layer_type="discontinuous",
            n=cfg.n,
            in_features=in_features,
            out_features=out_features,
            segments=cfg.segments,
        )

        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.model(x_direction)


layer_dict = {"standard": Layer, "high_order": HighOrderLayer}


@hydra.main(config_path="../config", config_name="mnist")
def run(cfg: DictConfig):
    data_dir = f"{hydra.utils.get_original_cwd()}/data"

    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders(
        train_batch_size=cfg.train_batch_size,
        test_batch_size=cfg.test_batch_size,
        data_dir=data_dir,
    )

    layer_type = layer_dict[cfg.layer_type]
    net = Net(dims=cfg.layer_dim, network_layer=layer_type, cfg=cfg)
    x, y = next(iter(train_loader))
    x, y = x.cuda(), y.cuda()
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    net.train(x_pos, x_neg)

    print("train error:", 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print("test error:", 1.0 - net.predict(x_te).eq(y_te).float().mean().item())


if __name__ == "__main__":
    run()
