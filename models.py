"""
models.py
---------
All neural network architectures used in the ReLU experiment article.
Covers: MLPWithReLU, MLPLinear, variable-depth MLPs,
        2D classification models, and multi-activation variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------
# 1. MNIST Models (5-layer)
# ---------------------------------------------

class MLPWithReLU(nn.Module):
    """
    5-layer MLP with ReLU activations.
    Architecture: 784 -> 512 -> 256 -> 128 -> 64 -> 10
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class MLPLinear(nn.Module):
    """
    5-layer MLP WITHOUT activation functions.
    Mathematically equivalent to a single linear layer.
    Architecture: 784 -> 512 -> 256 -> 128 -> 64 -> 10
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)   # No activation — this is the key difference
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return self.fc5(x)


# ---------------------------------------------
# 2. Variable-Depth Linear Model (for depth sweep)
# ---------------------------------------------

class DeepLinearMLP(nn.Module):
    """
    Configurable-depth linear MLP for the depth vs accuracy sweep.
    All hidden layers have equal width (default 256).
    """
    def __init__(self, depth: int, hidden_size: int = 256,
                 input_size: int = 784, output_size: int = 10):
        super().__init__()
        assert depth >= 1, "depth must be at least 1"
        layers = []
        in_features = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x


class DeepReLUMLP(nn.Module):
    """
    Configurable-depth ReLU MLP — mirror of DeepLinearMLP with activations.
    Used to confirm that depth + ReLU continues to improve; depth alone doesn't.
    """
    def __init__(self, depth: int, hidden_size: int = 256,
                 input_size: int = 784, output_size: int = 10):
        super().__init__()
        assert depth >= 1, "depth must be at least 1"
        layers = []
        in_features = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:   # No activation on final layer
                x = F.relu(x)
        return x


# ---------------------------------------------
# 3. 2D Models (for two-moons decision boundary)
# ---------------------------------------------

class TwoD_ReLU(nn.Module):
    """3 hidden layers, ReLU, 2D input — for two-moons visualization."""
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.net(x)


class TwoD_Linear(nn.Module):
    """3 hidden layers, no activations, 2D input — for two-moons visualization."""
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------
# 4. Multi-activation variants (for activation comparison)
# ---------------------------------------------

ACTIVATION_MAP = {
    "relu":        nn.ReLU(),
    "leaky_relu":  nn.LeakyReLU(negative_slope=0.01),
    "sigmoid":     nn.Sigmoid(),
    "tanh":        nn.Tanh(),
    "gelu":        nn.GELU(),
    "none":        nn.Identity(),   # Linear baseline
}


class MLPWithActivation(nn.Module):
    """
    5-layer MLP that accepts any activation function as a string key.
    Used for the activation comparison experiment.
    """
    def __init__(self, activation: str = "relu"):
        super().__init__()
        assert activation in ACTIVATION_MAP, \
            f"Unknown activation '{activation}'. Choose from: {list(ACTIVATION_MAP.keys())}"
        self.activation_name = activation
        act = ACTIVATION_MAP[activation]

        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        # Each layer gets its own activation instance to avoid state sharing
        self.act1 = ACTIVATION_MAP[activation].__class__(**self._act_kwargs(activation))
        self.act2 = ACTIVATION_MAP[activation].__class__(**self._act_kwargs(activation))
        self.act3 = ACTIVATION_MAP[activation].__class__(**self._act_kwargs(activation))
        self.act4 = ACTIVATION_MAP[activation].__class__(**self._act_kwargs(activation))

    @staticmethod
    def _act_kwargs(activation: str) -> dict:
        if activation == "leaky_relu":
            return {"negative_slope": 0.01}
        return {}

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        return self.fc5(x)
