"""A ResNet-like MLP with dueling DQN support."""

import torch
from torch.nn.init import xavier_normal_


class MLP(torch.nn.Module):
    """ResNet-like Multi-layer perceptron with ReLU activation functions.

    Attributes:
        input_size: Input size of the linear layer.
        hidden_size: Hidden size of the linear layer.
        num_hidden_layers: Number of layers in the MLP.
        n_actions: Number of actions.
        device: "cpu" or "cuda".
        dueling_dqn: Whether to use dueling DQN.

    """

    def __init__(
        self,
        n_actions: int,
        input_size: int,
        hidden_size: int,
        device: str,
        num_hidden_layers: int = 1,
        dueling_dqn: bool = True,
    ) -> None:
        """Initialize the MLP.

        Args:
            n_actions: Number of actions.
            input_size: Input size of the linear layer.
            hidden_size: Hidden size of the linear layer.
            device: "cpu" or "cuda".
            num_hidden_layers: int, number of layers in the MLP.
            dueling_dqn: Whether to use dueling DQN.

        """
        super(MLP, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.n_actions = n_actions
        self.dueling_dqn = dueling_dqn

        # Define the layers for the advantage stream
        self.advantage_layers = torch.nn.ModuleList()
        self.advantage_layers.append(
            self._init_layer(
                torch.nn.Linear(self.input_size, self.hidden_size, device=self.device)
            )
        )
        for _ in range(self.num_hidden_layers - 1):
            self.advantage_layers.append(
                self._init_layer(
                    torch.nn.Linear(
                        self.hidden_size, self.hidden_size, device=self.device
                    )
                )
            )
        self.advantage_layers.append(
            self._init_layer(
                torch.nn.Linear(self.hidden_size, self.n_actions, device=self.device)
            )
        )
        self.advantage_skip = torch.nn.Linear(
            self.input_size, self.hidden_size, device=self.device
        )

        if self.dueling_dqn:
            # Define the layers for the value stream
            self.value_layers = torch.nn.ModuleList()
            self.value_layers.append(
                self._init_layer(
                    torch.nn.Linear(
                        self.input_size, self.hidden_size, device=self.device
                    )
                )
            )
            for _ in range(self.num_hidden_layers - 1):
                self.value_layers.append(
                    self._init_layer(
                        torch.nn.Linear(
                            self.hidden_size, self.hidden_size, device=self.device
                        )
                    )
                )
            self.value_layers.append(
                self._init_layer(
                    torch.nn.Linear(self.hidden_size, 1, device=self.device)
                )
            )
            self.value_skip = torch.nn.Linear(
                self.input_size, self.hidden_size, device=self.device
            )

    def _init_layer(self, layer):
        if isinstance(layer, torch.nn.Linear):
            xavier_normal_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.zero_()
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x: Input tensor. The shape is (batch_size, lstm_hidden_size).
        Returns:
            torch.Tensor: Output tensor. The shape is (batch_size, n_actions).

        """

        def forward_layers(layers, x, skip_layer):
            identity = skip_layer(x)
            for layer in layers[:-1]:
                out = torch.relu(layer(x))
                x = out + identity  # Residual connection
                identity = x  # Update identity for next block
            out = layers[-1](x)
            return out

        if self.dueling_dqn:
            value = forward_layers(self.value_layers, x, self.value_skip)
            advantage = forward_layers(self.advantage_layers, x, self.advantage_skip)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q = forward_layers(self.advantage_layers, x, self.advantage_skip)

        return q
