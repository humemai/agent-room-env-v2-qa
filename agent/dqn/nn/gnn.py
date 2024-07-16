from typing import Literal
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class MLP(torch.nn.Module):
    """Multi-layer perceptron with ReLU activation functions.

    Attributes:
        hidden_size: Hidden size of the linear layer.
        num_hidden_layers: Number of layers in the MLP.
        n_actions: Number of actions.
        device: "cpu" or "cuda".
        dueling_dqn: Whether to use dueling DQN.

    """

    def __init__(
        self,
        n_actions: int,
        hidden_size: int,
        device: str,
        num_hidden_layers: int = 1,
        dueling_dqn: bool = True,
    ) -> None:
        """Initialize the MLP.

        Args:
            n_actions: Number of actions.
            hidden_size: Hidden size of the linear layer.
            device: "cpu" or "cuda".
            num_hidden_layers: int, number of layers in the MLP.
            dueling_dqn: Whether to use dueling DQN.

        """
        super(MLP, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.n_actions = n_actions
        self.dueling_dqn = dueling_dqn

        # Define the layers for the advantage stream
        advantage_layers = []
        for _ in range(self.num_hidden_layers):
            advantage_layers.append(
                torch.nn.Linear(self.hidden_size, self.hidden_size, device=self.device)
            )
            advantage_layers.append(torch.nn.ReLU())
        advantage_layers.append(
            torch.nn.Linear(self.hidden_size, self.n_actions, device=self.device)
        )
        self.advantage_layer = torch.nn.Sequential(*advantage_layers)

        if self.dueling_dqn:
            # Define the layers for the value stream
            value_layers = []
            for _ in range(self.num_hidden_layers):
                value_layers.append(
                    torch.nn.Linear(
                        self.hidden_size, self.hidden_size, device=self.device
                    )
                )
                value_layers.append(torch.nn.ReLU())
            value_layers.append(
                torch.nn.Linear(self.hidden_size, 1, device=self.device)
            )
            self.value_layer = torch.nn.Sequential(*value_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x: Input tensor. The shape is (batch_size, lstm_hidden_size).
        Returns:
            torch.Tensor: Output tensor. The shape is (batch_size, n_actions).

        """

        if self.dueling_dqn:
            value = self.value_layer(x)
            advantage = self.advantage_layer(x)
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q = self.advantage_layer(x)

        return q


# Define the GNN Model
class GNN(torch.nn.Module):
    """Graph Neural Network model.

    Attributes:
        entities: List of entities
        relations: List of relations
        embedding_dim: The dimension of the embeddings. This will be the size of the
            input node embeddings.
        num_layers_GNN: The number of layers in the GNN.
        num_hidden_layers_MLP: The number of layers in the MLP.
        device: The device to use.
        entity_embeddings: The entity embeddings.
        relation_embeddings: The relation embeddings.
        convs: The graph convolutional layers.
        entity_to_idx: The entity to index mapping.
        relation_to_idx: The relation to index mapping.
        mlp_mm: The MLP for the memory management policy.
        mlp_explore: The MLP for the explore policy.


    """

    def __init__(
        self,
        entities: list[str],
        relations: list[str],
        embedding_dim: int = 8,
        num_layers_GNN: int = 2,
        num_hidden_layers_MLP: int = 2,
        dueling_dqn: bool = True,
        device: str = "cpu",
    ):
        """Initialize the GNN model.

        Args:
            entities: List of entities
            relations: List of relations
            embedding_dim: The dimension of the embeddings. This will be the size of the
                input node embeddings.
            num_layers_GNN: The number of layers in the GNN.
            num_hidden_layers_MLP: The number of layers in the MLP.
            dueling_dqn: Whether to use dueling DQN.
            device: The device to use. Default is "cpu".

        """
        super(GNN, self).__init__()
        self.entities = entities
        self.relations = relations
        self.embedding_dim = embedding_dim
        self.num_layers_GNN = num_layers_GNN
        self.num_hidden_layers_MLP = num_hidden_layers_MLP
        self.dueling_dqn = dueling_dqn
        self.device = device

        self.entity_embeddings = torch.nn.Embedding(len(entities), embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(len(relations), embedding_dim)

        self.gnn = torch.nn.ModuleList(
            [
                GCNConv(self.embedding_dim, self.embedding_dim, normalize=True)
                for _ in range(self.num_layers_GNN)
            ]
        ).to(self.device)

        self.entity_to_idx = {entity: idx for idx, entity in enumerate(self.entities)}
        self.relation_to_idx = {
            relation: idx for idx, relation in enumerate(self.relations)
        }

        self.mlp_mm = MLP(
            n_actions=3,
            hidden_size=embedding_dim,
            device=device,
            num_hidden_layers=num_hidden_layers_MLP,
            dueling_dqn=dueling_dqn,
        )
        self.mlp_explore = MLP(
            n_actions=5,
            hidden_size=embedding_dim,
            device=device,
            num_hidden_layers=num_hidden_layers_MLP,
            dueling_dqn=dueling_dqn,
        )

    def convert_sample_to_data(self, sample: list[list]) -> Data:
        """Convert a sample to a Data object.

        Args:
            sample: The sample to convert.

        Returns:
            Data: The Data object.

        """
        node_features = []
        edge_index = []
        short_triples = []
        node_map = {}
        current_node_idx = 0
        agent_node = []

        for quadruple in sample:
            head, relation, tail, qualifiers = quadruple

            if head not in node_map:
                node_map[head] = current_node_idx
                node_features.append(
                    self.entity_embeddings(torch.tensor(self.entity_to_idx[head]))
                )
                current_node_idx += 1

            if tail not in node_map:
                node_map[tail] = current_node_idx
                node_features.append(
                    self.entity_embeddings(torch.tensor(self.entity_to_idx[tail]))
                )
                current_node_idx += 1

            if "current_time" in qualifiers:
                short_triples.append(
                    {
                        "head_idx": node_map[head],
                        "relation_idx": self.relation_to_idx[relation],
                        "tail_idx": node_map[tail],
                    }
                )
            if head == "agent":
                agent_node.append(node_map[head])

            edge_index.append([node_map[head], node_map[tail]])

        x = torch.stack(node_features).to(self.device)
        edge_index = (
            torch.tensor(edge_index, dtype=torch.long, device=self.device)
            .t()
            .contiguous()
        )

        to_return = Data(
            x=x,
            edge_index=edge_index,
            short_triples=short_triples,
            agent_node=agent_node,
        )

        return to_return

    def forward(
        self, data: np.ndarray, policy_type: Literal["mm", "explore"]
    ) -> list[torch.Tensor]:
        """Forward pass of the GNN model.

        Args:
            data: The input data as a batch.
            policy_type: The policy type to use.

        Returns:
            The Q-values. The number of elements in the list is equal to the number of
            samples in the batch. Each element is a tensor of Q-values for the actions
            in the sample. The length of the tensor is equal to the number of actions
            in the sample.

        """
        batch = [self.convert_sample_to_data(sample) for sample in data]

        loader = DataLoader(batch, batch_size=len(batch), shuffle=False)
        batch = next(iter(loader))

        x, edge_index = batch.x, batch.edge_index

        for gnn_layer in self.gnn:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)

        if policy_type == "mm":
            triple = []
            short_lengths = [0] + [len(short) for short in batch.short_triples]
            short_lengths = np.cumsum(short_lengths)
            for i, j, short in zip(batch.ptr, batch.ptr[1:], batch.short_triples):
                x_ = x[i:j]
                for k in short:
                    triple_ = (
                        x_[k["head_idx"]]
                        + self.relation_embeddings(torch.tensor(k["relation_idx"]))
                        + x_[k["tail_idx"]]
                    )
                    triple.append(triple_)
            triple = torch.stack(triple)

            q_mm = self.mlp_mm(triple)
            q_mm = [q_mm[i:j] for i, j in zip(short_lengths[:-1], short_lengths[1:])]

            return q_mm

        elif policy_type == "explore":
            node = []
            node_lengths = [i for i in range(batch.ptr.shape[0])]

            for i, j, an in zip(batch.ptr, batch.ptr[1:], batch.agent_node):
                x_ = x[i:j]
                if an:  # if there is an agent node (list is not empty)
                    node.append(x_[an[0]])  # use the agent node
                else:  # if there is no agent node (list is empty)
                    node.append(torch.mean(x_, dim=0))  # use the average of all nodes

            node = torch.stack(node)

            q_explore = self.mlp_explore(node)
            q_explore = [
                q_explore[i:j] for i, j in zip(node_lengths[:-1], node_lengths[1:])
            ]

            return q_explore

        else:
            raise ValueError("Invalid policy type.")
