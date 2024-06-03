from typing import Literal
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# Define the GNN Model
class GNN(torch.nn.Module):
    """Graph Neural Network model.

    Attributes:
        entities: List of entities
        relations: List of relations
        embedding_dim: The dimension of the embeddings. This will be the size of the
            input node embeddings.
        num_layers_GNN: The number of layers in the GNN.
        num_layers_MLP: The number of layers in the MLP.
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
        num_layers_MLP: int = 2,
        device: str = "cpu",
    ):
        """Initialize the GNN model.

        Args:
            entities: List of entities
            relations: List of relations
            embedding_dim: The dimension of the embeddings. This will be the size of the
                input node embeddings.
            num_layers_GNN: The number of layers in the GNN.
            num_layers_MLP: The number of layers in the MLP.
            device: The device to use. Default is "cpu".

        """
        super(GNN, self).__init__()
        self.entities = entities
        self.relations = relations
        self.embedding_dim = embedding_dim
        self.num_layers_GNN = num_layers_GNN
        self.num_layers_MLP = num_layers_MLP
        self.device = device

        self.entity_embeddings = torch.nn.Embedding(len(entities), embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(len(relations), embedding_dim)

        self.convs = torch.nn.ModuleList(
            [
                GCNConv(self.embedding_dim, self.embedding_dim, normalize=True)
                for _ in range(self.num_layers_GNN)
            ]
        ).to(self.device)

        self.entity_to_idx = {entity: idx for idx, entity in enumerate(self.entities)}
        self.relation_to_idx = {
            relation: idx for idx, relation in enumerate(self.relations)
        }

        self.mlp_mm = self.build_mlp(embedding_dim, 3).to(self.device)
        self.mlp_explore = self.build_mlp(embedding_dim, 5).to(self.device)

    def build_mlp(self, input_dim: int, output_dim: int) -> torch.nn.Sequential:
        """Builds an MLP with the given input and output dimensions.

        Note that input_dim and hidden_dim are the same.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.

        Returns:
            torch.nn.Sequential: The MLP model.

        """
        layers = []
        for _ in range(self.num_layers_MLP - 1):
            layers.append(torch.nn.Linear(input_dim, input_dim))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(input_dim, output_dim))

        return torch.nn.Sequential(*layers)

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
        agent_node = None

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
                agent_node = node_map[head]

            edge_index.append([node_map[head], node_map[tail]])

        if agent_node is None:
            raise ValueError("No 'agent' found in the sample")

        x = torch.stack(node_features).to(self.device)
        edge_index = (
            torch.tensor(edge_index, dtype=torch.long, device=self.device)
            .t()
            .contiguous()
        )

        return Data(
            x=x,
            edge_index=edge_index,
            short_triples=short_triples,
            agent_node=agent_node,
        )

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

        for conv in self.convs:
            x = conv(x, edge_index)
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
            for i, j, agent in zip(batch.ptr, batch.ptr[1:], batch.agent_node):
                x_ = x[i:j]
                node.append(x_[agent])

            node = torch.stack(node)

            q_explore = self.mlp_explore(node)
            q_explore = [
                q_explore[i:j] for i, j in zip(node_lengths[:-1], node_lengths[1:])
            ]

            return q_explore

        else:
            raise ValueError("Invalid policy type.")
