"""A lot copied from https://github.com/migalkin/StarE"""

import os
from typing import Literal
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .mlp import MLP
from .stare_conv import StarEConvLayer
from .utils import process_graph


class GNN(torch.nn.Module):
    """Graph Neural Network model. This model is used to compute the Q-values for the
    memory management and explore policies. This model has N layers of GCNConv or
    StarEConv layers and two MLPs for the memory management and explore policies,
    respectively.

    Attributes:
        entities: List of entities
        relations: List of relations
        gcn_layer_params: The parameters for the GCN layers
        gcn_type: The type of GCN layer
        mlp_params: The parameters for the MLPs
        rotational_for_relation: Whether to use rotational embeddings for relations
        pretrained_path: The path to the pretrained model
        qa_entities: The entities to consider for the QA policy.
        device: The device to use
        embedding_dim: The dimension of the embeddings
        entity_to_idx: The mapping from entities to indices
        relation_to_idx: The mapping from relations to indices
        entity_embeddings: The entity embeddings
        relation_embeddings: The relation embeddings
        relu_between_gcn_layers: Whether to apply ReLU activation between GCN layers
        dropout_between_gcn_layers: Whether to apply dropout between GCN layers
        relu: The ReLU activation function
        drop: The dropout layer
        gcn_layers: The GCN layers
        mlp_mm: The MLP for memory management policy
        mlp_explore: The MLP for explore policy

    """

    def __init__(
        self,
        entities: list[str],
        relations: list[str],
        qa_entities: list[str],
        gcn_layer_params: dict = {
            "type": "StarE",
            "embedding_dim": 8,
            "num_layers": 2,
            "gcn_drop": 0.1,
            "triple_qual_weight": 0.8,
        },
        relu_between_gcn_layers: bool = True,
        dropout_between_gcn_layers: bool = True,
        mlp_params: dict = {"num_hidden_layers": 2, "dueling_dqn": True},
        rotational_for_relation: bool = True,
        pretrained_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize the GNN model.

        Args:
            entities: List of entities
            relations: List of relations
            qa_entities: The entities to consider for the QA policy.
            gcn_layer_params: The parameters for the GCN layers
            relu_between_gcn_layers: Whether to apply ReLU activation between GCN layers
            dropout_between_gcn_layers: Whether to apply dropout between GCN layers
            mlp_params: The parameters for the MLPs
            rotational_for_relation: Whether to use rotational embeddings for relations
            pretrained_path: The path to the pretrained model. This is only for loading
                GNN, mm, and explore models. The qa model is not loaded.
            device: The device to use. Default is "cpu".

        """
        super(GNN, self).__init__()
        self.entities = entities
        self.relations = relations
        self.qa_entities = qa_entities
        self.gcn_layer_params = gcn_layer_params
        self.gcn_type = gcn_layer_params["type"].lower()
        self.mlp_params = mlp_params
        self.rotational_for_relation = rotational_for_relation
        self.pretrained_path = pretrained_path
        self.device = device
        self.embedding_dim = gcn_layer_params["embedding_dim"]

        self.entity_to_idx = {entity: idx for idx, entity in enumerate(self.entities)}
        self.relation_to_idx = {
            relation: idx for idx, relation in enumerate(self.relations)
        }

        self.entity_embeddings = torch.nn.Parameter(
            torch.Tensor(len(self.entities), self.embedding_dim)
        ).to(self.device)
        torch.nn.init.xavier_normal_(self.entity_embeddings)

        if self.rotational_for_relation:
            # init relation embeddings with phase values
            phases = (
                2 * np.pi * torch.rand(len(self.relations), self.embedding_dim // 2)
            )
            self.relation_embeddings = torch.nn.Parameter(
                torch.cat(
                    [
                        torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
                        torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1),
                    ],
                    dim=0,
                )
            )
        else:
            self.relation_embeddings = torch.nn.Parameter(
                torch.Tensor(len(self.relations), self.embedding_dim)
            ).to(self.device)
            torch.nn.init.xavier_normal_(self.relation_embeddings)
        # self.relation_embeddings.data[0] = 0  # NOT SURE ABOUT THIS

        self.relu_between_gcn_layers = relu_between_gcn_layers
        self.dropout_between_gcn_layers = dropout_between_gcn_layers
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(self.gcn_layer_params["gcn_drop"])

        if "stare" in self.gcn_type:
            self.gcn_layers = torch.nn.ModuleList(
                [
                    StarEConvLayer(
                        in_channels=self.embedding_dim,
                        out_channels=self.embedding_dim,
                        num_rels=len(relations),
                        gcn_drop=self.gcn_layer_params["gcn_drop"],
                        triple_qual_weight=self.gcn_layer_params["triple_qual_weight"],
                    )
                    for _ in range(self.gcn_layer_params["num_layers"])
                ]
            ).to(self.device)

        elif "vanilla" in self.gcn_type:
            self.gcn_layers = torch.nn.ModuleList(
                [
                    GCNConv(
                        self.embedding_dim,
                        self.embedding_dim,
                        improved=False,
                        add_self_loops=False,
                        normalize=False,
                    )
                    for _ in range(self.gcn_layer_params["num_layers"])
                ]
            ).to(self.device)

            for layer in self.gcn_layers:
                if isinstance(layer, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        layer.bias.data.zero_()

        else:
            raise ValueError(f"{self.gcn_type} is not a valid GNN type.")

        self.mlp_mm = MLP(
            n_actions=3,
            input_size=self.embedding_dim * 3,  # [head, relation, tail]
            hidden_size=self.embedding_dim,
            device=device,
            **mlp_params,
        )
        self.mlp_explore = MLP(
            n_actions=5,
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            device=device,
            **mlp_params,
        )
        self.mlp_qa = MLP(
            n_actions=1,
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            device=device,
            num_hidden_layers=mlp_params["num_hidden_layers"],
            dueling_dqn=False,
        )

        if self.pretrained_path is not None:
            self.load_pretrained_model()

    def load_pretrained_model(self):
        """Load the pretrained model for the GNN, mm, and explore. qa is not loaded.

        This means that we freeze all but qa parameters."""
        pt_path = glob(os.path.join(self.pretrained_path, "*.pt"))[0]

        # Load the pretrained model state dict
        pretrained_state_dict = torch.load(pt_path)

        # only load the pretrained model's state dict. This doesn't load qa params
        self.load_state_dict(pretrained_state_dict, strict=False)

        # Freeze the parameters of the pretrained model
        for param in self.parameters():
            param.requires_grad = False

        for param in self.mlp_qa.parameters():
            param.requires_grad = True

    def process_batch(self, data: np.ndarray) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[list[str]],
        list[list[str]],
    ]:
        r"""Process the data batch. This is done by "flattening" the batch. There is no
        batch index returned. It's because graph is not euclidean, so we can't just
        stack the graphs. Instead, we need to process the batch as a whole.

        Args:
            data: The input data as a batch. This is the same as what the `forward`
                method receives. We will make them in to a batched version of the
                entity embeddings, relation embeddings, edge index, edge type, and
                qualifiers. StarE needs all of them, while vanilla-GCN only needs the
                entity embeddings and edge index.

        Returns:

            entity_embeddings: The shape is [num_entities in batch * 2, embedding_dim]
                This allows dupliciate entities in the batch.
                *2 stands for the forward and backward embeddings.
            relation_embeddings: The shape is [num_relations in batch *2, embedding_dim]
                This allows dupliciate relations in the batch.
                *2 stands for the forward and backward embeddings.
            edge_idx: The shape is [2, num_quadruples in batch * 2]
                *2 stands for the forward and backward edges.
            edge_type: The shape is [num_quadruples in batch * 2]
                *2 stands for the forward and backward edges.
            quals: The shape is [3, number of qualifier key-value pairs in batch *2]
                *2 stands for the forward and backward quals.
            short_memory_idx: The shape is [number of short-term memories in batch]
                the idx indexes `edge_idx` and `edge_type`
            num_short_memories: The shape is batch_size
            agent_entity_idx: The shape is batch_size

            entities_used: The entities used in the batch. The shape is batch_size.
                Leaf list is a list of entities used in the sample.
            relations_used: The relations used in the batch. The shape is batch_size.
                Leaf list is a list of relations used in the sample.

        """
        entity_embeddings_batch = []
        relation_embeddings_batch = []
        edge_idx_batch = []
        edge_type_batch = []
        quals_batch = []

        entity_embeddings_batch_inv = []
        relation_embeddings_batch_inv = []
        edge_idx_inv_batch = []
        edge_type_inv_batch = []
        quals_inv_batch = []

        short_memory_idx_batch = []
        agent_entity_idx_batch = []

        entity_offset_batch = [0]
        relation_offset_batch = [0]
        edge_offset_batch = [0]

        entities_used = []
        relations_used = []

        for idx, sample in enumerate(data):
            entities_used_sample = []
            relations_used_sample = []

            (
                entities,
                relations,
                edge_idx,
                edge_type,
                quals,
                edge_idx_inv,
                edge_type_inv,
                quals_inv,
                short_memory_idx,
                agent_entity_idx,
            ) = process_graph(sample)

            for j, entity in enumerate(entities):
                entity_embeddings_batch.append(
                    self.entity_embeddings[self.entity_to_idx[entity]]
                )
                entities_used_sample.append(entity)
            entities_used.append(entities_used_sample)

            for j, relation in enumerate(relations):
                relation_embeddings_batch.append(
                    self.relation_embeddings[self.relation_to_idx[relation]]
                )
                relations_used_sample.append(relation)
            relations_used.append(relations_used_sample)

            edge_idx_batch.append(edge_idx)
            edge_type_batch.append(edge_type)
            quals_batch.append(quals)

            for entity in entities:
                entity_embeddings_batch_inv.append(
                    self.entity_embeddings[self.entity_to_idx[entity]]
                )
            for relation in relations:
                relation_embeddings_batch_inv.append(
                    self.relation_embeddings[self.relation_to_idx[relation]]
                )
            edge_idx_inv_batch.append(edge_idx_inv)
            edge_type_inv_batch.append(edge_type_inv)
            quals_inv_batch.append(quals_inv)

            short_memory_idx_batch.append(short_memory_idx)
            agent_entity_idx_batch.append(agent_entity_idx)

            if idx < len(data) - 1:
                entity_offset_batch.append(len(entities) + entity_offset_batch[-1])
                relation_offset_batch.append(len(relations) + relation_offset_batch[-1])
                edge_offset_batch.append(edge_idx.size(1) + edge_offset_batch[-1])

        entity_embeddings_batch = torch.stack(entity_embeddings_batch, dim=0)
        entity_embeddings_batch_inv = torch.stack(entity_embeddings_batch_inv, dim=0)
        entity_embeddings = torch.cat(
            [entity_embeddings_batch, entity_embeddings_batch_inv], dim=0
        )

        relation_embeddings_batch = torch.stack(relation_embeddings_batch, dim=0)
        relation_embeddings_batch_inv = torch.stack(
            relation_embeddings_batch_inv, dim=0
        )
        relation_embeddings = torch.cat(
            [relation_embeddings_batch, relation_embeddings_batch_inv], dim=0
        )

        edge_idx_batch = [a + b for a, b in zip(edge_idx_batch, entity_offset_batch)]
        edge_idx_batch = torch.cat(edge_idx_batch, dim=1)

        edge_idx_batch_inv = torch.cat(
            [
                a + b + entity_embeddings_batch.shape[0]
                for a, b in zip(edge_idx_inv_batch, entity_offset_batch)
            ],
            dim=1,
        )
        edge_idx = torch.cat([edge_idx_batch, edge_idx_batch_inv], dim=1)

        edge_type_batch = [
            a + b for a, b in zip(edge_type_batch, relation_offset_batch)
        ]
        edge_type_batch = torch.cat(edge_type_batch, dim=0)

        edge_type_batch_inv = torch.cat(
            [
                a + b + relation_embeddings_batch.shape[0]
                for a, b in zip(edge_type_inv_batch, relation_offset_batch)
            ],
            dim=0,
        )
        edge_type = torch.cat([edge_type_batch, edge_type_batch_inv], dim=0)

        quals_batch = torch.cat(
            [
                a + torch.tensor([c, b, d]).reshape(-1, 1)
                for a, b, c, d in zip(
                    quals_batch,
                    entity_offset_batch,
                    relation_offset_batch,
                    edge_offset_batch,
                )
            ],
            dim=1,
        )
        quals = quals_batch.repeat(1, 2)

        num_short_memories = torch.tensor(
            [len(short_memory_idx) for short_memory_idx in short_memory_idx_batch]
        )
        short_memory_idx = torch.cat(
            [a + b for a, b in zip(short_memory_idx_batch, edge_offset_batch)], dim=0
        )
        agent_entity_idx = torch.tensor(
            [a + b for a, b in zip(agent_entity_idx_batch, entity_offset_batch)]
        )

        return (
            entity_embeddings,
            relation_embeddings,
            edge_idx,
            edge_type,
            quals,
            short_memory_idx,
            num_short_memories,
            agent_entity_idx,
            entities_used,
            relations_used,
        )

    def forward(
        self,
        data: np.ndarray,
        policy_type: Literal["mm", "explore", "qa"],
        questions: list[list[list[str]]] = None,
    ) -> list[torch.Tensor]:
        """Forward pass of the GNN model.

        Args:
            data: The input data as a batch. This data should have a batch dimension.
                However, internally in this function, we will process the batch as a
                whole. This is because the graph is not euclidean, so we can't just
                stack the graphs. Instead, we need to process the batch as a whole.

            policy_type: The policy type to use. Choose from "mm", "explore", or "qa".
            questions: The questions to answer. This is only used when `policy_type` is
                "qa". An example question is ["sta_000", "atlocation", "?"].

        Returns:
            The Q-values, with a batch dimension. The number of elements in the list is
            the number of actions in the sample.

        """

        data = np.array(
            [
                list(
                    [
                        ["agent", "atlocation", "room_000", {"current_time": 0}],
                        ["room_000", "east", "room_001", {"current_time": 0}],
                        ["dep_001", "atlocation", "room_000", {"current_time": 0}],
                        ["room_000", "west", "wall", {"current_time": 0}],
                        ["dep_007", "atlocation", "room_000", {"current_time": 0}],
                        ["room_000", "north", "wall", {"current_time": 0}],
                        ["room_000", "south", "room_004", {"current_time": 0}],
                    ]
                ),
                list(
                    [
                        ["room_001", "south", "room_005", {"current_time": 1}],
                        ["agent", "atlocation", "room_001", {"current_time": 1}],
                        ["room_001", "west", "room_000", {"current_time": 1}],
                        ["room_001", "north", "wall", {"current_time": 1}],
                        ["room_001", "east", "wall", {"current_time": 1}],
                        ["dep_007", "atlocation", "room_000", {"timestamp": [0]}],
                        ["room_000", "north", "wall", {"strength": 1}],
                    ]
                ),
                list(
                    [
                        ["room_005", "east", "room_006", {"current_time": 2}],
                        ["agent", "atlocation", "room_005", {"current_time": 2}],
                        ["room_005", "north", "room_001", {"current_time": 2}],
                        ["room_005", "south", "wall", {"current_time": 2}],
                        ["room_005", "west", "room_004", {"current_time": 2}],
                        ["dep_007", "atlocation", "room_000", {"timestamp": [0]}],
                        ["room_000", "north", "wall", {"strength": 1}],
                        ["agent", "atlocation", "room_001", {"strength": 1}],
                        ["room_001", "west", "room_000", {"timestamp": [1]}],
                        ["room_001", "north", "wall", {"strength": 1}],
                        ["room_001", "east", "wall", {"timestamp": [1]}],
                    ]
                ),
                list(
                    [
                        ["agent", "atlocation", "room_006", {"current_time": 3}],
                        ["room_006", "north", "wall", {"current_time": 3}],
                        ["sta_004", "atlocation", "room_006", {"current_time": 3}],
                        ["room_006", "south", "room_010", {"current_time": 3}],
                        ["room_006", "west", "room_005", {"current_time": 3}],
                        ["room_006", "east", "room_007", {"current_time": 3}],
                        ["dep_007", "atlocation", "room_000", {"timestamp": [0]}],
                        ["room_000", "north", "wall", {"strength": 1}],
                        ["agent", "atlocation", "room_001", {"strength": 1}],
                        ["room_001", "west", "room_000", {"timestamp": [1]}],
                        ["room_001", "north", "wall", {"strength": 1}],
                        ["room_001", "east", "wall", {"timestamp": [1]}],
                        ["room_005", "east", "room_006", {"timestamp": [2]}],
                        ["room_005", "north", "room_001", {"strength": 1}],
                    ]
                ),
            ],
            dtype=object,
        )
        (
            entity_embeddings,
            relation_embeddings,
            edge_idx,
            edge_type,
            quals,
            short_memory_idx,
            num_short_memories,
            agent_entity_idx,
            entities_used,
            relations_used,
        ) = self.process_batch(data)

        import pdb; pdb.set_trace()

        for layer_ in self.gcn_layers:
            if "stare" in self.gcn_type:
                entity_embeddings, relation_embeddings = layer_(
                    entity_embeddings=entity_embeddings,
                    relation_embeddings=relation_embeddings,
                    edge_idx=edge_idx,
                    edge_type=edge_type,
                    quals=quals,
                )
            elif "vanilla" in self.gcn_type:
                entity_embeddings = layer_(entity_embeddings, edge_idx)
            else:
                raise ValueError(f"{self.gcn_type} is not a valid GNN type.")

            if self.dropout_between_gcn_layers:
                entity_embeddings = self.drop(entity_embeddings)
            if self.relu_between_gcn_layers:
                entity_embeddings = F.relu(entity_embeddings)

        if policy_type == "mm":
            assert num_short_memories.sum() == short_memory_idx.size(0)
            triple = []
            for idx in short_memory_idx:
                triple_ = torch.cat(
                    [
                        entity_embeddings[edge_idx[0, idx]],
                        relation_embeddings[edge_type[idx]],
                        entity_embeddings[edge_idx[1, idx]],
                    ],
                    dim=0,
                )
                triple.append(triple_)

            triple = torch.stack(triple, dim=0)

            q_mm_ = self.mlp_mm(triple)

            # restore the original batch dimension
            q_mm = [
                q_mm_[start : start + num]
                for start, num in zip(
                    num_short_memories.cumsum(0).roll(1), num_short_memories
                )
            ]
            q_mm[0] = q_mm_[: num_short_memories[0]]

            return q_mm

        elif policy_type == "explore":
            node = []
            for idx in agent_entity_idx:
                node_ = entity_embeddings[idx]
                node.append(node_)

            node = torch.stack(node, dim=0)

            q_explore = self.mlp_explore(node)

            # restore the original batch dimension
            q_explore = [row.unsqueeze(0) for row in list(q_explore.unbind(dim=0))]

            return q_explore

        elif policy_type == "qa":
            assert (
                len(questions) == len(data) == len(entities_used)
            ), "the batch size doesn't match."

            tensor_qa = []

            offset = 0
            for entities_, relations_, questions_ in zip(entities_used, relations_used, questions):
                entity_embeddings_ = entity_embeddings[offset : offset + len(entities_)]
                relation_embeddings_ = relation_embeddings[
                    offset : offset + len(relations_)
                ]

                str2embedding = {}
                for i, entity_str in enumerate(entities_):
                    if entity_str in self.qa_entities:
                        str2embedding[entity_str] = entity_embeddings_[i]

                offset += len(entities_)

            # Step 1: Calculate the cumulative offsets
            entity_offsets = [0]  # Initialize with 0 as the first offset
            relation_offsets = [0]

            for entities_ in entities_used:
                entity_offsets.append(entity_offsets[-1] + len(entities_))

            for relations_ in relations_used:
                relation_offsets.append(relation_offsets[-1] + len(relations_))

            # Step 2: Use precomputed offsets in the loop
            for idx, (entities_, relations_, questions_) in enumerate(zip(entities_used, relations_used, questions)):
                # Use the precomputed offsets
                entity_embeddings_ = entity_embeddings[entity_offsets[idx]:entity_offsets[idx + 1]]
                relation_embeddings_ = relation_embeddings[relation_offsets[idx]:relation_offsets[idx + 1]]

                str2embedding = {}
                for i, entity_str in enumerate(entities_):
                    if entity_str in self.qa_entities:
                        str2embedding[entity_str] = entity_embeddings_[i]


            #     pass

            # num_entities_per_sample = [len(list_) for list_ in entities_used]

            # for list_ in entities_used:
            #     for dict_ in list_:
            #         for entity, embedding in dict_.items():
            #             if entity in self.qa_entities:
            #                 entity_tensor.append(embedding)

        else:
            raise ValueError(f"{policy_type} is not a valid policy type.")
