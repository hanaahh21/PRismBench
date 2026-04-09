from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class GNNClassifier(nn.Module):
    """Simple heterogeneous GNN for PR-level multilabel classification."""

    def __init__(
        self,
        in_dim_by_type: Dict[str, int],
        rel_types: Tuple[Tuple[str, str, str], ...],
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        out_dim: int = 4,
    ):
        super().__init__()

        self.node_types = tuple(sorted(in_dim_by_type.keys()))
        self.rel_types = tuple(rel_types)
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, int(num_layers))

        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(in_dim_by_type[nt], hidden_dim)
            for nt in self.node_types
        })

        self.self_linears = nn.ModuleList([
            nn.ModuleDict({nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types})
            for _ in range(self.num_layers)
        ])

        self.rel_linears = nn.ModuleList([
            nn.ModuleDict({
                self._rel_key(rel): nn.Linear(hidden_dim, hidden_dim)
                for rel in self.rel_types
            })
            for _ in range(self.num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.head = nn.Linear(hidden_dim * len(self.node_types), out_dim)

    @staticmethod
    def _rel_key(rel: Tuple[str, str, str]) -> str:
        return f"{rel[0]}__{rel[1]}__{rel[2]}"

    @staticmethod
    def _mean_pool_by_graph(h: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
        out = h.new_zeros((num_graphs, h.shape[1]))
        out.index_add_(0, batch_idx, h)
        counts = h.new_zeros((num_graphs,))
        ones = h.new_ones((batch_idx.shape[0],))
        counts.index_add_(0, batch_idx, ones)
        return out / counts.clamp_min(1.0).unsqueeze(1)

    def forward(self, batch_graph: dict) -> torch.Tensor:
        node_x = batch_graph["node_x"]
        edge_index = batch_graph["edge_index"]
        node_batch = batch_graph["node_batch"]
        num_graphs = int(batch_graph["num_graphs"])

        h: Dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            if nt not in node_x:
                continue
            h[nt] = self.activation(self.input_proj[nt](node_x[nt]))

        for layer_idx in range(self.num_layers):
            updated: Dict[str, torch.Tensor] = {}
            for nt in h.keys():
                updated[nt] = self.self_linears[layer_idx][nt](h[nt])

            for rel in self.rel_types:
                src_t, _, dst_t = rel
                if rel not in edge_index or src_t not in h or dst_t not in h:
                    continue

                rel_ei = edge_index[rel]
                if rel_ei.numel() == 0:
                    continue

                src_idx = rel_ei[0]
                dst_idx = rel_ei[1]

                msg = self.rel_linears[layer_idx][self._rel_key(rel)](h[src_t][src_idx])
                agg = h[dst_t].new_zeros((h[dst_t].shape[0], self.hidden_dim))
                agg.index_add_(0, dst_idx, msg)

                deg = h[dst_t].new_zeros((h[dst_t].shape[0],))
                deg.index_add_(0, dst_idx, h[dst_t].new_ones((dst_idx.shape[0],)))
                agg = agg / deg.clamp_min(1.0).unsqueeze(1)

                updated[dst_t] = updated[dst_t] + agg

            for nt in list(updated.keys()):
                updated[nt] = self.dropout(self.activation(updated[nt]))
            h = updated

        pooled_parts = []
        for nt in self.node_types:
            if nt in h and nt in node_batch and h[nt].numel() > 0:
                pooled_parts.append(self._mean_pool_by_graph(h[nt], node_batch[nt], num_graphs))
            else:
                pooled_parts.append(torch.zeros((num_graphs, self.hidden_dim), device=self.head.weight.device))

        graph_repr = torch.cat(pooled_parts, dim=1)
        return self.head(graph_repr)
