from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    Backward-compatible dataset module.

    - If X_df has 'pr_number', this loads heterogeneous PR graphs from
      graphs/apache_kafka/PR_<pr_number>.pkl (GNN mode).
    - Otherwise it behaves as the original tabular tensor dataset.
    """

    def __init__(self, X_df, y_df, graph_root: Path | str = Path("graphs/apache_kafka")):
        self.graph_mode = "pr_number" in X_df.columns
        self.graph_root = Path(graph_root)

        if self.graph_mode:
            self.pr_numbers = X_df["pr_number"].astype(int).tolist()
            self.y = torch.from_numpy(y_df.to_numpy(dtype=np.float32, copy=True))
            if len(self.pr_numbers) != len(self.y):
                raise ValueError("X and y row count mismatch in graph dataset mode.")
        else:
            X = X_df.to_numpy(dtype=np.float32, copy=True)
            y = y_df.to_numpy(dtype=np.float32, copy=True)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.pr_numbers) if self.graph_mode else self.X.shape[0]

    def _graph_path(self, pr_number: int) -> Path:
        return self.graph_root / f"PR_{pr_number}.pkl"

    @staticmethod
    def _to_2d(raw) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _load_graph(self, pr_number: int) -> tuple[dict, dict]:
        path = self._graph_path(pr_number)
        if not path.exists():
            raise FileNotFoundError(f"Missing graph file for PR {pr_number}: {path}")

        with path.open("rb") as f:
            g = pickle.load(f)

        node_x: Dict[str, torch.Tensor] = {}
        for ntype in ("pr", "commit", "file", "author"):
            if ntype not in g or "x" not in g[ntype]:
                continue
            node_x[ntype] = torch.from_numpy(self._to_2d(g[ntype]["x"]))

        edge_index: Dict[Tuple[str, str, str], torch.Tensor] = {}
        for key, val in g.items():
            if not isinstance(key, tuple) or len(key) != 3:
                continue
            if "edge_index" not in val:
                continue
            ei = np.asarray(val["edge_index"], dtype=np.int64)
            if ei.size == 0:
                continue
            edge_index[key] = torch.from_numpy(ei)

        if not node_x:
            raise ValueError(f"Graph has no node features: {path}")

        return node_x, edge_index

    def __getitem__(self, idx):
        if not self.graph_mode:
            return self.X[idx], self.y[idx]

        pr_number = self.pr_numbers[idx]
        node_x, edge_index = self._load_graph(pr_number)
        sample = {
            "pr_number": pr_number,
            "node_x": node_x,
            "edge_index": edge_index,
        }
        return sample, self.y[idx]


# Keep collate utility in same existing module to avoid creating new files.
def collate_graph_batch(batch):
    samples, labels = zip(*batch)
    batch_size = len(samples)

    all_node_types = sorted({nt for s in samples for nt in s["node_x"].keys()})
    all_rel_types = sorted({rel for s in samples for rel in s["edge_index"].keys()})

    node_x = {}
    node_batch = {}
    edge_index = {}

    node_offsets_per_graph: List[Dict[str, int]] = []
    running_offsets = {nt: 0 for nt in all_node_types}

    for graph_id, sample in enumerate(samples):
        offsets_this_graph = {}
        for nt in all_node_types:
            x = sample["node_x"].get(nt)
            if x is None:
                offsets_this_graph[nt] = running_offsets[nt]
                continue

            node_x.setdefault(nt, [])
            node_batch.setdefault(nt, [])
            node_x[nt].append(x)
            node_batch[nt].append(torch.full((x.shape[0],), graph_id, dtype=torch.long))
            offsets_this_graph[nt] = running_offsets[nt]
            running_offsets[nt] += x.shape[0]
        node_offsets_per_graph.append(offsets_this_graph)

    for nt in list(node_x.keys()):
        node_x[nt] = torch.cat(node_x[nt], dim=0)
        node_batch[nt] = torch.cat(node_batch[nt], dim=0)

    for graph_id, sample in enumerate(samples):
        offsets = node_offsets_per_graph[graph_id]
        for rel in all_rel_types:
            ei = sample["edge_index"].get(rel)
            if ei is None or ei.numel() == 0:
                continue
            src_t, _, dst_t = rel
            adjusted = ei.clone()
            adjusted[0] += offsets.get(src_t, 0)
            adjusted[1] += offsets.get(dst_t, 0)
            edge_index.setdefault(rel, [])
            edge_index[rel].append(adjusted)

    for rel in list(edge_index.keys()):
        edge_index[rel] = torch.cat(edge_index[rel], dim=1)

    y = torch.stack(labels, dim=0)
    batch_graph = {
        "node_x": node_x,
        "edge_index": edge_index,
        "node_batch": node_batch,
        "num_graphs": batch_size,
        "pr_number": torch.tensor([s["pr_number"] for s in samples], dtype=torch.long),
    }
    return batch_graph, y
