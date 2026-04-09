import numpy as np
import pandas as pd
import random

from Model.model_config import DEVICE

import torch
from torch.utils.data import DataLoader

from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)

from Model.graph_dataset import TabularDataset, collate_graph_batch

from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple


def _move_graph_batch_to_device(graph_batch: dict) -> dict:
    node_x = {k: v.to(DEVICE) for k, v in graph_batch["node_x"].items()}
    edge_index = {k: v.to(DEVICE) for k, v in graph_batch["edge_index"].items()}
    node_batch = {k: v.to(DEVICE) for k, v in graph_batch["node_batch"].items()}
    return {
        "node_x": node_x,
        "edge_index": edge_index,
        "node_batch": node_batch,
        "num_graphs": graph_batch["num_graphs"],
        "pr_number": graph_batch["pr_number"].to(DEVICE),
    }


def set_seed(seed: int) -> None:
    """
    Function set_seed controls random seed for reproducability
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_data_loaders(
    X: pd.DataFrame,
    y: pd.DataFrame,
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, int]:

    dataset = TabularDataset(X, y)
    graph_mode = getattr(dataset, "graph_mode", False)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_graph_batch if graph_mode else None,
    )

    return dataloader, len(dataset)


@torch.no_grad()
def get_prediction_probs(model: torch.nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function get_prediction_probs computes the raw logits and compute & return the probabilities, truth labels as ndarrys .
    """
    # set model for inference mode: stops batch-norm and dropouts
    model.eval()
    probalities, y_source = [], []

    # process data in batches: xb, yb
    for xb, yb in loader:

        if isinstance(xb, dict):
            xb = _move_graph_batch_to_device(xb)
        else:
            xb = xb.to(DEVICE)

        yb = yb.to(DEVICE)

        # raw output from model
        logits = model(xb)

        # sigmoid to convert logits to prob
        prob = torch.sigmoid(logits).cpu().numpy()

        probalities.append(prob)
        y_source.append(yb.cpu().numpy())

    return np.concatenate(probalities, axis=0), np.concatenate(y_source, axis=0)


def calc_pos_class_weight(y: pd.DataFrame) -> torch.Tensor:
    """
    Function calc_pos_class_weight returns pos/neg class ratio.
    """

    n_labels = y.shape[1]
    pos_weights_list = []

    for i in range(n_labels):

        y_i = y.iloc[:, i]
        # unique classes in the given y
        unique_classes=np.unique(y_i)

        if len(unique_classes)==1:
            pos_weight=1
        else:
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=unique_classes,
                y=y_i
            )
            # class_weights --> [class_weight_for_class 0, class_weight_for_class 1]
            pos_weight = class_weights[1]

        pos_weights_list.append(pos_weight)

    return torch.tensor(pos_weights_list, dtype=torch.float32).to(DEVICE)


def calculate_evaluation_metrics(probs: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    """
    Calculate multilabel evaluation metrics: accuracy, F1 score, precision, and recall.
    """

    # labels from prediction probability values
    preds = (probs > threshold).astype(int)

    preds_tensor = torch.tensor(preds, dtype=torch.float32)
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)

    num_labels = y_true.shape[1]


    accuracy = MultilabelAccuracy(num_labels=num_labels)
    precision = MultilabelPrecision(num_labels=num_labels)
    recall = MultilabelRecall(num_labels=num_labels)
    f1_metric = MultilabelF1Score(num_labels=num_labels, average="micro")

    acc_val = accuracy(preds_tensor, y_true_tensor)
    prec_val = precision(preds_tensor, y_true_tensor)
    rec_val = recall(preds_tensor, y_true_tensor)
    f1_val = f1_metric(preds_tensor, y_true_tensor)

    return {
        'accuracy': acc_val.item(),
        'precision': prec_val.item(),
        'recall': rec_val.item(),
        'f1': f1_val.item(),
    }
