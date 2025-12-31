import torch
import numpy as np
from tqdm import tqdm
import os
import boto3

from metrics import compute_metrics


def evaluate_mlp(
    model,
    data_loader,
    device,
    metrics_cfg: dict | None = None,
):
    model.eval()

    all_logits = []
    all_labels = []

    progress_bar = tqdm(
        data_loader,
        desc="Evaluating",
        leave=False,
        ncols=100,
    )

    with torch.no_grad():
        for batch in progress_bar:
            ids = batch["ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device) if "labels" in batch else None

            logits = model(ids, attention_mask)

            all_logits.append(logits.detach().cpu())
            if labels is not None: all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_labels, dim=0).numpy() if len(all_labels) > 0 else None

    y_proba = torch.softmax(logits, dim=1).numpy()
    y_pred = np.argmax(y_proba, axis=1)

    if y_true is not None:
        output = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    else:
        output = {
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist(),
        }

    if metrics_cfg is not None:
        metrics = compute_metrics(
            metrics_cfg=metrics_cfg["metrics"],
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            prefix=metrics_cfg.get("prefix", ""),
            log_to_mlflow=metrics_cfg.get("log_to_mlflow", False),
        )
        output["metrics"] = metrics

    return output


def get_s3_client():
    return boto3.client(
        service_name="s3",
        endpoint_url= os.getenv("AWS_ENDPOINT_URL"),
        region_name= os.getenv("AWS_DEFAULT_REGION", "ru-central1"),
        aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key= os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
