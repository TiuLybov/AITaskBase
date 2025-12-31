import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import json
from tqdm import tqdm

import mlflow
import mlflow.pytorch

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from dataset import MainDataset, TaskDataset
from models import MLPTaskClassifier, CatboostMultiClf
from utils import evaluate_mlp

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train_model(
    cfg: DictConfig
):
    if cfg.model == "MLP":
        mlflow.set_tracking_uri(cfg.mlflow.uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        with mlflow.start_run(run_name=cfg.mlflow.run_name):
            encoded_train = torch.load(cfg.path_to_encoded_train)
            encoded_test = torch.load(cfg.path_to_encoded_test)

            with open(cfg.path_to_train_labels, 'r') as f:
                train_labels = json.load(f)
            with open(cfg.path_to_test_labels, 'r') as f:
                test_labels = json.load(f)

            train_dataset = TaskDataset(
                ids=encoded_train['input_ids'],
                attention_mask=encoded_train['attention_mask'],
                labels=train_labels
            )

            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=cfg.init_params.batch_size,
                shuffle=True,
                num_workers=cfg.init_params.num_workers,
                pin_memory=True,
                prefetch_factor=2
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MLPTaskClassifier(
                num_classes=len(set(train_labels)),
                bert_path=cfg.init_params.bert_model,
                linear_shape=cfg.fit_params.linear_shape,
                dropout=cfg.fit_params.dropout
            ).to(device)

            lr = cfg.fit_params.lr
            num_epoch = cfg.fit_params.num_epoch
            accumalation_steps = cfg.fit_params.accumalation_steps

            mlflow.log_params({
                "model": cfg.model,
                "lr": cfg.fit_params.lr,
                "batch_size": cfg.init_params.batch_size,
                "epochs": cfg.fit_params.num_epoch,
                "dropout": cfg.fit_params.dropout,
                "linear_shape": cfg.fit_params.linear_shape,
                "accumulation_steps": cfg.fit_params.accumalation_steps,
            })

            total_loss = 0
            correct = 0
            total = 0

            for epoch in range(num_epoch):
                model.train()

                scaler = GradScaler()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                progress_bar = tqdm(train_dataloader, desc="Train", leave=False)

                for step, batch in enumerate(progress_bar):
                    ids = batch['ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    with autocast():
                        logits = model(ids, attention_mask)
                        loss = nn.CrossEntropyLoss()(logits, labels) / accumalation_steps
                    scaler.scale(loss).backward()

                    if (step + 1) % accumalation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    total_loss += loss.item() * accumalation_steps
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    current_loss = total_loss / (step + 1)
                    current_acc = correct / total

                    if step % 10 == 0:
                        gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3
                        progress_bar.set_postfix({
                            'loss': f'{current_loss:.3f}',
                            'acc': f'{current_acc:.3f}',
                            'gpu': f'{gpu_mem:.3f}GB'
                        })

                train_loss = total_loss / len(train_dataloader)
                train_acc = correct / total
                logger.info(
                    f"[Epoch {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                )

                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                    },
                    step=epoch,
                )

                if (step + 1) % accumalation_steps != 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            logger.info("Evaluating on train data")
            metrics_train_cfg = OmegaConf.to_container(
                cfg.metrics.train_data,
                resolve=True,
            )
            evaluate_mlp(model,
                         train_dataloader,
                         device,
                         metrics_train_cfg)

            test_dataset = TaskDataset(
                ids=encoded_test['input_ids'],
                attention_mask=encoded_test['attention_mask'],
                labels=test_labels
            )

            test_dataloader = DataLoader(
                dataset=test_dataset,
                batch_size=cfg.init_params.batch_size,
                shuffle=True,
                num_workers=cfg.init_params.num_workers,
                pin_memory=True,
                prefetch_factor=2
            )

            logger.info("Evaluating on test data")
            metrics_test_cfg = OmegaConf.to_container(
                cfg.metrics.test_data,
                resolve=True,
            )
            evaluate_mlp(
                model,
                test_dataloader,
                device,
                metrics_test_cfg
            )

            torch.save(model, cfg.save_model_path)

    if cfg.model == "CatBoost":
        pass


if __name__ == "__main__":
    train_model()
