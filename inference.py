import torch
from torch.utils.data import DataLoader
import json

import mlflow
import mlflow.pytorch

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from dataset import MainDataset, TaskDataset
from utils import evaluate_mlp, get_s3_client

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="inference")
def inference(
    cfg: DictConfig
):
    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    s3 = get_s3_client()
    s3.download_file(
        Bucket=cfg.s3.bucket,
        Key=cfg.s3.path_to_download,
        Filename=cfg.path_to_encoded_data
    )

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        logger.info("Чтение данных")
        encoded_data = torch.load(cfg.path_to_encoded_data)

        dataset = TaskDataset(
            ids=encoded_data['input_ids'],
            attention_mask=encoded_data['attention_mask']
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg.init_params.batch_size,
            shuffle=True,
            num_workers=cfg.init_params.num_workers,
            pin_memory=True,
            prefetch_factor=2
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Загрузка модели")
        model = torch.load(cfg.path_to_model, weights_only=False)

        results = evaluate_mlp(
            model,
            dataloader,
            device
        )

        logger.info("Сохранение результатов")
        with open(cfg.save_scores_path, 'w') as f:
            json.dump(results, f)

        s3.upload_file(
            Filename=cfg.save_scores_path,
            Bucket=cfg.s3.bucket,
            Key=cfg.s3.path_to_upload,
        )


if __name__ == '__main__':
    inference()
