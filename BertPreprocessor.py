from transformers import AutoTokenizer
import torch
import numpy as np
import json

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import logging
import mlflow

from dataset import MainDataset
from utils import get_s3_client

logger = logging.getLogger(__name__)


def batch_tokenize(
        tokenizer,
        texts,
        batch_size,
        max_length):
    ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None
        )

        ids.extend(encoded['input_ids'])
        attention_mask.extend(encoded['attention_mask'])
        logger.info(f"Токенизировано: {min(i + batch_size, len(texts))} / {len(texts)}")
    ids = torch.tensor(ids)
    attention_mask = torch.tensor(attention_mask)

    return {'input_ids': ids, 'attention_mask': attention_mask}


@hydra.main(version_base=None, config_path="configs", config_name="BertPreprocessor")
def bert_proprocessor(
     cfg: DictConfig
):
    np.random.seed(cfg.union_random_seed)

    mlflow.set_tracking_uri(cfg.mlflow.uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    s3 = get_s3_client()
    s3.download_file(
        Bucket=cfg.s3.bucket,
        Key=cfg.s3.path_to_download,
        Filename=cfg.s3.path_to_save
    )

    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        dataset = instantiate(cfg.datasets)
        train_test_split = instantiate(cfg.train_test_split)

        train_dataset, test_dataset = train_test_split(dataset=dataset)

        tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model)
        batch_size = cfg.batch_size
        max_length = cfg.max_length

        encoded_train = batch_tokenize(tokenizer,
                                       train_dataset.dataframe[train_dataset.text_col].tolist(),
                                       batch_size,
                                       max_length)
        encoded_test = batch_tokenize(tokenizer,
                                       test_dataset.dataframe[test_dataset.text_col].tolist(),
                                       batch_size,
                                       max_length)

        torch.save(encoded_train, cfg.save_encoded_train)
        torch.save(encoded_test, cfg.save_encoded_test)

        labels = train_dataset.dataframe[train_dataset.labels]
        unique_labels = list(set(labels))

        labels_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        train_labels = [labels_to_id[label] for label in labels]
        test_labels = [labels_to_id[label] if label in labels_to_id else -1 for label in test_dataset.dataframe[test_dataset.labels]]

        with open(cfg.save_train_labels, 'w') as f:
            json.dump(train_labels, f)
        with open(cfg.save_test_labels, 'w') as f:
            json.dump(test_labels, f)

        s3.upload_file(
            Filename=cfg.save_encoded_test,
            Bucket=cfg.s3.bucket,
            Key=cfg.s3.path_to_encoded_test_data,
        )
        s3.upload_file(
            Filename=cfg.save_encoded_train,
            Bucket=cfg.s3.bucket,
            Key=cfg.s3.path_to_encoded_train_data,
        )


if __name__ == "__main__":
    bert_proprocessor()
