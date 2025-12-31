from torch.utils.data import Dataset
import pandas as pd

import logging
from omegaconf import ListConfig

logger = logging.getLogger(__name__)


class TaskDataset(Dataset):
    def __init__(self,
                 ids,
                 attention_mask,
                 labels=None):
        self.ids = ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        item = {
            'ids': self.ids[idx],
            'attention_mask': self.attention_mask[idx]
        }
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


class MainDataset:
    def __init__(self,
                 dataframe,
                 text_col,
                 labels):
        if isinstance(dataframe, ListConfig):
            dataframes = []
            for path in dataframe:
                logger.info(f"Loading dataframe from {path}")
                df = pd.read_parquet(path, engine='pyarrow')
                dataframes.append(df)
            self.dataframe = pd.concat(dataframes)
        elif isinstance(dataframe, pd.DataFrame):
            self.dataframe = dataframe
        else:
            raise TypeError(f"dataframe must be a Path, str pd.Dataframe")

        self.text_col = text_col
        self.labels = labels

    def create_from_dataframe(self,
                              dataframe: pd.DataFrame):
        return MainDataset(
            dataframe=dataframe,
            text_col=self.text_col,
            labels=self.labels
        )
