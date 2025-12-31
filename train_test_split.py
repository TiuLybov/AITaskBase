import logging

from dataset import MainDataset


logger = logging.getLogger(__name__)


def train_test_split(dataset: MainDataset,
                      percent_test: float = 0.25,
                      random_sample: int = 42,
                      ) -> tuple[MainDataset, MainDataset]:

    df = dataset.dataframe

    train_df = df.sample(n=int(df.shape[0] * (1 - percent_test)), random_state=random_sample)
    test_df = df.loc[~df.index.isin(train_df.index)]

    return (
        dataset.create_from_dataframe(train_df),
        dataset.create_from_dataframe(test_df)
    )
