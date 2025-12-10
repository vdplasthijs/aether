import os
from abc import ABC, abstractmethod
from typing import final, Dict, Any

import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset

class BaseDataset(Dataset, ABC):
    def __init__(
            self,
            df_path: str,
            modalities: list[str],
            target: bool=True,
            numericals: bool=False,
            dataset_name:str='BaseDataset',
            random_state=42
    ) -> None:

        # read and shuffle df
        assert os.path.exists(df_path), f'{df_path} does not exist.'
        self.df: pd.DataFrame = pd.read_csv(df_path)
        self.df = shuffle(self.df, random_state=random_state)

        # Set attributes
        self.dataset_name: str = dataset_name + '_'+ '_'.join(modalities)

        self.modalities: list[str] = modalities
        self.target: bool = target
        self.numericals: bool = numericals

        # Set placeholders
        self.num_classes: int | None = None
        self.target_names: list[str] | None = None
        self.numerical_names: list[str] | None = None
        self.records: Dict[str] | None = None

    @final
    def __len__(self) -> int:
        return len(self.records)

    @abstractmethod
    def __getitem__(
            self,
            idx: int
    ) ->  Dict[str, Any]:
        pass