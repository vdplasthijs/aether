import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import re

class BaseButterflyDataset(Dataset):
    def __init__(self, df_path: str, mode="coords", target: bool=True):
        """
        Dataset returning desired features/outputs
        df: dataframe with file paths or coordinates
        target: indicator if target variable should be returned
        mode: what your __getitem__ should return
              options: 'coord', 's2_l1', 'variables'
        """
        df = pd.read_csv(df_path)
        df = shuffle(df, random_state=42)
        self.mode = mode
        self.target = target

        if self.mode in ['coords', 's2_l1']:
            df = df.loc[:, ['coords', 'name']]

            df['lat'] = df.apply(
                lambda row: re.search(r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)", row['coords']).group(1), axis=1)
            df['lon'] = df.apply(
                lambda row: re.search(r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)", row['coords']).group(2), axis=1)
            df.drop(['coords'], axis=1, inplace=True)

        elif self.mode in ['variables']:
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if not target:
            df = df.drop(columns=['name'])
        else:
            df = df.rename(columns={'name': 'class'})

        self.records = df.to_dict('records')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        return row

