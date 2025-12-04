import os

import torch
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, df_path: str, modalities: list[str], target: bool=True, numericals: bool=False, dataset_name:str='BaseDataset'):
        for mod in modalities:
            assert mod in ['coords'], NotImplementedError(f'{mod} modality is not implemented.')
            # TODO: add other implemented modalities

        # read and shuffle df
        assert os.path.exists(df_path), f'{df_path} does not exist.'
        self.df = pd.read_csv(df_path)
        self.df = shuffle(self.df, random_state=42)

        self.modalities = modalities
        self.target = target
        self.numericals = numericals
        self.dataset_name = dataset_name + '_'+ '_'.join(modalities)

        # self.num_classes = None
        # TODO: target

    def __len__(self):
        return len(self.df)


class BaseButterflyDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(dataset_name='Butterflies', **kwargs)

        # Placeholder for filtered columns
        columns = ['name_loc']

        # If eo data is coords
        if 'coords' in self.modalities:
            self.modalities.remove('coords')
            self.modalities.extend(['lon', 'lat'])
            columns.extend(['lat', 'lon'])


        if self.target:
            self.target_names = [c for c in self.df.columns if 'target' in c]
            columns.extend(self.target_names)

        if self.numericals:
            self.numerical_names = [c for c in self.df.columns if c not in columns]
            columns.extend(self.numerical_names)

        self.records = self.df.loc[:, columns].to_dict('records')

    def __getitem__(self, idx):
        row = self.records[idx]

        formated_row = {'eo': {},
                        'target': {} if self.target else None,
                        'numericals': {} if self.numericals else None}

        for modality in self.modalities:
            if modality in ['lat', 'lon']:
                formated_row['eo'][modality] = torch.tensor(row[modality])
            elif modality == 's2':
                # TODO: tensor reading
                # formated_row['eo'][modality] =
                pass

        if self.target:
            formated_row['target'] = torch.tensor([row[k] for k in self.target_names], dtype=torch.float32)

        if self.numericals:
            formated_row['numericals'] = torch.tensor([row[k] for k in self.numerical_names], dtype= torch.float32)

        return formated_row