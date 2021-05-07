""" CNN/DM dataset"""
import json
import re
import os
import pandas as pd
from os.path import join

from torch.utils.data import Dataset


class CnnDmDataset(Dataset):
    def __init__(self, split: str, path: str, cross_rev_bucket=None) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self.cross_rev_bucket = cross_rev_bucket
        if self.cross_rev_bucket:
            cross_rev_buc_file_path = join(path, 'cross_rev_bucket_{}.csv'.format(self.cross_rev_bucket))
            cross_rev_buc_df = pd.read_csv(cross_rev_buc_file_path)
            self.bucket_samples = cross_rev_buc_df['filename']
            self._n_data = len(self.bucket_samples)
        else:
            self._n_data = _count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        if self.cross_rev_bucket:
            reqd_file_name = self.bucket_samples[i]
            with open(join(self._data_path, reqd_file_name)) as f:
                js = json.loads(f.read())
        else:
            with open(join(self._data_path, '{}.json'.format(i))) as f:
                js = json.loads(f.read())
        return js


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data
