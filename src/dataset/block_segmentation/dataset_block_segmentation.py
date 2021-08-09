from pathlib import Path
from typing import List, Union
import pandas as pd

class BlockSegmentationDataset:

    def __init__(self,df: pd.DataFrame) -> None:
        self._lazy_val = None
        self._lazy_test = None
        self._lazy_train = None
        self.df = df

    def add_path_root(
        self,
        path_root: Path,
        columns: Union[str,List[str]] = ['lung','mask']
    ) -> None:
        for column in columns:
            new_column = []
            for row in self.df[column]:
                new_column.append(str(path_root) + '\\'+ row)
            self.df[column] = new_column
        return None

    @property
    def val(self) -> pd.DataFrame:
        if self._lazy_val is None:
            index_val = self.df['lung'].str.contains('val')
            self._lazy_val = self.df.loc[index_val]
        return self._lazy_val

    @property
    def train(self) -> pd.DataFrame:
        if self._lazy_train is None:
            index_train = self.df['lung'].str.contains('train')
            self._lazy_train = self.df.loc[index_train]
        return self._lazy_train

    @property
    def test(self) -> pd.DataFrame:
        if self._lazy_test is None:
            index_test = self.df['lung'].str.contains('test')
            self._lazy_test = self.df.loc[index_test]
        return self._lazy_test
