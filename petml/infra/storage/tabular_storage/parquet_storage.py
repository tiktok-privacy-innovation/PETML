# Copyright 2024 TikTok Pte. Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from petml.infra.storage.tabular_storage.base import TabularStorage


class ParquetStorage(TabularStorage):

    @staticmethod
    def read(path: str, usecols: list = None) -> pd.DataFrame:
        """
        Load a parquet object from the file path, returning a DataFrame.

        Parameters
        ----------
        path : str
            path of csv file

        usecols: list, default=None
            If not None, only these columns will be read from the file.

        Returns
        -------
        pandas.DataFrame
        """
        return pd.read_parquet(path, columns=usecols)

    @staticmethod
    def write(obj: pd.DataFrame, path: str, index: bool = None, partition_cols: list = None) -> None:
        """
        Write a DataFrame to the binary parquet format.

        Parameters
        ----------
        obj : pandas.DataFrame
            obj to be written

        path : str
            path of csv file

        index : bool, default None
            If True, include the dataframe's index(es) in the file output. If False, they will not be written to the file.

        partition_cols : list, optional, default None
            Column names by which to partition the dataset. Columns are partitioned in the order they are given.
            Must be None if path is not a string.
        """
        obj.to_parquet(path, index=index, partition_cols=partition_cols)
