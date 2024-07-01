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


class CsvStorage(TabularStorage):

    @staticmethod
    def read(path: str,
             sep: str = ',',
             header: str = 'infer',
             usecols: list = None,
             dtype: str = None,
             nrows: int = None) -> pd.DataFrame:
        """
        read a csv as pd.DataFrame

        Parameters
        ----------
        path : str
            path of csv file

        sep : str, default ','
            Character or regex pattern to treat as the delimiter.

        header : int, Sequence of int, 'infer' or None, default 'infer'
            Row number(s) containing column labels and marking the start of the data (zero-indexed).

        usecols: Sequence of Hashable or Callable, optional
            Subset of columns to select, denoted either by column labels or column indices.

        dtype : dtype or dict of {Hashabledtype}, optional
            Data type(s) to apply to either the whole dataset or individual columns.

        nrows : int, optional
            Number of rows of file to read. Useful for reading pieces of large files.

        Returns
        -------
        pandas.DataFrame
        """
        return pd.read_csv(path, sep=sep, header=header, usecols=usecols, dtype=dtype, nrows=nrows)

    @staticmethod
    def write(obj: pd.DataFrame,
              path: str,
              columns: list = None,
              sep: str = ',',
              header: str = True,
              index: bool = False) -> None:
        """
        write a DataFrame to file_path

        Parameters
        ----------
        obj : pandas.DataFrame
            obj to be written

        path : str
            path of csv file

        sep : str, default ','
            String of length 1. Field delimiter for the output file.

        columns : sequence, optional
            Columns to write.

        header : bool or list of str, default True
            Write out the column names. If a list of strings is given it is assumed to be aliases for the column names.

        index : bool, default True
            Write row names (index).
        """
        obj.to_csv(path, columns=columns, sep=sep, header=header, index=index)
