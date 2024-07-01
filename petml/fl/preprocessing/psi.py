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

from petml.fl.base import FlBase


class PSI(FlBase):
    """
    Private Set Intersection (PSI) model.

    Parameters
    ----------------
    column_name : str
        The column to PSI.
    """

    def __init__(self, column_name: str):
        super().__init__()
        self.column_name = column_name

    def set_infra(self, psi_engine):
        self._psi_engine = psi_engine.engine

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        data: pd.DataFrame
            The data to PSI.

        Returns
        -------
        intersection: pd.DataFrame
            The intersection of the data.
        """
        if data[self.column_name].duplicated().any():
            raise ValueError(f"Duplicated values in column {self.column_name}")

        id_dtype = data[self.column_name].dtype
        data[self.column_name] = data[self.column_name].astype(str)  # PSI engine only support string type

        intersection = self._psi_engine.process(data[self.column_name].tolist(), obtain_result=True)
        self.logger.info(f"Origin data count {len(data)}, Intersection count {len(intersection)}")

        intersection_id = pd.DataFrame(intersection, columns=[self.column_name])
        intersection_df = pd.merge(intersection_id, data, how="left", on=self.column_name)
        intersection_df[self.column_name] = intersection_df[self.column_name].astype(id_dtype)
        return intersection_df
