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
from petml.infra.abc import StorageABC


class TabularStorage(StorageABC):
    """
    TabularStorage provides functionality to efficiently work with tabular.

    The supported file formats currently are Parquet, Feather / Arrow IPC, CSV and ORC
    """

    @staticmethod
    def read(path, *args, **kwargs) -> pd.DataFrame:
        pass

    @staticmethod
    def write(obj: pd.DataFrame, path: str, *args, **kwargs) -> None:
        pass
