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

from petml.infra.storage.tabular_storage import CsvStorage, ParquetStorage


class TestCsv:

    def test_read_write(self, tmp_path):
        path = "examples/data/breast_hetero_mini_client.csv"
        df = CsvStorage.read(path)
        save_path = str(tmp_path / "res1.csv")
        CsvStorage.write(df, save_path)


class TestParquet:

    def test_read_write(self, tmp_path):
        path = "examples/data/breast_hetero_mini_client.parquet"
        df = ParquetStorage.read(path)
        save_path = str(tmp_path / "res1.parquet")
        ParquetStorage.write(df, save_path)
