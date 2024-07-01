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

from tests.configs import CLUSTER_DEF
from tests.utils import run_multi_process
from petml.fl.preprocessing import PSI
from petml.infra.network import NetworkFactory
from petml.infra.engine.cipher_engine import CipherEngine
from petml.infra.storage.tabular_storage import CsvStorage


class TestPSI:

    def test_psi(self):

        def run_psi(party_id, column_name, cluster_def, data, plain_res):
            net = NetworkFactory.get_network(party_id, cluster_def)
            psi_engine = CipherEngine("kkrt_psi", party_id, net)
            model = PSI(column_name)
            model.set_infra(psi_engine=psi_engine)
            res = model.transform(data)
            assert set(res[column_name]) == set(plain_res)

        column_name = "id"
        data_a = CsvStorage.read("examples/data/breast_hetero_mini_client.csv")
        data_b = CsvStorage.read("examples/data/breast_hetero_mini_server.csv")
        plain_res = set(data_a["id"]) & set(data_b["id"])

        configs = [(0, column_name, CLUSTER_DEF, data_a, plain_res), (1, column_name, CLUSTER_DEF, data_b, plain_res)]
        run_multi_process(run_psi, configs)
