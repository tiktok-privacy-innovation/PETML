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

from petml.fl.graph import Leiden
from petml.infra.engine.cipher_engine import CipherEngine
from petml.infra.storage.tabular_storage import CsvStorage
from petml.infra.storage.graph_storage import JsonStorage
from petml.operators.operator_base import OperatorBase


class LeidenTransform(OperatorBase):

    def _run(self, net, configs: dict) -> bool:
        """
        Leiden algorithm.

        Expects the following configmap:
        {
            "common": {
                "max_move": 20,
                "max_merge": 10,
                "network_mode": "petnet",
                "network_scheme": "socket",
                "parties": {
                    "party_a": {
                        "address": ["IP_ADDRESS:50011"]
                    },
                    "party_b": {
                        "address": ["IP_ADDRESS:50012"]
                    }
                }
            },
            "party_a": {
                "inputs": {
                    "user_weights": "/path/to/data.json",
                    "local_nodes": "/path/to/data.json",
                }
                "outputs": {
                    "cluster": "/path/to/data.csv"
                }
            },
            "party_b": {
                "inputs": {
                    "user_weights": "/path/to/data.json",
                    "local_nodes": "/path/to/data.json",
                }
                "outputs": {
                    "cluster": "/path/to/data.csv"
                }
            }
        }
        """
        mpc_engine = CipherEngine("mpc", self.party_id, net)

        max_move = configs.get("max_move", 20)
        max_merge = configs.get("max_move", 10)

        # init io
        user_weights = JsonStorage.read(configs["inputs"]["user_weights"])
        local_nodes = JsonStorage.read(configs["inputs"]["local_nodes"])

        # construct model
        model = Leiden(max_move, max_merge)
        model.set_infra(self.party_id, net, mpc_engine)
        res = model.transform(user_weights, local_nodes)

        CsvStorage.write(res, configs["outputs"]["cluster"], index=False)
        return True
