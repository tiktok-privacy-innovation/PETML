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

from petml.operators.operator_base import OperatorBase
from petml.fl.preprocessing import PSI
from petml.infra.engine.cipher_engine import CipherEngine
from petml.infra.storage.tabular_storage import CsvStorage


class PSITransform(OperatorBase):

    def _run(self, net, configs: dict) -> bool:
        """
        Perform PSI transform on the data.

        Expects the following configmap:
        {
            "common": {
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
                "protocol": "kkrt"
            },
            "party_a": {
                "column_name" : "id"
                "inputs": {
                    "data": "/path/to/data.csv"
                }
                "outputs": {
                    "data": "/path/to/data.csv"
                }
            },
            "party_b": {
                "column_name" : "id"
                "inputs": {
                    "data": "/path/to/data.csv"
                }
                "outputs": {
                    "data": "/path/to/data.csv"
                }
            }
        }
        """
        column_name = configs["column_name"]
        protocol = configs.get("protocol", "kkrt")
        if protocol not in ("ecdh", "kkrt"):
            raise ValueError(f"Protocol {protocol} not supported")
        psi_engine = CipherEngine(f"{protocol}_psi", self.party_id, net)

        # init io
        data = CsvStorage.read(configs["inputs"]["data"])

        # construct model
        model = PSI(column_name)
        model.set_infra(psi_engine)
        res = model.transform(data)

        CsvStorage.write(res, configs["outputs"]["data"], index=False)
        return True
