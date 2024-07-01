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

import petml

from tests.utils import run_multi_process

config_map = {
    "common": {
        "max_move": 20,
        "max_merge": 10,
        "network_mode": "petnet",
        "network_scheme": "socket",
        "parties": {
            "party_a": {
                "address": ["127.0.0.1:50011"]
            },
            "party_b": {
                "address": ["127.0.0.1:50012"]
            }
        }
    },
    "party_a": {
        "inputs": {
            "user_weights": "examples/data/karate_club_user_weight1.json",
            "local_nodes": "examples/data/karate_club_local_nodes1.json",
        },
        "outputs": {
            "cluster": "tmp/server/cluster.csv"
        }
    },
    "party_b": {
        "inputs": {
            "user_weights": "examples/data/karate_club_user_weight2.json",
            "local_nodes": "examples/data/karate_club_local_nodes2.json",
        },
        "outputs": {
            "cluster": "tmp/client/cluster.csv"
        }
    }
}


class TestLeidenTransform:

    def test_leiden_transform(self):

        def run_leiden(party, config_map):
            operator = petml.operators.graph.LeidenTransform(party)
            operator.run(config_map)

        run_multi_process(run_leiden, [("party_a", config_map), ("party_b", config_map)])
