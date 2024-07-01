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

import collections
import pandas as pd

from tests.configs import CLUSTER_DEF
from tests.utils import run_multi_process
from petml.fl.graph import Leiden
from petml.infra.storage.graph_storage import JsonStorage
from petml.infra.storage.tabular_storage import CsvStorage
from petml.infra.network.network_factory import NetworkFactory
from petml.infra.engine.cipher_engine import CipherEngine


class TestLeiden:

    def run_leiden(self, party_id, cluster_def, user_weights, local_nodes, output_path):
        net = NetworkFactory.get_network(party_id, cluster_def)
        mpc_engine = CipherEngine("mpc", party_id, net)
        model = Leiden(max_move=20, max_merge=10)
        model.set_infra(
            party_id=party_id,
            federation=net,
            mpc_engine=mpc_engine,
        )
        res = model.transform(user_weights, local_nodes)
        CsvStorage.write(res, output_path, index=False)

    def test_leiden(self):
        user_weights_a = JsonStorage.read("examples/data/karate_club_user_weight1.json")
        user_weights_b = JsonStorage.read("examples/data/karate_club_user_weight2.json")
        local_nodes_a = JsonStorage.read("examples/data/karate_club_local_nodes1.json")
        local_nodes_b = JsonStorage.read("examples/data/karate_club_local_nodes2.json")
        tmpfile_a = "tmp/cluster_a.csv"
        tmpfile_b = "tmp/cluster_b.csv"

        expected_modularity = 0.41
        configs = [(0, CLUSTER_DEF, user_weights_a, local_nodes_a, tmpfile_a),
                   (1, CLUSTER_DEF, user_weights_b, local_nodes_b, tmpfile_b)]

        run_multi_process(self.run_leiden, configs)
        g = load_graph("examples/data/karate_club_user_weight1.json", "examples/data/karate_club_user_weight2.json")
        partition = get_partition(tmpfile_a, tmpfile_b)
        assert modularity(g, partition) > expected_modularity


def get_partition(file1, file2):
    partitions1 = []
    for file in (file1, file2):
        with open(file, "r") as f:
            data = f.readlines()
            data = data[1:]
            for line in data:
                vertices = line.strip().split(',')
                vid = int(vertices[0])
                cid = vertices[1]
                partitions1.append([vid, cid])
    partitions1 = pd.DataFrame(partitions1, columns=["user_id", "label"])
    return partitions1


def load_graph(path1, path2):
    g1 = JsonStorage.read(path1)
    g2 = JsonStorage.read(path2)
    for n1, n1_adj in g2.items():
        for n2, w in n1_adj.items():
            if n1 not in g1[n2]:
                g1[n2][n1] = 0
            if n2 not in g1[n1]:
                g1[n1][n2] = 0
            g1[n1][n2] += w / 2
            g1[n2][n1] += w / 2
    return g1


def modularity(G, partition):
    community = collections.defaultdict(set)
    m = 0
    for u, v in partition.values:
        community[v].add(u)
    for u in G:
        for v in G[u]:
            m += G[u][v]
    m /= 2
    sum_in = 0
    sum_tot = 0
    for com in community.values():
        c_sum_in = 0
        c_sum_tot = 0
        for u in com:
            for v in G[u]:
                c_sum_tot += G[u][v]
                if v in com:
                    c_sum_in += G[u][v]
        sum_in += c_sum_in
        sum_tot += c_sum_tot**2
    return sum_in / (2 * m) - sum_tot / (4 * m**2)
