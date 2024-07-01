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

import multiprocessing
import numpy as np
import numpy.testing as npt
import petace.securenumpy as snp

from petml.infra.engine.cipher_engine import CipherEngine
from petml.infra.network.network_factory import NetworkFactory
from tests.configs import CLUSTER_DEF


class TestCipherEngine:

    @staticmethod
    def create_petace_engine(que, data, party_id, cluster_def):
        net = NetworkFactory.get_network(party_id, cluster_def)
        petace_engine = CipherEngine("mpc", party_id, net).engine
        snp.set_vm(petace_engine)
        data0 = snp.array(data, 0)
        data1 = snp.array(data, 1)
        res_add = data0 + data1
        res_mul = data0 * data1
        res_comp = data0 > data1
        res_add_plain = res_add.reveal_to(0)
        res_mul_plian = res_mul.reveal_to(0)
        res_comp_plain = res_comp.reveal_to(0)
        if party_id == 0:
            que.put(res_add_plain)
            que.put(res_mul_plian)
            que.put(res_comp_plain)

    @staticmethod
    def create_psi_engine(que, data, party_id, cluster_def, protocol):
        net = NetworkFactory.get_network(party_id, cluster_def)
        psi_engine = CipherEngine(protocol, party_id, net).engine
        intersection = psi_engine.process(data, True)
        que.put(intersection)

    @staticmethod
    def run_process(target, args0, args1):
        p0 = multiprocessing.Process(target=target, args=args0)
        p1 = multiprocessing.Process(target=target, args=args1)
        p0.start()
        p1.start()
        p0.join()
        p1.join()

    def test_petace(self):
        que = multiprocessing.Queue()
        data0 = np.arange(10).astype(np.float64).reshape((1, -1))
        data1 = np.arange(5, 15).astype(np.float64).reshape((1, -1))
        cluster_def = CLUSTER_DEF
        args0 = (que, data0, 0, cluster_def)
        args1 = (que, data1, 1, cluster_def)
        self.run_process(self.create_petace_engine, args0, args1)
        result = que.get()
        npt.assert_equal(result, data0 + data1)
        result = que.get()
        npt.assert_almost_equal(result, data0 * data1, decimal=4)
        result = que.get()
        npt.assert_equal(result, data0 > data1)

    def test_psi(self):
        for protocol in ("ecdh_psi", "kkrt_psi"):
            que = multiprocessing.Queue()
            data0 = [f"{i}" for i in range(10)]
            data1 = [f"{i}" for i in range(5, 15)]
            cluster_def = CLUSTER_DEF
            args0 = (que, data0, 0, cluster_def, protocol)
            args1 = (que, data1, 1, cluster_def, protocol)
            self.run_process(self.create_psi_engine, args0, args1)
            result = que.get()
            assert (set(data0) & set(data1)) == set(result)
