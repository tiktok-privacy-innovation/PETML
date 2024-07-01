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

import numpy.testing as npt

from petml.infra.network import NetworkFactory
from tests.process import Process
from tests.configs import CLUSTER_DEF


class TestSocketFederation:

    def sync_message(self, party):
        message = b"hello world"
        F = NetworkFactory.get_network(party, CLUSTER_DEF)
        if party == 0:
            F.remote(message)
        else:
            res = F.get()
            npt.assert_equal(res, message)

        F.clear()

    def test_basic(self):
        p0 = Process(target=self.sync_message, args=(0,))
        p1 = Process(target=self.sync_message, args=(1,))
        p0.start()
        p1.start()
        p0.join()
        p1.join()
        for p in (p0, p1):
            if p.exception:
                error, _ = p.exception
                p.terminate()
                raise AssertionError(error)
