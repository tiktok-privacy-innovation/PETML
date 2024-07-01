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

from petace.duet import VM
from petace.setops.psi import PSI
from petace.setops import PSIScheme

from petml.infra.abc.network import NetworkABC


class CipherEngine:
    support_engine = ("mpc", "kkrt_psi", "ecdh_psi")

    def __init__(self, engine_type: str, party_id: int, net: NetworkABC):
        if engine_type not in self.support_engine:
            raise ValueError(f"engine_type must be in {self.support_engine}")

        if engine_type == "mpc":
            self._engine = VM(net.net, party_id)
        elif engine_type == "kkrt_psi":
            self._engine = PSI(net.net, party_id, PSIScheme.KKRT_PSI)
        elif engine_type == "ecdh_psi":
            self._engine = PSI(net.net, party_id, PSIScheme.ECDH_PSI)

    @property
    def engine(self):
        return self._engine
