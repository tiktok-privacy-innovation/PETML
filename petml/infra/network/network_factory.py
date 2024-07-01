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

from typing import Union

from petml.infra.network.petnetNetwork.network import PetnetNetwork
from .cluster_def import ClusterDef

FEDMAP = {
    "petnet": PetnetNetwork,
}


class NetworkFactory:

    @staticmethod
    def get_network(party_id_or_party: Union[int, str], cluster_def: ClusterDef):
        if cluster_def is None:
            return None
        if cluster_def.mode not in FEDMAP:
            raise ValueError(f"{cluster_def.mode} not support yet.")
        federation = FEDMAP.get(cluster_def.mode)(party_id_or_party, cluster_def)
        return federation
