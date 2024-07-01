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

from copy import deepcopy
from typing import Dict
from abc import ABC, abstractmethod

from petml.infra.network import ClusterDef, NetworkFactory


class OperatorBase(ABC):

    def __init__(self, party: str, **kwargs):
        self.party = party
        self.party_id = None

    def _trans_parties(self, network_scheme: str, parties: dict):
        """
        Transform the parties to a int party_id.
        """
        parties_map = dict(zip(sorted(parties.keys()), range(len(parties))))
        self.party_id = parties_map[self.party]
        if network_scheme == "agent":
            new_parties = parties
        else:
            new_parties = {parties_map[party]: parties[party] for party in parties}
        return new_parties

    def run(self, configmap: Dict, config_manager: "ConfigManager" = None):
        # init infra
        parties = self._trans_parties(configmap["common"]["network_scheme"], configmap["common"]["parties"])
        cluster_def = ClusterDef(configmap["common"]["network_mode"], configmap["common"]["network_scheme"], parties,
                                 configmap["common"].get("shared_topic"))
        if configmap["common"]["network_scheme"] == "agent":
            net = NetworkFactory.get_network(self.party, cluster_def)
        elif configmap["common"]["network_scheme"] == "socket":
            net = NetworkFactory.get_network(self.party_id, cluster_def)
        else:
            raise ValueError(f"network scheme {configmap['common']['network_scheme']} not supported")

        configs = deepcopy(configmap["common"])
        configs.update(configmap[self.party])
        self._run(net, configs)
        return True

    @abstractmethod
    def _run(self, net, configs: dict):
        pass
