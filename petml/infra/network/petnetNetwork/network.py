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
import struct
from typing import Union

from petace.network import NetParams, NetScheme, NetFactory

from petml.infra.abc import NetworkABC
from ..cluster_def import ClusterDef


class PetnetNetwork(NetworkABC):
    """Only supports two parties to synchronously send and receive data
    """

    def __init__(self, party_id_or_party: Union[int, str], cluster_def: ClusterDef):
        self.party_id = party_id_or_party
        self.cluster_def = cluster_def
        self.channel_cache = collections.deque([])
        self.net = None

        if len(cluster_def.parties) != 2:
            raise ValueError(f"PetnetNetwork only support 2-party, not {len(cluster_def['parties'])}")
        self.__init_net()

    def __init_net(self):
        net_params = NetParams()
        if self.cluster_def.scheme == "socket":
            net_scheme = NetScheme.SOCKET
            net_params.local_port = int(self.cluster_def.parties[self.party_id]["address"][0].split(":")[-1])
            net_params.remote_addr, remote_port = self.cluster_def.parties[1 - self.party_id]["address"][0].split(":")
            net_params.remote_port = int(remote_port)
        elif self.cluster_def.scheme == "agent":
            remote_party = None
            for party in self.cluster_def.parties:
                if party != self.party_id:
                    remote_party = party
                    break
            if remote_party is None:
                raise ValueError("remote party not found")
            net_scheme = NetScheme.AGENT
            net_params.shared_topic = self.cluster_def.topic
            net_params.remote_party = remote_party
            net_params.local_agent = self.cluster_def.parties[self.party_id]["address"][0]

        else:
            raise ValueError(f"mode {self.cluster_def.mode} not supported")

        self.net = NetFactory.get_instance().build(net_scheme, net_params)

    def clear(self):
        pass

    def get(self, name: str = None, party: int = None) -> bytes:
        ret = bytearray(4)
        self.net.recv_data(ret, 4)
        data_length = struct.unpack('!I', bytes(ret))[0]

        ret = bytearray(data_length)
        self.net.recv_data(ret, data_length)
        self.logger.debug(f"recv {name} successfully with length {data_length}")
        return bytes(ret)

    def remote(self, obj: bytes, name=None, party=None):
        if not isinstance(obj, bytes):
            raise TypeError(f"obj must be bytes, not {type(obj)}")
        data_length = len(obj)
        if data_length >= 2**32:
            raise ValueError(f"obj length must be less than 2**32, not {data_length}")

        length_prefix = struct.pack('!I', data_length)
        self.net.send_data(length_prefix, 4)
        self.logger.debug(f"send length prefix {length_prefix}")
        self.net.send_data(obj, len(obj))
        self.logger.debug(f"send {name} successfully")
