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

from abc import ABC, abstractmethod
from petml.infra.common.log_utils import LoggerFactory


class NetworkABC(ABC):
    """NetworkABC is the base class.
    """
    logger = LoggerFactory.get_logger(__name__)
    net = None

    @abstractmethod
    def get(self, name: str = None, party: int = None) -> bytes:
        """
        Get data from party.

        Parameters
        ----------
        party : int
            party id

        name : str
            name of the data

        Returns
        -------
        data : bytes
            data from party

        """

    @abstractmethod
    def remote(self, obj: bytes, name: str = None, party: int = None) -> None:
        """Send obj to party named name.

        Parameters
        ----------
        party : int
            party id

        obj : bytes
            data to send

        name : str
            name of the data
        """

    @abstractmethod
    def clear(self):
        pass
