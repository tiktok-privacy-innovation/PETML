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


class Cache:
    """Cache for the information to compute modularity.
    """

    def __init__(self):
        self.vid = None
        self.cid = None
        self.w_cids = []

    def clear(self):
        self.vid = None
        self.cid = None
        self.w_cids = []


class Message:
    """Message for the communication between parties.
    """

    def __init__(self, state: str, data=None):
        self.state = state
        self.data = data

    def dumps(self):
        if self.data is None:
            return {"state": self.state}
        return {"state": self.state, "data": self.data}
