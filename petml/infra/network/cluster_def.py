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


class ClusterDef:
    """
    The cluster config of multi parties.

    cluster_def like
    {
        "mode": "petnet",
        "scheme": "socket"
        "parties": {
            0 : {"address": ["127.0.0.1:50011"]},
            1 : {"address": ["127.0.0.1:50012"]},
        }
        "topic": "petnet_topic"
    }
    """

    def __init__(self, mode: str = None, scheme: str = None, parties: dict = None, topic: str = None):
        self.mode = mode
        self.scheme = scheme
        self.parties = parties
        if scheme == "agent" and topic is None:
            raise ValueError("topic must be set when scheme is agent")
        self.topic = topic

    def __repr__(self):
        return f"""
            mode : {self.mode}
            scheme : {self.scheme}
            parties : {self.parties}
            tpopic : {self.topic}
        """
