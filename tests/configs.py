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

from petml.infra.network import ClusterDef

parties = {0: {'address': ['127.0.0.1:50011'],}, 1: {'address': ['127.0.0.1:50012'],}}
CLUSTER_DEF = ClusterDef(mode="petnet", scheme="socket", parties=parties)
