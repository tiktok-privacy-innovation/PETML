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

from __future__ import annotations
import json
import collections
from petml.infra.abc import StorageABC


def keystoint(x):
    # to support int key
    for k in x:
        if k.isdigit():
            return {int(k): v for k, v in x.items()}
        break
    return x


class JsonStorage(StorageABC):

    @staticmethod
    def read(path: str) -> collections.defaultdict:
        """
        read a json file to a dict

        Parameters
        ----------
        path : str
            path of json file

        Returns
        -------
        dict
        """
        with open(path, "rb") as f:
            data = json.load(f, object_hook=keystoint)
        if isinstance(data, list):
            return data
        G = collections.defaultdict(dict)
        G.update(data)
        return G

    @staticmethod
    def write(obj: list | dict, path: str) -> None:
        """
        write a dict to a json file

        Parameters
        ----------
        obj : list|dict
            dict to write

        path : str
            path of json file
        """
        if not isinstance(obj, (list, dict)):
            raise TypeError(f"GraphStorage only support write list or dict, not {type(obj)}")

        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
