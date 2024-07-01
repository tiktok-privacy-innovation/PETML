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

from tests.process import Process


class FLTestException(Exception):

    def __init__(self, message) -> None:
        self.message = message

    def __str__(self):
        return f"test failed: {self.message}"


def run_multi_process(func, args_list):
    process_list = []
    for args in args_list:
        p = Process(target=func, args=args)
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()

    for p in process_list:
        if p.exception:
            error, _ = p.exception
            p.terminate()
            raise FLTestException(error)
