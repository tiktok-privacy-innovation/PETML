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

import os
import re
import sys
from setuptools import find_packages, setup


def find_version(*filepath):
    # Extract version information from filepath
    with open(os.path.join('.', *filepath)) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        print("Unable to find version string.")
        sys.exit(-1)


def read_requirements():
    requirements = []
    _dependency_links = []
    with open('./requirements.txt') as file:
        requirements = file.read().splitlines()
    for r in requirements:
        if r.startswith("--extra-index-url"):
            requirements.remove(r)
            _dependency_links.append(r)
    print("Requirements: ", requirements)
    print("Dependency: ", _dependency_links)
    return requirements, _dependency_links


install_requires, dependency_links = read_requirements()

setup(
    name='petml',
    version=find_version("petml", "version.py"),
    license='Apache 2.0',
    description='privacy preserving machine learning',
    author='PrivacyGo-PETPlatform',
    author_email='privacygo-petplatform@tiktok.com',
    python_requires="==3.9",
    packages=find_packages(exclude=('examples', 'examples.*', 'tests', 'tests.*')),
    install_requires=install_requires,
    dependency_links=dependency_links,
)
