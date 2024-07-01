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

import logging


class LoggerFactory:

    @staticmethod
    def get_logger(name: str, level: int = logging.INFO, formater: str = None):
        """
        Return a logger

        Parameters
        ----------
        name : str
            name of logger.

        level : {logging.DEBUG, logging.INFO, logging.ERROR, logging.WARNING}, default is LOGGING.DEBUG
            log level.

        formater : str

        Returns
        -------
        logger
        """
        if not formater:
            formater = "[%(asctime)s] [%(name)s] [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d] [pid-thread:%(process)d-%(thread)d]: %(message)s"
        logger = logging.getLogger(name)
        logger.setLevel(level)

        formatter = logging.Formatter(formater)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        return logger
