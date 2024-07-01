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

from typing import Union

import numpy as np
from petace.securenumpy import SecureArray
import petace.securenumpy as snp
import petace.secureml as sml


class LogisticLoss:
    """
    class for calculate logistic loss function
    """

    def _sigmoid(self, y_pred: np.ndarray):
        """
        Implemented sigmoid equation

        Parameters
        ----------
        y_pred : array, shape = (n_samples, )
            Predicted labels


        Returns
        -------
        values : float
        """
        return 1.0 / (1.0 + np.exp(-y_pred) + 1e-16)

    def grad(self, y_pred: Union[SecureArray, np.ndarray], label: Union[SecureArray, np.ndarray]):
        """
        First order derivative of the logistic loss function

        Parameters
        ----------
        y_pred : array or SecureArray, shape = (n_samples, 1)
            Predicted labels

        label : array-like, shape = (n_samples,1)
            Ground truth (correct) labels

        Returns
        -------
        values : array or SecureArray, shape = (n_samples, 1)
        """
        if isinstance(y_pred, np.ndarray):
            y_pred = self._sigmoid(y_pred)
            return np.array(y_pred - label).reshape(-1, 1)

        y_pred = sml.sigmoid(y_pred)
        return snp.reshape(y_pred - label, (-1, 1))

    def hess(self, y_pred: Union[SecureArray, np.ndarray]):
        """
        Second order derivative of the logistic loss function

        Parameters
        ----------
        y_pred : array or SecureArray, shape = (n_samples, 1)
            Predicted labels

        Returns
        -------
        values : array or SecureArray, shape = (n_samples, 1)
        """
        if isinstance(y_pred, np.ndarray):
            y_pred = self._sigmoid(y_pred)
            return np.array([max(i * (1.0 - i), 1e-16) for i in y_pred]).reshape(-1, 1)

        y_pred = sml.sigmoid(y_pred)
        return snp.reshape(y_pred * (snp.ones(y_pred.shape) - y_pred), (-1, 1))


class SquareLoss:
    """
    class for calculate square loss function
    """

    def __init__(self):
        pass

    def grad(self, y_pred: Union[SecureArray, np.ndarray], label: Union[SecureArray, np.ndarray]):
        """
        First order derivative of the square loss function

        Parameters
        ----------
        y_pred : array or SecureArray, shape = (n_samples, 1)
            Predicted labels

        label : array-like, shape = (n_samples,1)
            Ground truth (correct) labels

        Returns
        -------
        values : array or SecureArray, shape = (n_samples, 1)
        """
        if isinstance(y_pred, np.ndarray):
            return np.reshape(y_pred - label, (-1, 1))

        return y_pred - label

    def hess(self, y_pred: Union[SecureArray, np.ndarray]):
        """
        Second order derivative of the square loss function

        Parameters
        ----------
        y_pred : array or SecureArray, shape = (n_samples, 1)
            Predicted labels

        Returns
        -------
        values : array or SecureArray, shape = (n_samples, 1)
        """
        if isinstance(y_pred, np.ndarray):
            return np.ones_like(y_pred).reshape(-1, 1)

        return snp.ones(y_pred.shape)
