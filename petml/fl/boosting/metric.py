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

from petace.securenumpy import SecureArray
import petace.securenumpy as snp


def error(y_pred: SecureArray, label: SecureArray):
    """
    Binary classification error rate. It is calculated as #(wrong cases)/#(all cases).

    Parameters
    ----------
    y_pred : SecureArray, shape = (n_samples, 1)
        Predicted labels, as returned by a classifier's
        predict method.

    label : SecureArray, shape = (n_samples, 1)
        Ground truth (correct) labels

    Returns
    -------
    error : float
    """
    wrong_cond = snp.where(y_pred != label, snp.ones(shape=y_pred.shape), snp.zeros(shape=y_pred.shape))
    return sum(wrong_cond) / len(y_pred)


def mean_absolute_error(y_pred: SecureArray, label: SecureArray):
    """
    Implementation of mean absolute error.

    Parameters
    ----------
    y_pred : SecureArray, shape = (n_samples, 1)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.

    label : SecureArray, shape = (n_samples, 1)
        Ground truth (correct) labels

    Returns
    -------
    error : float
    """

    absolut_error = snp.where(y_pred > label, y_pred - label, label - y_pred)
    return snp.sum(absolut_error) / len(y_pred)
