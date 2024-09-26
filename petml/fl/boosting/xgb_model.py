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

import hashlib
import json
import time

import numpy as np
import pandas as pd
from petace.securenumpy import SecureArray
import petace.securenumpy as snp
from sklearn.model_selection import train_test_split

from petml.fl.base import FlBase
from .decision_tree import MPCTree
from .loss import LogisticLoss, SquareLoss
from .metric import error, mean_absolute_error


class BaseXGB(FlBase):
    """Base class for XGBoost"""

    def __init__(self,
                 min_split_loss: float,
                 learning_rate: float,
                 base_score: float,
                 max_depth: int,
                 min_child_samples: int,
                 min_child_weight: float,
                 reg_alpha: float = 0.,
                 reg_lambda: float = 1.0,
                 label_name: str = 'label',
                 test_size: float = 0.):
        super().__init__()
        self.min_split_loss = min_split_loss
        self.learning_rate = learning_rate
        self.base_score = base_score
        self.max_depth = max_depth
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.label_name = label_name
        self.test_size = test_size

        self.party_id = None
        self.peer_party = None
        self._federation = None
        self._mpc_engine = None
        self.do_eval = False

    def set_infra(self, party_id, federation, mpc_engine):
        self.party_id = party_id
        self.peer_party = 1 - party_id
        self._federation = federation
        self._mpc_engine = mpc_engine
        snp.set_vm(self._mpc_engine.engine)

    @staticmethod
    def _set_hash_column(columns):
        """
        Transform string column to numerical hashed values

        Parameters
        ----------
        columns: array-like
            Columns in train dataset

        Returns
        -------
        hashed_column: array
            Numerical hashed values of columns
        """
        hash_set = set()
        hashed_column = []
        for col in columns:
            hash_value = hashlib.sha256(col.encode()).digest()
            hash_float_value = int.from_bytes(hash_value[:8], 'big') % (2**16)
            while hash_float_value in hash_set:
                hash_float_value = (hash_float_value + 1) % (2**16)
            hash_set.add(hash_float_value)
            hashed_column.append(hash_float_value)
        hashed_column = np.array(hashed_column)
        return hashed_column

    def prepare_train_data(self, data):
        """
        prepare_train_data

        Parameters
        ----------
        data: DataFrame
           Train data

        Returns
        -------
        train_x_cipher: SecureArray, shape=(n_samples, k)
            Secret sharing train data which concat data from two parties

        train_y: array
            local true label

        train_y_cipher: SecureArray, shape=(n_samples, )
            Secret sharing data which concat train label from two parties

        eval_x_cipher: SecureArray
            Secret sharing eval data which concat data from two parties

        eval_y: SecureArray, shape=(n_samples, )
            Secret sharing data which concat eval label from two parties

        train_column_cipher_party0: SecureArray, shape=(k, )
            Secret sharing data of columns

        hashed_column: array
           Numerical hashed values of columns
        """
        x = data[[col for col in data.columns if col != self.label_name]]
        y = data['label']

        columns = x.columns.tolist()
        hashed_column = self._set_hash_column(columns)

        if self.test_size > 0:
            self.do_eval = True
            train_x, eval_x, train_y, eval_y = train_test_split(x.values,
                                                                y.values,
                                                                test_size=self.test_size,
                                                                random_state=42)
        else:
            train_x = x.reset_index(drop='True').values
            train_y = y.values
            eval_x = None
            eval_y = None

        if self.party_id == 0:
            train_x_party0 = train_x
            train_x_party1 = None
            train_y_party0 = train_y.reshape(-1, 1)
            train_y_party1 = None
            eval_x_party0 = eval_x
            eval_x_party1 = None
            eval_y_party0 = eval_y
            eval_y_party1 = None
            train_column_party0 = hashed_column
            train_column_party1 = None

        else:
            train_x_party0 = None
            train_x_party1 = train_x
            train_y_party0 = None
            train_y_party1 = train_y.reshape(-1, 1)
            eval_x_party0 = None
            eval_x_party1 = eval_x
            eval_y_party0 = None
            eval_y_party1 = eval_y
            train_column_party0 = None
            train_column_party1 = hashed_column

        train_x_cipher_party0 = snp.array(train_x_party0, party=0)
        train_x_cipher_party1 = snp.array(train_x_party1, party=1)
        train_y_cipher_party0 = snp.array(train_y_party0, party=0)
        train_y_cipher_party1 = snp.array(train_y_party1, party=1)
        train_column_cipher_party0 = snp.array(train_column_party0, party=0)
        train_column_cipher_party1 = snp.array(train_column_party1, party=1)

        column_check = snp.where(train_column_cipher_party0 == train_column_cipher_party1,
                                 snp.ones(train_column_cipher_party0.shape),
                                 snp.zeros(train_column_cipher_party0.shape))
        column_check_lens = snp.sum(column_check)
        column_check_lens = column_check_lens.reveal_to(0)

        if column_check_lens.size > 0 and column_check_lens != len(hashed_column):
            raise ValueError("Columns in the two parties have different order or values")

        train_x_cipher = snp.vstack((train_x_cipher_party0, train_x_cipher_party1))
        train_y_cipher = snp.vstack((train_y_cipher_party0, train_y_cipher_party1))
        eval_x_cipher = None
        eval_y_cipher = None

        if self.do_eval:
            eval_x_cipher_party0 = snp.array(eval_x_party0, party=0)
            eval_x_cipher_party1 = snp.array(eval_x_party1, party=1)
            eval_y_cipher_party0 = snp.array(eval_y_party0, party=0)
            eval_y_cipher_party1 = snp.array(eval_y_party1, party=1)
            eval_x_cipher = snp.vstack((eval_x_cipher_party0, eval_x_cipher_party1))
            eval_y_cipher = snp.hstack((eval_y_cipher_party0, eval_y_cipher_party1))
            eval_y_cipher = snp.reshape(eval_y_cipher, (-1, 1))

        return train_x_cipher, train_y, train_y_cipher, eval_x_cipher, \
            eval_y_cipher, train_column_cipher_party0, hashed_column

    def prepare_predict_data(self, data):
        """

        Parameters
        ----------
        data: DataFrame
          Predict data

        Returns
        -------
        predict_x_cipher: SecureArray, shape=(n_samples, k)
            Secret sharing train data which concat data from two parties

        hashed_column: array
           Numerical hashed values of columns
        """
        x = data[[x for x in data.columns if x != self.label_name]]

        columns = x.columns.tolist()
        hashed_column = self._set_hash_column(columns)

        if self.party_id == 0:
            predict_x_party0 = x.values
            predict_x_party1 = None
            predict_column_party0 = hashed_column
            predict_column_party1 = None

        else:
            predict_x_party0 = None
            predict_x_party1 = x.values
            predict_column_party0 = None
            predict_column_party1 = hashed_column

        predict_x_cipher_party0 = snp.array(predict_x_party0, party=0)
        predict_x_cipher_party1 = snp.array(predict_x_party1, party=1)
        predict_column_cipher_party0 = snp.array(predict_column_party0, party=0)
        predict_column_cipher_party1 = snp.array(predict_column_party1, party=1)
        predict_x_cipher = snp.vstack((predict_x_cipher_party0, predict_x_cipher_party1))

        if predict_x_cipher.shape[0] > x.shape[0]:
            column_check = snp.where(predict_column_cipher_party0 == predict_column_cipher_party1,
                                     snp.ones(predict_column_cipher_party0.shape),
                                     snp.zeros(predict_column_cipher_party0.shape))
            column_check_lens = snp.sum(column_check)
            column_check_lens = column_check_lens.reveal_to(0)

            if column_check_lens.size > 0 and column_check_lens != len(hashed_column):
                raise ValueError("Columns in the two parties have different order or values")

        return predict_x_cipher, hashed_column

    def calc_gradient_local(self, loss_func, y):
        """
        Calculate first and second order derivative in the first round

        Parameters
        ----------
        loss_func: loss class
        y: array
            Ground true label

        Returns
        -------
        grads_cipher: SecureArray, shape=(n_samples, )
            Secret sharing data of first order derivative

        hess_cipher:  SecureArray, shape=(n_samples, )
           Secret sharing data of second order derivative
        """
        grads = loss_func.grad(np.array([self.base_score] * y.shape[0]), y)
        hess = loss_func.hess(np.array([self.base_score] * y.shape[0]))

        if self.party_id == 0:
            grads_party0 = grads
            grads_party1 = None
            hess_party0 = hess
            hess_party1 = None

        else:
            grads_party0 = None
            grads_party1 = grads
            hess_party0 = None
            hess_party1 = hess

        grads_cipher_party0 = snp.array(grads_party0, party=0)
        grads_cipher_party1 = snp.array(grads_party1, party=1)
        hess_cipher_party0 = snp.array(hess_party0, party=0)
        hess_cipher_party1 = snp.array(hess_party1, party=1)
        grads_cipher = snp.vstack((grads_cipher_party0, grads_cipher_party1))
        hess_cipher = snp.vstack((hess_cipher_party0, hess_cipher_party1))

        return grads_cipher, hess_cipher

    def transform_one_tree(self, train_x, train_y, train_y_cipher, eval_x_cipher, y_hat, eval_y_hat, column_cipher,
                           hashed_column, loss_func, num_t):
        """
        Build one decision tree

        Parameters
        ----------
        train_x: SecureArray, shape=(n_samples, k)
            Secret sharing train data which concat data from two parties
        train_y: array
            local true label
        train_y_cipher: SecureArray, shape=(n_samples, )
            Secret sharing data which concat train label from two parties
        eval_x_cipher: SecureArray
            Secret sharing eval data which concat data from two parties
        y_hat: SecureArray, shape=(n_samples, )
            predict train y of last epoch
        eval_y_hat: SecureArray
            predict eval y of last epoch
        column_cipher: SecureArray, shape=(k,)
            Secret sharing data of columns
        hashed_column: array
            Numerical hashed values of columns
        loss_func: loss class
        num_t: int
            current epoch


        Returns
        -------
        tree: MPCTree class
            Trained tree in current epoch
        y_hat: SecureArray
            Predict train y in current epoch
        eval_y_hat: SecureArray
            Predict eval y in current epoch
        """

        index_array = np.ones((train_x.shape[0], 1))

        if num_t == 0:
            grads, hess = self.calc_gradient_local(loss_func, train_y)
        else:
            grads = loss_func.grad(y_hat, train_y_cipher)
            hess = loss_func.hess(y_hat)

        tree = MPCTree(reg_alpha=self.reg_alpha,
                       reg_lambda=self.reg_lambda,
                       min_child_weight=self.min_child_weight,
                       columns=column_cipher,
                       min_child_samples=self.min_child_samples,
                       min_split_loss=self.min_split_loss,
                       max_depth=self.max_depth)

        tree.root = tree.build_tree(train_x, grads, hess, index_array, current_depth=0)

        y_preds = snp.zeros(shape=(train_x.shape[0], 1))
        for i in range(train_x.shape[0]):
            leaf_values = tree.predict_proba(tree.root, train_x[i, :], hashed_column, 0)
            y_preds[i] = sum(leaf_values)

        y_hat += self.learning_rate * y_preds

        if self.do_eval:
            eval_y_preds = snp.zeros(shape=(eval_x_cipher.shape[0], 1))
            for i in range(eval_x_cipher.shape[0]):
                leaf_values = tree.predict_proba(tree.root, eval_x_cipher[i, :], hashed_column, 0)
                eval_y_preds[i] = sum(leaf_values)
            eval_y_hat += self.learning_rate * eval_y_preds

        return tree, y_hat, eval_y_hat

    @staticmethod
    def export_share(share_value) -> list:
        return share_value.to_share().astype(np.int64).tolist()

    @staticmethod
    def load_share(load_data):
        return snp.fromshare(np.array(load_data).astype(np.int64), np.float64)

    def save_tree_from_ss_to_numpy(self, trees):
        for tree in trees:
            tree.columns = self.export_share(tree.columns)
            self._save_tree_from_ss_to_numpy(tree.root)

    def _save_tree_from_ss_to_numpy(self, tree_node):
        """Convert secure object to numerical value"""
        if tree_node.is_leaf:
            tree_node.leaf_weight = self.export_share(tree_node.leaf_weight)
            return

        tree_node.split_feat = self.export_share(tree_node.split_feat)
        tree_node.split_val = self.export_share(tree_node.split_val)
        self._save_tree_from_ss_to_numpy(tree_node.left_child)
        self._save_tree_from_ss_to_numpy(tree_node.right_child)

    def load_tree_from_numpy_to_ss(self, trees):
        for tree in trees:
            tree.columns = self.load_share(tree.columns)
            self._load_tree_from_numpy_to_ss(tree.root)

    def _load_tree_from_numpy_to_ss(self, tree_node):
        """Convert to secure object from numerical value"""
        if tree_node.is_leaf:
            tree_node.leaf_weight = self.load_share(tree_node.leaf_weight)
            return

        tree_node.split_feat = self.load_share(tree_node.split_feat)
        tree_node.split_val = self.load_share(tree_node.split_val)
        self._load_tree_from_numpy_to_ss(tree_node.left_child)
        self._load_tree_from_numpy_to_ss(tree_node.right_child)


class XGBoostClassifier(BaseXGB):
    """
    XGBoosting for classification

    Parameters
    ----------
    min_split_loss: float, default=1e-5
        The minimum number of gain to split an internal node:

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    base_score : float, default=0.5
        Initial value of y hat
        Values must be in the range `[0.0, 1)`.

    max_depth : int or None, default=3
        Maximum depth of a tree.
        If int, values must be in the range `[1, inf)`.

    reg_alpha : float, default=0.
        L1 regularization term on weights
        - values must be in the range `[0.0, inf)`.

    reg_lambda : float, default=1.0
        L2 regularization term on weights
        Values must be in the range `[0.0, inf)`.

    min_child_samples : int, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        Values must be in the range `[1, inf)`.

    min_child_weight : float, default=1.0
        Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results
        in a leaf node with the sum of instance weight less than min_child_weight, then the building
        process will give up further partitioning.
        Values must be in the range `(0, inf)`.

    test_size : float, default=0.0
        Size of eval dataset of input data
        Values must be in the range `(0, 1)`.

    eval_epochs : int, default=10
        Calculate the evaluation metric after every certain number of epochs.
        Values must be in the range `(1, inf)`.

    eval_threshold : float, default=0.5
        Regard the instances with eval prediction value larger than threshold
        as positive instances, and the others as negative instances

    objective : {'logitraw', 'logistic'}, default='logitraw'
        The loss function to be optimized. Only support logistic

    """

    def __init__(
        self,
        min_split_loss: float = 1e-5,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        base_score: float = 0.5,
        max_depth: int = 3,
        reg_alpha: float = 0.,
        reg_lambda: float = 1.0,
        min_child_samples: int = 1,
        min_child_weight: float = 1.,
        label_name: str = 'label',
        test_size: float = 0.,
        eval_epochs: int = 10,
        eval_threshold: float = 0.5,
        objective: str = 'logitraw',
    ):
        super().__init__(min_split_loss=min_split_loss,
                         learning_rate=learning_rate,
                         base_score=base_score,
                         max_depth=max_depth,
                         min_child_samples=min_child_samples,
                         min_child_weight=min_child_weight,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         label_name=label_name,
                         test_size=test_size)

        self.n_estimators = n_estimators
        self.eval_epochs = eval_epochs
        self.objective = objective
        self.eval_threshold = eval_threshold
        self.trees = []
        self.loss_func = LogisticLoss()

    def to_dict(self):
        result = vars(self).copy()
        del result['logger']
        result['trees'] = [tree.to_dict() for tree in self.trees]
        result['loss_func'] = self.loss_func.to_dict()
        return result

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        for k, v in data.items():
            if k == 'trees':
                setattr(obj, k, [MPCTree.from_dict(tree_dict) for tree_dict in v])
            elif k == 'loss_func':
                loss_map = {
                    'LogisticLoss': LogisticLoss,
                    'SquareLoss': SquareLoss,
                }
                loss_class = loss_map[v['class']]
                setattr(obj, k, loss_class.from_dict(v))
            else:
                setattr(obj, k, v)
        return obj

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model

        Parameters
        ----------
        data : DataFrame
            Train data
        """
        train_x_cipher, train_y, train_y_cipher, eval_x_cipher, eval_y_cipher, column_cipher, \
            hashed_column = self.prepare_train_data(data)

        if self.objective == 'logistic':
            self.base_score = 0.0
            self.loss_func = LogisticLoss()
        elif self.objective == 'logitraw':
            self.loss_func = LogisticLoss()
        else:
            raise ValueError("Only support logistic loss when apply XGBoostClassifier")

        y_hat = np.full((train_x_cipher.shape[0], 1), self.base_score)
        if self.do_eval:
            eval_y_hat = np.full((eval_x_cipher.shape[0], 1), self.base_score)
        else:
            eval_y_hat = None
        for num_t in range(self.n_estimators):
            start_time = time.time()
            self.logger.info(f"fitting tree {num_t + 1}...")
            tree, y_hat, eval_y_hat = self.transform_one_tree(train_x_cipher, train_y, train_y_cipher, eval_x_cipher,
                                                              y_hat, eval_y_hat, column_cipher, hashed_column,
                                                              self.loss_func, num_t)
            self.trees.append(tree)
            self.logger.info(f"tree {num_t + 1} fit done!, time: {time.time() - start_time}")

            if self.do_eval and (num_t + 1) % self.eval_epochs == 0:
                eval_predict_label = snp.where(eval_y_hat > self.eval_threshold, snp.ones(eval_y_hat.shape),
                                               snp.zeros(eval_y_hat.shape))
                errors = error(eval_predict_label, eval_y_cipher)
                plain_eval_error0 = errors.reveal_to(0)
                plain_eval_error1 = errors.reveal_to(1)
                if self.party_id == 0:
                    self.logger.info(f"eval error in {num_t + 1}: {plain_eval_error0}")
                else:
                    self.logger.info(f"eval error in {num_t + 1}: {plain_eval_error1}")

        self.logger.info("Finished training")

    def predict_proba(self, data: pd.DataFrame) -> SecureArray:
        """
        Predict probabilities.

        Parameters
        ----------
        data : DataFrame
            Predict data

        Returns
        -------
        proba : array of shape (n_samples, 1)
            Probabilities values
        """
        predict_x, hashed_column = self.prepare_predict_data(data)
        if len(predict_x) == 0:
            raise ValueError('Predict data in empty!')

        y_pred = [self.base_score] * predict_x.shape[0]

        for tree in self.trees:
            for i in range(predict_x.shape[0]):
                leaf_array = tree.predict_proba(tree.root, predict_x[i, :], hashed_column, 0)
                y_pred[i] = y_pred[i] + sum(leaf_array) * self.learning_rate

        return y_pred

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict label.

        Parameters
        ----------
        data : DataFrame
            Predict data

        Returns
        ----------
        predict_proba: Series
            Probabilities values of classifier
        """
        predict_proba_cipher = self.predict_proba(data)
        if self.party_id == 0:
            data_length_in_party0 = len(data)
        else:
            data_length_in_party0 = len(predict_proba_cipher) - len(data)

        predict_proba0 = [i.reveal_to(0) for i in predict_proba_cipher[:data_length_in_party0]]
        predict_proba1 = [i.reveal_to(1) for i in predict_proba_cipher[data_length_in_party0:]]
        if self.party_id == 0:
            predict_proba = predict_proba0
        else:
            predict_proba = predict_proba1
        predict_proba = np.array(predict_proba).reshape(-1,)

        return pd.Series(predict_proba)

    def save_model(self, model_path: str) -> None:
        """
        Save trained model.

        Parameters
        ----------
        model_path: string
           File path of the saved model
        """
        self.save_tree_from_ss_to_numpy(self.trees)
        try:
            self._federation = None
            self._mpc_engine = None
            json_str = json.dumps(self.to_dict())
            with open(model_path, 'w') as f:
                f.write(json_str)
            self.logger.info("Save model success")
        except json.JSONDecodeError as e:
            self.logger.error(f"Save model file. err={e}")

    def load_model(self, model_path: str) -> None:
        """
        Load model.

        Parameters
        ----------
        model_path: string
           File path of the saved model

        """
        try:
            with open(model_path, 'r') as f:
                load_obj = json.load(f)
            load_attributes = self.from_dict(load_obj)

            for attr in vars(self):
                if attr not in ['logger', '_federation', '_mpc_engine']:
                    setattr(self, attr, getattr(load_attributes, attr))

            self.load_tree_from_numpy_to_ss(self.trees)
            self.logger.info("Load model success")

        except json.JSONDecodeError as e:
            self.logger.error(f"Load model fail. err={e}")


class XGBoostRegressor(BaseXGB):
    """
    XGBoosting for regression

    Parameters
    ----------
    min_split_loss: float, default=1e-5
        The minimum number of gain to split an internal node:

    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
        There is a trade-off between learning_rate and n_estimators.
        Values must be in the range `[0.0, inf)`.

    n_estimators : int, default=100
        The number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
        Values must be in the range `[1, inf)`.

    base_score : float, default=0.5
        Initial value of y hat
        Values must be in the range `[0.0, 1)`.

    max_depth : int or None, default=3
        Maximum depth of the Tree.
        If int, values must be in the range `[1, inf)`.

    reg_alpha : float, default=0.
        L1 regularization term on weights
        - values must be in the range `[0.0, inf)`.

    reg_lambda : float, default=1.0
        L2 regularization term on weights
        Values must be in the range `[0.0, inf)`.

    min_child_samples : int, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        Values must be in the range `[1, inf)`.

    min_child_weight : float, default=1.0
        The minimum number of weights required to split an internal node:
        Values must be in the range `(0, inf)`.

    test_size : float, default=0.0
        Size of eval dataset of input data
        Values must be in the range `(0, 1)`.

    eval_epochs : int, default=10
        Calculate the evaluation metric after every certain number of epochs.
        Values must be in the range `(1, inf)`.

    objective : {'squarederror', 'logistic'}, default='squarederror'
        The loss function to be optimized. Only support logistic and squarederror

    """

    def __init__(
        self,
        min_split_loss: float = 1e-5,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        base_score: float = 0.5,
        max_depth: int = 3,
        reg_alpha: float = 0.,
        reg_lambda: float = 1.0,
        min_child_samples: int = 1,
        min_child_weight: float = 1,
        label_name: str = 'label',
        test_size: float = 0.,
        eval_epochs: int = 0,
        objective: str = 'squarederror',
    ):
        super().__init__(min_split_loss=min_split_loss,
                         learning_rate=learning_rate,
                         base_score=base_score,
                         max_depth=max_depth,
                         min_child_samples=min_child_samples,
                         min_child_weight=min_child_weight,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda,
                         label_name=label_name,
                         test_size=test_size)

        self.n_estimators = n_estimators
        self.eval_epochs = eval_epochs
        self.objective = objective
        self.trees = []
        self.loss_func = LogisticLoss()

    def to_dict(self):
        result = vars(self).copy()
        del result['logger']
        result['trees'] = [tree.to_dict() for tree in self.trees]
        result['loss_func'] = self.loss_func.to_dict()
        return result

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        for k, v in data.items():
            if k == 'trees':
                setattr(obj, k, [MPCTree.from_dict(tree_dict) for tree_dict in v])
            elif k == 'loss_func':
                loss_map = {'LogisticLoss': LogisticLoss, 'SquareLoss': SquareLoss}
                loss_class = loss_map[v['class']]
                setattr(obj, k, loss_class.from_dict(v))
            else:
                setattr(obj, k, v)
        return obj

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model

        Parameters
        ----------
        data : DataFrame
            Train data

        """
        train_x_cipher, train_y, train_y_cipher, eval_x_cipher, eval_y_cipher, column_cipher, \
            hashed_column = self.prepare_train_data(data)

        if self.objective == 'logistic':
            self.base_score = 0.0
            self.loss_func = LogisticLoss()
        elif self.objective == 'squarederror':
            self.loss_func = SquareLoss()
        else:
            raise ValueError("Only support logistic loss and squarederror when apply XGBoostRegressor")

        y_hat = np.full((train_x_cipher.shape[0], 1), self.base_score)
        if self.do_eval:
            eval_y_hat = np.full((eval_x_cipher.shape[0], 1), self.base_score)
        else:
            eval_y_hat = None
        for num_t in range(self.n_estimators):
            start_time = time.time()
            self.logger.info(f"fitting tree {num_t + 1}...")
            tree, y_hat, eval_y_hat = self.transform_one_tree(train_x_cipher, train_y, train_y_cipher, eval_x_cipher,
                                                              y_hat, eval_y_hat, column_cipher, hashed_column,
                                                              self.loss_func, num_t)
            self.trees.append(tree)
            self.logger.info(f"tree {num_t + 1} fit done!, time: {time.time() - start_time}")

            if self.do_eval and (num_t + 1) % self.eval_epochs == 0:
                errors = mean_absolute_error(eval_y_hat, eval_y_cipher)
                plain_eval_error0 = errors.reveal_to(0)
                plain_eval_error1 = errors.reveal_to(1)
                if self.party_id == 0:
                    self.logger.info(f"eval error in {num_t + 1}: {plain_eval_error0}")
                else:
                    self.logger.info(f"eval error in {num_t + 1}: {plain_eval_error1}")

        self.logger.info("Finished training")

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict label.

        Parameters
        ----------
        data : DataFrame
            Predict data

        Return
        ----------
        predict_values: Series
            Predict value of regression
        """
        predict_x, hashed_column = self.prepare_predict_data(data)
        if len(predict_x) == 0:
            raise ValueError('Predict data is empty!')

        y_pred = [self.base_score] * predict_x.shape[0]

        for tree in self.trees:
            for i in range(predict_x.shape[0]):
                leaf_array = tree.predict_proba(tree.root, predict_x[i, :], hashed_column, 0)
                y_pred[i] = y_pred[i] + sum(leaf_array) * self.learning_rate

        if self.party_id == 0:
            data_length_in_party0 = len(data)
        else:
            data_length_in_party0 = len(y_pred) - len(data)

        predict_values0 = [i.reveal_to(0) for i in y_pred[:data_length_in_party0]]
        predict_values1 = [i.reveal_to(1) for i in y_pred[data_length_in_party0:]]

        if self.party_id == 0:
            predict_values = predict_values0
        else:
            predict_values = predict_values1
        predict_values = np.array(predict_values).reshape(-1,)
        return pd.Series(predict_values)

    def save_model(self, model_path: str) -> None:
        """
        Save trained model.

        Parameters
        ----------
        model_path: string
           File path of the saved model
        """
        self.save_tree_from_ss_to_numpy(self.trees)
        try:
            self._federation = None
            self._mpc_engine = None
            json_str = json.dumps(self.to_dict())
            with open(model_path, 'w') as f:
                f.write(json_str)
            self.logger.info("Save model success")
        except json.JSONDecodeError as e:
            self.logger.error(f"Save model file. err={e}")

    def load_model(self, model_path: str) -> None:
        """
        Load model.

        Parameters
        ----------
        model_path: string
           File path of the saved model
        """
        try:
            with open(model_path, 'r') as f:
                load_obj = json.load(f)
            load_attributes = self.from_dict(load_obj)

            for attr in vars(self):
                if attr not in ['logger', '_federation', '_mpc_engine']:
                    setattr(self, attr, getattr(load_attributes, attr))

            self.load_tree_from_numpy_to_ss(self.trees)
            self.logger.info("Load model success")

        except json.JSONDecodeError as e:
            self.logger.error(f"Load model fail. err={e}")
