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

import numpy as np
from petace.securenumpy import SecureArray
import petace.securenumpy as snp


class MPCTreeNode:
    """
    Class for secure decision tree node

    Attributes
    ----------
    is_leaf : Bool
        Flag to check the node is leaf or not

    leaf_weight : float
        Leaf weight value in secret share format of the current node

    split_feat : SecureArray
        Best split feature in secret share format in current node

    split_val : SecureArray
        Pplit value of the best split feature in secret share format in current node

    left_child : MPCTreeNode
        Record the information of the left node of the current node.

    right_child : MPCTreeNode
        Record the information of the right node of the current node.
    """

    def __init__(self,
                 is_leaf: bool = False,
                 leaf_weight: float = None,
                 split_feat: SecureArray = None,
                 split_val: SecureArray = None,
                 left_child: "MPCTreeNode" = None,
                 right_child: "MPCTreeNode" = None):
        self.is_leaf = is_leaf
        self.leaf_weight = leaf_weight
        self.split_feat = split_feat
        self.split_val = split_val
        self.left_child = left_child
        self.right_child = right_child

    def to_dict(self):
        """transform object to dict"""
        return {k: v.to_dict() if isinstance(v, MPCTreeNode) else v for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, data):
        """transform from dict to object"""
        obj = cls()
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(obj, k, cls.from_dict(v))
            else:
                setattr(obj, k, v)
        return obj


class MPCTree:
    """
    Tree class

    Parameters
    ----------
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

    min_split_loss: float, default=1e-5
        The minimum number of gain to split an internal node:
        Values must be in the range `[0.0, inf)

    max_depth : int or None, default=3
        Maximum depth of the tree.
        If int, values must be in the range `[1, inf)`.

    columns: SecureArray
        Encrypted column names in the training data to prevent leakage during training.

    Attributes
    ----------
    root: MPCTreeNode
        The root node of the tree

    """

    def __init__(self,
                 reg_alpha: float = None,
                 reg_lambda: float = None,
                 min_child_weight: float = None,
                 min_child_samples: int = None,
                 min_split_loss: float = None,
                 max_depth: int = None,
                 columns: SecureArray = None):

        self.columns = columns
        self.root = None
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.min_split_loss = min_split_loss
        self.max_depth = max_depth

    def to_dict(self):
        """transform object to dict"""
        return {k: v.to_dict() if isinstance(v, MPCTreeNode) else v for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, data):
        """transform from dict to object"""
        obj = cls()
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(obj, k, cls.from_dict(v))
            else:
                setattr(obj, k, v)
        return obj

    def _calc_threshold(self, gsum):
        """clip the value of gain"""
        res = snp.where(gsum > self.reg_alpha, gsum - self.reg_alpha,
                        snp.where(gsum < -1. * self.reg_alpha, gsum + self.reg_alpha, snp.zeros(1)))
        return res[0]

    def _calc_leaf_weight(self, best_gradient, best_hessian):
        weight = -self._calc_threshold(best_gradient) / (best_hessian + self.reg_lambda)
        return weight

    def _calc_split_gain(self, left_sum_grads, right_sum_grads, left_sum_hess, right_sum_hess):

        left_sum_grads_threshold = self._calc_threshold(left_sum_grads)
        right_sum_grads_threshold = self._calc_threshold(right_sum_grads)
        sum_grads_threshold = self._calc_threshold((left_sum_grads + right_sum_grads))

        gain = left_sum_grads_threshold * left_sum_grads_threshold / (
            left_sum_hess + self.reg_lambda) + right_sum_grads_threshold * right_sum_grads_threshold / (
                right_sum_hess + self.reg_lambda) - sum_grads_threshold * sum_grads_threshold / (
                    left_sum_hess + right_sum_hess + self.reg_lambda)

        return gain

    def _find_best_split(self, X, grads, hess, sum_grads, sum_hess, index_array):
        """find the best split feature and value in current node"""
        max_gain, best_feat, best_split_val, best_feat_idx = snp.zeros(1), snp.zeros(1), snp.zeros(1), snp.zeros(1)

        for feat_idx, item in enumerate(self.columns):
            x = X[:, feat_idx]
            filter_x = snp.reshape(x, (-1, 1)) * index_array
            cur_feat_data = snp.hstack((filter_x, grads, hess))
            sorted_cur_feat_data = cur_feat_data.quick_sort_by_column(0)
            left_sum_grads, left_sum_hess = np.array([0.]), np.array([0.])

            for i in range(sorted_cur_feat_data.shape[0] - 1):
                x_i = sorted_cur_feat_data[i, 0]
                x_i_next = sorted_cur_feat_data[i + 1, 0]
                grads_i = sorted_cur_feat_data[i, 1]
                hess_i = sorted_cur_feat_data[i, 2]
                left_sum_grads += grads_i
                right_sum_grads = sum_grads - left_sum_grads
                left_sum_hess += hess_i
                right_sum_hess = sum_hess - left_sum_hess

                same_val_flag = snp.where(
                    snp.reshape(x_i, (-1,)) == snp.reshape(x_i_next, (-1,)), snp.zeros(1), snp.ones(1))

                if (self.min_child_samples and (i + 1 < self.min_child_samples or
                                                (len(x) - i - 1) < self.min_child_samples)):
                    continue

                gain = self._calc_split_gain(left_sum_grads, right_sum_grads, left_sum_hess, right_sum_hess)
                gain = gain * same_val_flag[0]
                if self.min_child_weight:
                    left_hess_flag = snp.where(left_sum_hess < self.min_child_weight, snp.zeros(left_sum_hess.shape),
                                               snp.ones(left_sum_hess.shape))
                    right_hess_flag = snp.where(right_sum_hess < self.min_child_weight, snp.zeros(right_sum_hess.shape),
                                                snp.ones(right_sum_hess.shape))

                    gain = gain * left_hess_flag[0] * right_hess_flag[0]

                gain_condition = gain > max_gain
                max_gain = snp.where(gain_condition, gain, max_gain)
                best_feat = snp.where(gain_condition, snp.reshape(item, (-1,)), best_feat)
                best_split_val = snp.where(gain_condition, snp.reshape(0.5 * (x_i + x_i_next), (-1,)), best_split_val)
                best_feat_idx = snp.where(gain_condition, snp.ones(1) * feat_idx, best_feat_idx)

        best_feat = snp.where(max_gain < self.min_split_loss, snp.zeros(1), best_feat)
        return best_feat, best_split_val, best_feat_idx

    def _create_leaf_node(self, grads, hess):
        is_leaf = True
        grads_sum = snp.reshape(snp.sum(grads), (-1,))
        hess_sum = snp.reshape(snp.sum(hess), (-1,))
        leaf_weight = self._calc_leaf_weight(grads_sum, hess_sum)
        return MPCTreeNode(is_leaf=is_leaf, leaf_weight=leaf_weight)

    def build_tree(self,
                   X: SecureArray,
                   grads: SecureArray,
                   hess: SecureArray,
                   index_array: SecureArray,
                   current_depth: int,
                   last_layer_feat: SecureArray = None,
                   last_layer_feat_val: SecureArray = None,
                   last_layer_feat_idx: SecureArray = None):
        """
        Build the tree node recursively. Will only stop when it reaches the maximum depth.
        When the current node cannot find the optimal split point, it will use the value
        of its parent node as a substitute.

        Parameters
        ----------
        X : SecureArray
           train data

        grads: SecureArray
            Secret sharing data of first order derivative

        hess: SecureArray
            Secret sharing data of second order derivative

        index_array: array or SecureArray
            An array to record the sample is in current node or not

        current_depth: int
            current depth of the node

        last_layer_feat: SecureArray or None
            Best split feature in secret share format in its parent node

        last_layer_feat_val: SecureArray or None
            Split value of the best feature in secret share format in its parent node

        last_layer_feat_idx: SecureArray or None
            Best feature idx in secret share format in its parent node

        Returns
        ----------
        sub_tre: MPCTreeNode
            Trained tree
        """
        cur_grads = grads * index_array
        cur_hess = hess * index_array

        if self.max_depth - 1 < current_depth:
            return self._create_leaf_node(cur_grads, cur_hess)

        sum_grads, sum_hess = snp.reshape(snp.sum(cur_grads), (-1,)), snp.reshape(snp.sum(cur_hess), (-1,))
        if current_depth == 0:
            cur_child_samples = np.sum(index_array)
            child_samples_flag = np.where(cur_child_samples < self.min_child_samples, np.zeros(1), np.ones(1))
        else:
            cur_child_samples = snp.reshape(snp.sum(index_array), (-1,))
            child_samples_flag = snp.where(cur_child_samples < self.min_child_samples,
                                           snp.zeros(cur_child_samples.shape), snp.ones(cur_child_samples.shape))

        child_weight_flag = snp.where(sum_hess < self.min_child_weight, snp.zeros(sum_hess.shape),
                                      snp.ones(sum_hess.shape))

        best_feat, best_split_val, best_feat_idx = self._find_best_split(
            X, cur_grads * child_weight_flag[0] * child_samples_flag[0],
            cur_hess * child_weight_flag[0] * child_samples_flag[0], sum_grads, sum_hess, index_array)

        check_flag = best_feat * child_weight_flag * child_samples_flag
        flag_compare_res = check_flag == 0

        if current_depth == 0:
            best_feat = snp.where(flag_compare_res, snp.reshape(self.columns[0], (1,)), best_feat)
            best_split_val = snp.where(flag_compare_res, snp.min(X[:, 0]), best_split_val)
            best_feat_idx = snp.where(flag_compare_res, snp.zeros(1), best_feat_idx)
        else:
            best_feat = snp.where(flag_compare_res, last_layer_feat, best_feat)
            best_split_val = snp.where(flag_compare_res, last_layer_feat_val, best_split_val)
            best_feat_idx = snp.where(flag_compare_res, last_layer_feat_idx, best_feat_idx)

        col_array = np.arange(self.columns.shape[0])
        col_array_matrix = np.tile(col_array, X.shape[0]).reshape(X.shape[0], -1)
        get_fidx = snp.where(col_array_matrix == best_feat_idx[0], snp.ones(col_array_matrix.shape),
                             snp.zeros(col_array_matrix.shape))
        filter_x = X * get_fidx
        x = snp.sum(filter_x, axis=1)
        x = snp.reshape(x, (-1, 1))
        left_idx = snp.where(x < best_split_val[0], snp.ones(x.shape), snp.zeros(x.shape))
        right_idx = snp.where(x >= best_split_val[0], snp.ones(x.shape), snp.zeros(x.shape))
        left_idx = snp.reshape(left_idx, (-1, 1))
        right_idx = snp.reshape(right_idx, (-1, 1))

        left_tree = self.build_tree(X, grads, hess, left_idx * index_array, current_depth + 1, best_feat,
                                    best_split_val, best_feat_idx)
        right_tree = self.build_tree(X, grads, hess, right_idx * index_array, current_depth + 1, best_feat,
                                     best_split_val, best_feat_idx)

        sub_tree = MPCTreeNode(is_leaf=False,
                               leaf_weight=None,
                               split_feat=best_feat,
                               split_val=best_split_val,
                               left_child=left_tree,
                               right_child=right_tree)

        return sub_tree

    def predict_proba(self, tree_node: MPCTreeNode, data: SecureArray, columns: SecureArray,
                      last_layer_flag: SecureArray):
        """
        Predict probabilities. Each piece of data will be compared with all nodes of the tree to ensure that
        data information is not leaked.

        Parameters
        ----------
        tree_node : MPCTreeNode
           Node in tree

        data: SecureArray
            Secret sharing of predict data

        columns: array or SecureArray
            Columns in data

        last_layer_flag: SecureArray
            Check the data is in current node or not

        Returns
        ----------
        sub_tre: MPCTreeNode
            Trained tree
        """
        leaf_values = []
        if tree_node.is_leaf:
            weight_flag = snp.where(last_layer_flag == self.max_depth, snp.ones(1), snp.zeros(1))
            leaf_values.append(tree_node.leaf_weight * weight_flag)
            return leaf_values

        get_fidx = snp.where(columns == tree_node.split_feat[0], snp.ones(columns.shape), snp.zeros(columns.shape))

        cur_col_data = data * get_fidx
        cur_col_data = snp.sum(cur_col_data, axis=1)

        left_flag = snp.where(cur_col_data < tree_node.split_val, snp.ones(1), snp.zeros(1))
        right_flag = snp.where(cur_col_data >= tree_node.split_val, snp.ones(1), snp.zeros(1))
        leaf_values.extend(self.predict_proba(tree_node.left_child, data, columns, left_flag + last_layer_flag))
        leaf_values.extend(self.predict_proba(tree_node.right_child, data, columns, right_flag + last_layer_flag))

        return leaf_values
