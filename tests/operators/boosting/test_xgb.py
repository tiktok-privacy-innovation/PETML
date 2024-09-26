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
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

from petml.operators.boosting import XGBoostClassifierFit, XGBoostClassifierPredict, XGBoostRegressorFit, \
    XGBoostRegressorPredict
from tests.utils import run_multi_process


class TestXGBoostClassifier:

    def _run_fit_classifier(self, party, config_map):
        operator = XGBoostClassifierFit(party)
        operator.run(config_map)

    def _run_predict_classifier(self, party, config_map):
        operator = XGBoostClassifierPredict(party)
        operator.run(config_map)

    def test_xgb_classifier(self):
        fit_configmap = {
            "common": {
                "objective:": "logitraw",
                "n_estimators": 1,
                "max_depth": 2,
                "reg_lambda": 1,
                "reg_alpha": 0.0,
                "base_score": 0.5,
                "learning_rate": 0.1,
                "min_child_weight": 0.1,
                "network_mode": "petnet",
                "network_scheme": "socket",
                "label_name": "label",
                "test_size": 0.,
                "parties": {
                    "party_a": {
                        "address": ["127.0.0.1:50011"]
                    },
                    "party_b": {
                        "address": ["127.0.0.1:50012"]
                    }
                }
            },
            "party_a": {
                "inputs": {
                    "train_data": "examples/data/iris_binary_mini_server.csv",
                },
                "outputs": {
                    "model_path": "tmp/test_binary_xgb_server.json"
                }
            },
            "party_b": {
                "inputs": {
                    "train_data": "examples/data/iris_binary_mini_client.csv",
                },
                "outputs": {
                    "model_path": "tmp/test_binary_xgb_client.json"
                }
            }
        }
        predict_configmap = {
            "common": {
                "network_mode": "petnet",
                "network_scheme": "socket",
                "parties": {
                    "party_a": {
                        "address": ["127.0.0.1:50011"]
                    },
                    "party_b": {
                        "address": ["127.0.0.1:50012"]
                    }
                }
            },
            "party_a": {
                "inputs": {
                    "predict_data": "examples/data/iris_binary_mini_server.csv",
                    "model_path": "tmp/test_binary_xgb_server.json"
                },
                "outputs": {
                    "inference_res_path": "tmp/test_binary_predict_server.csv"
                }
            },
            "party_b": {
                "inputs": {
                    "predict_data": "examples/data/iris_binary_mini_client.csv",
                    "model_path": "tmp/test_binary_xgb_client.json"
                },
                "outputs": {
                    "inference_res_path": "tmp/test_binary_predict_client.csv"
                }
            }
        }
        data1 = pd.read_csv(predict_configmap['party_a']['inputs']['predict_data'])
        data2 = pd.read_csv(predict_configmap['party_b']['inputs']['predict_data'])
        true_label = pd.concat([data1, data2], axis=0)['label'].values
        run_multi_process(self._run_fit_classifier, [("party_a", fit_configmap), ("party_b", fit_configmap)])
        run_multi_process(self._run_predict_classifier, [("party_a", predict_configmap),
                                                         ("party_b", predict_configmap)])
        server_result_save_path = predict_configmap["party_a"]["outputs"]["inference_res_path"]
        client_result_save_path = predict_configmap["party_b"]["outputs"]["inference_res_path"]
        server_y_pred = pd.read_csv(f"{server_result_save_path}").values
        client_y_pred = pd.read_csv(f"{client_result_save_path}").values
        y_pred = np.vstack((server_y_pred, client_y_pred))
        y_pred = np.where(y_pred >= 0.5, 1, 0)
        acc_value = accuracy_score(true_label, y_pred)
        assert abs(acc_value - 1) < 0.01


class TestXGBoostRegressor:

    def _run_fit_regressor(self, party, config_map):
        operator = XGBoostRegressorFit(party)
        operator.run(config_map)

    def _run_predict_regressor(self, party, config_map):
        operator = XGBoostRegressorPredict(party)
        operator.run(config_map)

    def test_xgb_regressor(self):
        fit_configmap = {
            "common": {
                "objective:": "squarederror",
                "n_estimators": 1,
                "max_depth": 2,
                "reg_lambda": 1,
                "reg_alpha": 0.0,
                "base_score": 0.5,
                "learning_rate": 0.1,
                "min_child_weight": 1,
                "network_mode": "petnet",
                "network_scheme": "socket",
                "label_name": "label",
                "test_size": 0.,
                "parties": {
                    "party_a": {
                        "address": ["127.0.0.1:50011"]
                    },
                    "party_b": {
                        "address": ["127.0.0.1:50012"]
                    }
                }
            },
            "party_a": {
                "inputs": {
                    "train_data": "examples/data/students_reg_mini_server.csv",
                },
                "outputs": {
                    "model_path": "tmp/test_reg_xgb_server.json"
                }
            },
            "party_b": {
                "inputs": {
                    "train_data": "examples/data/students_reg_mini_client.csv",
                },
                "outputs": {
                    "model_path": "tmp/test_reg_xgb_client.json"
                }
            }
        }
        predict_configmap = {
            "common": {
                "network_mode": "petnet",
                "network_scheme": "socket",
                "parties": {
                    "party_a": {
                        "address": ["127.0.0.1:50011"]
                    },
                    "party_b": {
                        "address": ["127.0.0.1:50012"]
                    }
                }
            },
            "party_a": {
                "inputs": {
                    "predict_data": "examples/data/students_reg_mini_server.csv",
                    "model_path": "tmp/test_reg_xgb_server.json"
                },
                "outputs": {
                    "inference_res_path": "tmp/test_reg_predict_server.csv"
                }
            },
            "party_b": {
                "inputs": {
                    "predict_data": "examples/data/students_reg_mini_client.csv",
                    "model_path": "tmp/test_reg_xgb_client.json"
                },
                "outputs": {
                    "inference_res_path": "tmp/test_reg_predict_client.csv"
                }
            }
        }
        data1 = pd.read_csv(predict_configmap['party_a']['inputs']['predict_data'])
        data2 = pd.read_csv(predict_configmap['party_b']['inputs']['predict_data'])
        true_label = pd.concat([data1, data2], axis=0)['label'].values
        run_multi_process(self._run_fit_regressor, [("party_a", fit_configmap), ("party_b", fit_configmap)])
        run_multi_process(self._run_predict_regressor, [("party_a", predict_configmap), ("party_b", predict_configmap)])
        server_result_save_path = predict_configmap["party_a"]["outputs"]["inference_res_path"]
        client_result_save_path = predict_configmap["party_b"]["outputs"]["inference_res_path"]
        server_y_pred = pd.read_csv(f"{server_result_save_path}").values
        client_y_pred = pd.read_csv(f"{client_result_save_path}").values
        y_pred = np.vstack((server_y_pred, client_y_pred))
        mse_error_value = mean_squared_error(true_label, y_pred)
        assert abs(mse_error_value - 122.04) < 10
