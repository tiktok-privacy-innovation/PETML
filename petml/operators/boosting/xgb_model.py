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

from pathlib import Path

from petml.fl.boosting import XGBoostClassifier, XGBoostRegressor
from petml.infra.engine.cipher_engine import CipherEngine
from petml.infra.storage.tabular_storage import CsvStorage
from petml.operators.operator_base import OperatorBase


class XGBoostClassifierFit(OperatorBase):

    def _run(self, net, configs: dict) -> bool:
        """
        Train a xgboost classification model base on the data.

        Expects the following configmap:
        {
            "common": {
                "objective": "logitraw",
                "n_estimators": 100,
                "max_depth": 3,
                "reg_lambda": 1,
                "reg_alpha": 0.0,
                "min_child_weight": 0.5,
                "base_score": 0.5,
                "learning_rate": 0.1,
                "network_mode": "petnet",
                "network_scheme": "socket",
                "label_name": "label",
                "test_size": 0.3,
                "parties": {
                    "party_a": {
                        "address": ["IP_ADDRESS:50011"]
                    },
                    "party_b": {
                        "address": ["IP_ADDRESS:50012"]
                    }
                }
            },
            "party_a": {
                "inputs": {
                    "train_data": "/path/to/data.csv",
                },
                "outputs": {
                    "model_path": "/path/to/model_name.pkl"
                }
            },
            "party_b": {
                "inputs": {
                    "train_data": "/path/to/data.csv",
                },
                "outputs": {
                    "model_path": "/path/to/model_name.pkl"
                }
            }
        }
        """
        min_split_loss = configs.get("min_split_loss", 1e-5)
        learning_rate = configs.get("learning_rate", 0.1)
        n_estimators = configs.get("n_estimators", 100)
        base_score = configs.get("base_score", 0.5)
        max_depth = configs.get("max_depth", 3)
        reg_alpha = configs.get("reg_alpha", 0.)
        reg_lambda = configs.get("reg_lambda", 1.0)
        min_child_samples = configs.get("min_child_samples", 1)
        min_child_weight = configs.get("min_child_weight", 1)
        label_name = configs.get("label_name", "label")
        objective = configs.get("objective", "logitraw")
        test_size = configs.get("test_size", 0.3)

        # init infra
        mpc_engine = CipherEngine("mpc", self.party_id, net)

        # init io
        train_data = CsvStorage.read(configs["inputs"]["train_data"])
        model_path = configs["outputs"]["model_path"]
        ext = Path(model_path)
        if ext.suffix != '.pkl':
            raise ValueError('The `model_path` should end with the `.pkl` format.')

        # construct model
        model = XGBoostClassifier(min_split_loss,
                                  learning_rate,
                                  n_estimators,
                                  base_score,
                                  max_depth,
                                  reg_alpha,
                                  reg_lambda,
                                  min_child_samples,
                                  min_child_weight,
                                  label_name,
                                  test_size=test_size,
                                  objective=objective)
        model.set_infra(self.party_id, net, mpc_engine)
        model.fit(train_data)
        model.save_model(model_path)
        return True


class XGBoostClassifierPredict(OperatorBase):

    def _run(self, net, configs: dict) -> bool:
        """
        Train a xgboost classification model base on the data.

        Expects the following configmap:
        {
            "common": {
                "network_mode": "petnet",
                "network_scheme": "socket",
                "parties": {
                    "party_a": {
                        "address": ["IP_ADDRESS:50011"]
                    },
                    "party_b": {
                        "address": ["IP_ADDRESS:50012"]
                    }
                }
            },
            "party_a": {
                "inputs": {
                    "predict_data": "/path/to/data.csv",
                    "model_path": "/path/to/model_name.pkl"
                },
                "outputs": {
                    "inference_res_path": "pathto/predict_proba_value.csv"
                }
            },
            "party_b": {
               "inputs": {
                    "predict_data": "/path/to/data.csv",
                    "model_path": "/path/to/model_name.pkl"
                },
                "outputs": {
                    "inference_res_path": "/path/to/predict_proba_value.csv"
                }
            }
        }
        """
        # init infra
        mpc_engine = CipherEngine("mpc", self.party_id, net)

        # init io
        predict_data = CsvStorage.read(configs["inputs"]["predict_data"])
        model_path = configs["inputs"]["model_path"]
        ext = Path(model_path)
        if ext.suffix != '.pkl':
            raise ValueError('The `model_path` should end with the `.pkl` format.')
        inference_res_path = configs["outputs"]["inference_res_path"]

        # inference model
        model = XGBoostClassifier()
        model.set_infra(self.party_id, net, mpc_engine)
        model.load_model(model_path)
        predict_proba = model.predict(predict_data)
        CsvStorage.write(predict_proba, f"{inference_res_path}", index=False)
        return True


class XGBoostRegressorFit(OperatorBase):

    def _run(self, net, configs: dict) -> bool:
        """
        Train a xgboost classification model base on the data.

        Expects the following configmap:
        {
            "common": {
                "objective": "squarederror",
                "n_estimators": 100,
                "max_depth": 3,
                "reg_lambda": 1,
                "reg_alpha": 0.0,
                "min_child_weight": 1,
                "base_score": 0.5,
                "learning_rate": 0.1,
                "network_mode": "petnet",
                "network_scheme": "socket",
                "label_name": "label",
                "test_size": 0.3,
                "parties": {
                    "party_a": {
                        "address": ["IP_ADDRESS:50011"]
                    },
                    "party_b": {
                        "address": ["IP_ADDRESS:50012"]
                    }
                }
            },
            "party_a": {
                "inputs": {
                    "train_data": "/path/to/data.csv",
                },
                "outputs": {
                    "model_path": "/path/to/model_name.pkl"
                }
            },
            "party_b": {
                "inputs": {
                    "train_data": "/path/to/data.csv",
                },
                "outputs": {
                    "model_path": "/path/to/model_name.pkl"
                }
            }
        }
        """
        min_split_loss = configs.get("min_split_loss", 1e-5)
        learning_rate = configs.get("learning_rate", 0.1)
        n_estimators = configs.get("n_estimators", 100)
        base_score = configs.get("base_score", 0.5)
        max_depth = configs.get("max_depth", 3)
        reg_alpha = configs.get("reg_alpha", 0.)
        reg_lambda = configs.get("reg_lambda", 1.0)
        min_child_samples = configs.get("min_child_samples", 1)
        min_child_weight = configs.get("min_child_weight", 1)
        label_name = configs.get("label_name", "label")
        objective = configs.get("objective", "squarederror")
        test_size = configs.get("test_size", 0.3)

        # init infra
        mpc_engine = CipherEngine("mpc", self.party_id, net)

        # init io
        train_data = CsvStorage.read(configs["inputs"]["train_data"])
        model_path = configs["outputs"]["model_path"]
        ext = Path(model_path)
        if ext.suffix != '.pkl':
            raise ValueError('The `model_path` should end with the `.pkl` format.')

        # construct model
        model = XGBoostRegressor(min_split_loss,
                                 learning_rate,
                                 n_estimators,
                                 base_score,
                                 max_depth,
                                 reg_alpha,
                                 reg_lambda,
                                 min_child_samples,
                                 min_child_weight,
                                 label_name,
                                 test_size=test_size,
                                 objective=objective)
        model.set_infra(self.party_id, net, mpc_engine)
        model.fit(train_data)
        model.save_model(model_path)
        return True


class XGBoostRegressorPredict(OperatorBase):

    def _run(self, net, configs: dict) -> bool:
        """
        Train a xgboost classification model base on the data.

        Expects the following configmap:
        {
            "common": {
                "network_mode": "petnet",
                "network_scheme": "socket",
                "parties": {
                    "party_a": {
                        "address": ["IP_ADDRESS:50011"]
                    },
                    "party_b": {
                        "address": ["IP_ADDRESS:50012"]
                    }
                }
            },
            "party_a": {
                "inputs": {
                    "predict_data": "/path/to/data.csv",
                    "model_path": "/path/to/model_name.pkl"
                },
                "outputs": {
                    "inference_res_path": "/path/to/predict_value.csv"
                }
            },
            "party_b": {
               "inputs": {
                    "predict_data": "/path/to/data.csv",
                    "model_path": "/path/to/model_name.pkl"
                }
                "outputs": {
                    "inference_res_path": "path/to/predict_value.csv"
                }
            }
        }
        """
        # init infra
        mpc_engine = CipherEngine("mpc", self.party_id, net)

        # init io
        predict_data = CsvStorage.read(configs["inputs"]["predict_data"])
        model_path = configs["inputs"]["model_path"]
        ext = Path(model_path)
        if ext.suffix != '.pkl':
            raise ValueError('The `model_path` should end with the `.pkl` format.')
        inference_res_path = configs["outputs"]["inference_res_path"]

        # inference model
        model = XGBoostRegressor()
        model.set_infra(self.party_id, net, mpc_engine)
        model.load_model(model_path)
        predict_label = model.predict(predict_data)

        CsvStorage.write(predict_label, f"{inference_res_path}", index=False)
        return True
