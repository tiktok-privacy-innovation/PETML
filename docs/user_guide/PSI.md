# PSI

## Introduction
In federated learning, a secure and dependable method is required to find any intersection in features or IDs between
datasets from two parties. The intersection data will then be used to update the model. We focus on the setting where
two parties (e.g., A & B) hold vertically partitioned data. They need to align their data before training the model.

Private Set Intersection (PSI) protocol is well-suited to solve this problem. Our implementation relies on the elliptic
curve Diffie-Hellman PSI (ECDH-PSI) provided by [PETAce](https://github.com/tiktok-privacy-innovation/PETAce).
(More detailed information is available in the PETAce documentation.)
To improve user experience, we encapsulated the PSI function in PETAce, presenting it as an intersection module.
Users can simply upload a CSV file along with the names of the columns to be aligned, enabling them to obtain their
aligned data effortlessly.

## Config

### Module Name
```
petml.operators.preprocessing.PSITransform
```

### Model Parameters
| Name              | Type | Description                      | Default |
|-------------------|------|----------------------------------|---------|
| ```column_name``` | str  | The column used to implement PSI |         |

### Input
| Name | File Type | Description      |
|------|-----------|------------------|
| data | csv       | The data for PSI |

### Output

| Name         | File Type | Description              |
|--------------|-----------|--------------------------|
| intersection | csv       | The intersection of data |

### Examples
```
config = {
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
        "column_name": "id",
        "inputs": {
            "data": "examples/data/breast_hetero_mini_server.csv",
        },
        "outputs": {
            "data": "tmp/server/data.csv"
        }
    },
    "party_b": {
        "column_name": "id",
        "inputs": {
            "data": "examples/data/breast_hetero_mini_client.csv",
        },
        "outputs": {
            "data": "tmp/client/data.csv"
        }
    }
}

#if run the code in party a, the party should be 'party_a' and vice versa
operator = petml.operators.preprocessing.PSITransform(party)
operator.run(config_map)
```
