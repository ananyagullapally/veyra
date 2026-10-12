import numpy as np
import pandas as pd

from veyra.features.preprocessing import build_preprocessor


def test_preprocessor_handles_missing_and_categorical_values():
    X = pd.DataFrame(
        {
            "income": [0.2, np.nan, 0.8],
            "customer_age": [20, 40, 60],
            "device_os": ["linux", "windows", "linux"],
        }
    )

    numeric_columns = [
        "income",
        "customer_age",
    ]

    categorical_columns = [
        "device_os",
    ]

    preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    transformed = preprocessor.fit_transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert transformed.shape[0] == 3
    assert not np.isnan(transformed).any()