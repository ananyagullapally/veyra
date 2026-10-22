import pytest
from unittest.mock import MagicMock, patch
from veyra.training.model_registry import get_champion_info

def test_get_champion_info():
    champion = MagicMock()
    champion.name = "veyra-fraud-detector"
    champion.version = "1"
    champion.run_id = "test-run-123"
    champion.source = "models:/m-test123"

    with (
        patch("veyra.training.model_registry.configure_mlflow"),
        patch("veyra.training.model_registry.MlflowClient") as mock_client,
    ):
        mock_client.return_value.get_model_version_by_alias.return_value = champion
        mock_run = MagicMock()
        mock_run.data.params = {
            "selected_threshold": "0.01320815272629261"
        }

        mock_client.return_value.get_run.return_value = mock_run
        result = get_champion_info()

    assert result == {
        "model_name": "veyra-fraud-detector",
        "version": "1",
        "run_id": "test-run-123",
        "source": "models:/m-test123",
        "model_uri": "models:/veyra-fraud-detector@champion",
        "threshold": 0.01320815272629261,
    }

    mock_client.return_value.get_model_version_by_alias.assert_called_once_with(
        name="veyra-fraud-detector",
        alias="champion",
    )

    mock_client.return_value.get_run.assert_called_once_with(
        "test-run-123"
    )

@pytest.mark.parametrize(
    "threshold_value",
    [None, "-0.1", "1.5", "invalid"],
)
def test_get_champion_info_invalid_threshold(threshold_value):
    champion = MagicMock()
    champion.run_id = "test-run-123"

    with (
        patch("veyra.training.model_registry.configure_mlflow"),
        patch("veyra.training.model_registry.MlflowClient") as mock_client,
    ):
        mock_client.return_value.get_model_version_by_alias.return_value = champion

        mock_run = MagicMock()
        mock_run.data.params = {}

        if threshold_value is not None:
            mock_run.data.params["selected_threshold"] = threshold_value

        mock_client.return_value.get_run.return_value = mock_run

        with pytest.raises(ValueError):
            get_champion_info()