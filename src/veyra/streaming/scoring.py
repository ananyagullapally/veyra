import httpx

from veyra.streaming.errors import classify_http_error
from veyra.streaming.retry import retry_operation


API_URL = "http://127.0.0.1:8000/predict"


def score_application(features: dict) -> dict:
    def request_prediction():
        try:
            response = httpx.post(
                API_URL,
                json={"features": features},
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as error:
            raise classify_http_error(error) from error

    return retry_operation(
        request_prediction,
        max_attempts=3,
    )