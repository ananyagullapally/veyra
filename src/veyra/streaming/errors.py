import httpx


class PermanentProcessingError(Exception):
    """An error that should not be retried."""


class TransientProcessingError(Exception):
    """An error that may succeed when retried."""


def classify_http_error(error: Exception) -> Exception:
    if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
        return TransientProcessingError(str(error))

    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code

        if status in (408, 429) or 500 <= status < 600:
            return TransientProcessingError(str(error))

        if 400 <= status < 500:
            return PermanentProcessingError(str(error))

    return error