class JobInitializationTimeout(Exception):  # noqa:N818
    def __init__(
        self,
        *,
        job_id: str,
        timeout_seconds: int,
        last_exception: Exception | None = None,
    ):
        message = f"Rollout job {job_id!r} did not become ready within {timeout_seconds} seconds."
        if last_exception is not None:
            message = f"{message} Last polling error: {last_exception}"

        super().__init__(message)
        self.job_id = job_id
        self.timeout_seconds = timeout_seconds
        self.last_exception = last_exception
