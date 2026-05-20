from httpx import Response


class TokenFactoryError(Exception):
    pass


class MissingAPIKeyError(TokenFactoryError):
    pass


class APIError(TokenFactoryError):
    def __init__(self, msg: str, response: Response | None = None):
        self.msg = msg
        self.response = response


class InformationalResponseError(APIError):
    pass


class RedirectResponseError(APIError):
    pass


class BadRequestError(APIError):
    pass


class NotFoundError(APIError):
    pass


class InternalServerError(APIError):
    pass


class JsonParseError(APIError):
    pass
