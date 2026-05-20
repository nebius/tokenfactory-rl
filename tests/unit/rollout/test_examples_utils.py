from unittest.mock import Mock

from tokenfactory.rl.examples.utils import parse_completion


def test_parse_completion_combines_prompt_and_completion_tokens():
    completion = Mock()
    completion.model_extra = {"prompt_token_ids": [11, 12]}

    completion.choices = [Mock()]
    completion.choices[0].logprobs = Mock()
    completion.choices[0].logprobs.content = [
        Mock(token="token_id:21", logprob=-0.1),
        Mock(token="token_id:22", logprob=-0.2),
    ]

    token_ids, logprobs, mask = parse_completion(completion)

    assert token_ids == [11, 12, 21, 22]
    assert logprobs == [0.0, 0.0, -0.1, -0.2]
    assert mask == [0, 0, 1, 1]
