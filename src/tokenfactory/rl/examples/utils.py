from openai.types.chat import ChatCompletion


TokenIDs = list[int]
Logprobs = list[float]
Mask = list[int]


def parse_completion(completion: ChatCompletion) -> tuple[TokenIDs, Logprobs, Mask]:
    assert completion.model_extra is not None
    prompt_token_ids = completion.model_extra["prompt_token_ids"]
    assert completion.choices[0].logprobs is not None
    assert completion.choices[0].logprobs.content is not None
    completion_token_ids = [
        int(x.token.split(":")[1]) for x in completion.choices[0].logprobs.content
    ]
    completion_logprobs = [x.logprob for x in completion.choices[0].logprobs.content]
    return (
        prompt_token_ids + completion_token_ids,
        [0.0] * len(prompt_token_ids) + completion_logprobs,
        [0] * len(prompt_token_ids) + [1] * len(completion_token_ids),
    )
