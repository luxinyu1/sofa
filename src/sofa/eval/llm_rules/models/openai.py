import os
from typing import List

from dotenv import load_dotenv
import openai
from tenacity import (
    retry,
    retry_if_not_exception_type,
    wait_random_exponential,
)

from ..message import Message
from .base import BaseModel, build_prompt


OPENAI_MODELS = [
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-3.5-turbo-instruct-0914",
]
OPENAI_DEFAULT = "gpt-3.5-turbo-0613"

COMPLETION_MODELS = [
    "gpt-3.5-turbo-instruct-0914",
]


class Response:
    """Response wrapper class for OpenAI API response object.

    Implements the iterator interface to enable simple iteration over streamed response chunks, such that
    `"".join(response)` returns the full completion.
    """

    def __init__(self, response, stream=False):
        self.response = response
        self.stream = stream
        self.complete = False
        if self.stream:
            self.response_iter = iter(self.response)
        else:
            self.response = self.response["choices"][0]["message"]["content"]

    def __iter__(self):
        return self

    def __next__(self):
        if self.complete:
            raise StopIteration

        if not self.stream:
            self.complete = True
            return self.response

        try:
            chunk = next(self.response_iter)
            delta = chunk["choices"][0]["delta"]
            return delta.get("content", "")
        except StopIteration as e:
            self.complete = True
            raise e


class CompletionResponse:
    """Response wrapper class for OpenAI API completion response object.

    Implements the iterator interface to enable simple iteration over streamed response chunks, such that
    `"".join(response)` returns the full completion.
    """

    def __init__(self, response, stream=False):
        self.response = response
        self.stream = stream
        self.complete = False
        if self.stream:
            self.response_iter = iter(self.response)
        else:
            self.response = self.response["choices"][0]["text"]

    def __iter__(self):
        return self

    def __next__(self):
        if self.complete:
            raise StopIteration

        if not self.stream:
            self.complete = True
            return self.response

        try:
            chunk = next(self.response_iter)
            delta = chunk["choices"][0]["text"]
            return delta.get("content", "")
        except StopIteration as e:
            self.complete = True
            raise e


class OpenAIModel(BaseModel):
    """Model builder for OpenAI API models.

    Call with a list of `Message` objects to generate a response.
    """

    supports_system_message = True

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        stream: bool = False,
        top_p: float = 1.0,
        max_tokens: int = 512,
        stop=None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict = {},
        **kwargs,
    ):
        self.model = model
        self.temperature = temperature
        self.stream = stream
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logit_bias = logit_bias
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        if self.model in COMPLETION_MODELS:
            self.supports_system_message = False

    def encode(self, messages: List[Message]):
        return [{"role": m.role.name.lower(), "content": m.content} for m in messages]

    def __call__(self, messages: List[Message], api_key: str = None):
        if api_key is not None:
            openai.api_key = api_key

        if self.model in COMPLETION_MODELS:
            prompt = (
                build_prompt(messages) if len(messages) > 0 else messages[0].content
            )
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                top_p=self.top_p,
                n=1,
                stream=self.stream,
                max_tokens=self.max_tokens,
                stop=self.stop,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                logit_bias=self.logit_bias,
            )
            if api_key is not None:
                openai.api_key = os.getenv("OPENAI_API_KEY", "")
            return CompletionResponse(response, stream=self.stream)
        else:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.encode(messages),
                temperature=self.temperature,
                top_p=self.top_p,
                n=1,
                stream=self.stream,
                max_tokens=self.max_tokens,
                stop=self.stop,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                logit_bias=self.logit_bias,
            )
            if api_key is not None:
                openai.api_key = os.getenv("OPENAI_API_KEY", "")
            return Response(response, stream=self.stream)


@retry(
    retry=retry_if_not_exception_type(openai.InvalidRequestError),
    wait=wait_random_exponential(min=1, max=10),
)
def openai_call_with_retries(model, messages, api_key=None):
    return model(messages, api_key)
