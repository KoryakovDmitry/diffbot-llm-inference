import logging
from typing import List, Dict, Union
from openai import OpenAI
from cog import BasePredictor, Input, Secret, AsyncConcatenateIterator
from typing_extensions import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class DiffBotReplicateServer(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    async def predict(
        self,
        diffbot_api_token: Secret,
        model: str = Input(
            description="Name of model", choices=["diffbot-small-xl", "diffbot-small"]
        ),
        messages: List[Dict[str, str]] = Input(
            description="Messages OpenAI alike: "
            '{"role": ["system", "user", "assistant", "tool"], "content": string}'
        ),
        temperature: Optional[float] = Input(
            description="Temperature of model", default=0, ge=0, le=2
        ),
        frequency_penalty: Optional[float] = Input(
            description="frequency_penalty of model", default=0.0, ge=-2.0, le=2.0
        ),
        logit_bias: Optional[Dict[str, int]] = Input(
            description="Modify the likelihood of specified tokens appearing in the completion.",
            default=None,
        ),
        max_tokens: Optional[int] = Input(
            description="The maximum number of [tokens](/tokenizer) to generate in the chat completion."
        ),
        n: Optional[int] = Input(
            description="How many chat completion choices to generate for each input message.",
            default=1,
            ge=1,
            le=128,
        ),
        presence_penalty: Optional[float] = Input(
            description="presence_penalty of model", default=0, ge=-1, le=2
        ),
        seed: Optional[int] = Input(
            description="If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same `seed` and parameters should return the same result.",
            default=None,
        ),
        stop: Optional[Union[str, List[str]]] = Input(
            description="Up to 4 sequences where the API will stop generating further tokens.",
            default=None,
        ),
        stream: Optional[bool] = Input(default=True),
        top_p: Optional[float] = Input(default=1, ge=0, le=1),
    ) -> AsyncConcatenateIterator[str]:
        """Run a single prediction on the model"""
        client = OpenAI(
            base_url="https://localhost:8001/rag/v1",
            api_key=diffbot_api_token.get_secret_value(),
        )

        completion = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            stream=stream,
            top_p=top_p,
        )
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                yield content
