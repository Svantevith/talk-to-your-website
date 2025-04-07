from time import sleep
from httpx import ReadTimeout
from collections.abc import Generator
from langchain_ollama import ChatOllama


class LLM():
    def __init__(
        self,
        system_prompt: str,
        model_variant: str = "llama3.2:1b",
        temperature: float = 0.3,
        max_tokens: int = -2,
        keep_alive: int = 3600,
        timeout: int = 30
    ) -> None:
        """
        LLM objects are used to respond to the queries based on the given context, usually obtained from RAGs.

        Parameters
        ----------
            system_prompt : str
                Tells the system expected behaviour.
            model_variant : str
                Ollama model to use.
            temperature : float
                Creativity of the model.
            max_tokens : int
                Truncate tokens in the response or use -2 to ensure context is filled. 
            keep_alive : int
                Number of seconds the model stays loaded in memory after generating a response.
            timeout : int
                Timeout in seconds for the request stream.
        """
        self.system_prompt = system_prompt
        self.model_variant = model_variant
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens > 0 else -2
        self.keep_alive = keep_alive if keep_alive > 0 else 0
        self.timeout = timeout if timeout > 1 else 1

        # Llama 3 instruction-tuned models are fine-tuned and optimized for dialogue/chat use cases
        self.__model = ChatOllama(
            # Meta's Llama model variant
            model=self.model_variant,
            # Balance between deterministic and creative responses
            temperature=self.temperature,
            # Maximum number of tokens to predict when generating text
            num_predict=self.max_tokens,
            # How long model stays loaded in memory
            keep_alive=self.keep_alive,
            # Additional kwargs to the httpx Client
            client_kwargs={
                # Timeout configuration to use when sending request
                "timeout": self.timeout
            }
        )
                
    def stream_response(self, query: str, context: str)  -> Generator[str]:
        """
        Stream generated response.

        Parameters
        ----------
            prompt : str
                Prompt to respond.
            context : str
                Context to use by the model.

        Returns
        -------
            Generator[str]
                Yield generated content one token at a time. 
        """
        # Input to the LLM
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"Question: {query}\nContext: {context}"
            }
        ]

        try:
            # Invoke LLM and stream the response
            yield from self.__model.stream(messages)
        
        except ReadTimeout:
            # Stream exception message
            for word in "Sorry, but I could not fulfill your request within the specified timeout threshold.".split():
                yield word + " "
                sleep(0.02)
