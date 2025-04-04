from langchain_ollama import ChatOllama


class LLM():
    def __init__(
        self,
        system_prompt: str,
        model_variant: str = "llama3.2:1b",
        temperature: float = 0.3,
        max_tokens: int = -2
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
        """
        self.system_prompt = system_prompt
        self.model_variant = model_variant
        self.temperature = temperature
        self.max_tokens = max_tokens if max_tokens >= 0 else -2

        # Llama 3 instruction-tuned models are fine-tuned and optimized for dialogue/chat use cases.
        self.__model = ChatOllama(
            # Meta's Llama 3.2 goes small with 1B and 3B models. 1B offers multilingual knowledge retrieval.
            model=self.model_variant,
            temperature=self.temperature,
            # Maximum number of tokens to predict when generating text.
            num_predict=self.max_tokens
        )

    def respond(self, query: str, context: str) -> str:
        """
        Respond to the question.

        Parameters
        ----------
            prompt : str
                Prompt to respond.
            context : str
                Context to use by the model.

        Returns
        -------
            str
                Generated content. 
        """
        # Input to the LLM.
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

        # Invoke LLM and get the response.
        print("=== Generating response ===")
        response = self.__model.invoke(messages)

        print("=== Response ready ===")
        return response.content
