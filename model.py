import os
import pdb
import time
from openai import OpenAI
import tiktoken
import google.generativeai as palm
import google.api_core.exceptions
from typing import Set, Tuple
from prompt import StreamPrompt
from colorama import Fore, Style
from argparse import Namespace
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAIError
import json

  # Importing error module for handling exceptions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Model(object):
    api_interval = 0.00 # seconds
    cost_per_1000tokens = 0.02
    gpt_models = {"text-davinci-003", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo"}
    palm_models = {"models/text-bison-001"}
    hf_models = {"mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3-8B-Instruct"}

    def __init__(self, config: Namespace):
        self._config = config
        if self._config.model in Model.gpt_models:
            self._tokenizer = tiktoken.encoding_for_model(self._config.model)
            self._original_max_tokens = self._config.max_tokens
            self._original_temperature = self._config.temperature
            api_key = os.getenv("OPENAI_API_KEY")
            self.openai = OpenAI(
                api_key = api_key
            )
        if self._config.model in Model.hf_models:
            self.tokenizer = AutoTokenizer.from_pretrained(self._config.model)
            self.model = AutoModelForCausalLM.from_pretrained(self._config.model).to(device) 
            self._original_max_tokens = self._config.max_tokens
            self._original_temperature = self._config.temperature
        else:
            palm.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def retry_with_exponential_backoff(
        func,
        initial_delay: float = 0.5,
        exponential_base: float = 2,
        max_retries: int = 20,
        errors: tuple = (
            OpenAIError,
            google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.ServiceUnavailable, google.api_core.exceptions.GoogleAPIError,
            json.JSONDecodeError 
        ),
    ):
        """Retry a function with exponential backoff."""
    
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay
            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                # Retry on specific errors
                except errors as e:
                    # Increment retries
                    num_retries += 1
                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            Fore.RED + f"Maximum number of retries ({max_retries}) exceeded." + Style.RESET_ALL
                        )
                    # Increment the delay
                    delay *= exponential_base
                    # Sleep for the delay
                    print(Fore.YELLOW + f"Error encountered. Retry ({num_retries}) after {delay} seconds..." + Style.RESET_ALL)
                    time.sleep(delay)
                # Raise exceptions for any errors not specified
                except Exception as e:
                    #pdb.set_trace()
                    raise e
    
        return wrapper

    @retry_with_exponential_backoff
    def complete(self, prompt: str, label_set: Set[str] = set(), temperature: float = None, is_json=False) -> Tuple[str, str]:
        time.sleep(Model.api_interval)
        if self._config.model in Model.gpt_models:
            if(is_json):
                return self.gpt_complete_json(prompt, label_set, temperature)
            else:
                return self.gpt_complete(prompt, label_set, temperature)
        elif self._config.model in Model.palm_models:
            return self.palm_complete(prompt, label_set, temperature)
        elif self._config.model in Model.hf_models:
            return self.hf_complete(prompt, label_set, temperature)
        else:
            raise Exception(Fore.RED + f"Model {self._config.model} not supported." + Style.RESET_ALL)

    def gpt_complete(self, prompt: str, label_set: Set[str] = set(), temperature: float = None) -> Tuple[str, str]:
        params = vars(self._config)
        if label_set:
            max_tokens = 0
            for label in label_set:
                tokens = label
                if '(' == tokens[0]:
                    tokens = tokens[1:] # remove parentheses
                    if prompt[-1] != '(':
                        prompt += " ("
                token_ids = self._tokenizer.encode(tokens)
                max_tokens = max(max_tokens, len(token_ids))
            if max_tokens > 0:
                # params["logit_bias"] = logit_bias
                if prompt[-1] != '(':
                    max_tokens += 1 # +1 for whitespace
                params["max_tokens"] = max_tokens
        else:
            params["max_tokens"] = self._original_max_tokens
        
        if temperature is not None:
            params["temperature"] = temperature
        else:
            params["temperature"] = self._original_temperature
        print(Fore.GREEN + f" (Predicting with temperature: {params['temperature']}, max_tokens: {params['max_tokens']}) " + Style.RESET_ALL, end='')
        return prompt, self.openai.completions.create(prompt=prompt, **params).choices[0].text.strip()

    def gpt_complete_json(self, prompt: str, label_set: Set[str] = set(), temperature: float = None) -> Tuple[str, str]:
        params = vars(self._config)
        #if label_set:
        #    max_tokens = 0
        #    for label in label_set:
        #        tokens = label
        #        #if '(' == tokens[0]:
        #        #    tokens = tokens[1:] # remove parentheses
        #        #    if prompt[-1] != '(':
        #        #        prompt += " ("
        #        token_ids = self._tokenizer.encode(tokens)
        #        max_tokens = max(max_tokens, len(token_ids))
        #    if max_tokens > 0:
        #        # params["logit_bias"] = logit_bias
        #        if prompt[-1] != '(':
        #            max_tokens += 1 # +1 for whitespace
        #        params["max_tokens"] = max_tokens
        #else:
        params["max_tokens"] = self._original_max_tokens
        
        if temperature is not None:
            params["temperature"] = temperature
        else:
            params["temperature"] = self._original_temperature
        print(Fore.GREEN + f" (Predicting with temperature: {params['temperature']}, max_tokens: {params['max_tokens']}) " + Style.RESET_ALL, end='')
        return prompt, json.loads(self.openai.completions.create(prompt=prompt, **params).choices[0].text.strip())

    def gpt_chat_complete(self, prompt: str, label_set: Set[str] = set(), temperature: float = None) -> Tuple[str, str]:
        params = vars(self._config)
        #if label_set:
        #    max_tokens = 0
        #    for label in label_set:
        #        tokens = label
        #        #if '(' == tokens[0]:
        #        #    tokens = tokens[1:] # remove parentheses
        #        #    if prompt[-1] != '(':
        #        #        prompt += " ("
        #        token_ids = self._tokenizer.encode(tokens)
        #        max_tokens = max(max_tokens, len(token_ids))
        #    if max_tokens > 0:
        #        # params["logit_bias"] = logit_bias
        #        if prompt[-1] != '(':
        #            max_tokens += 1 # +1 for whitespace
        #        params["max_tokens"] = max_tokens
        #else:
        params["max_tokens"] = self._original_max_tokens
        
        if temperature is not None:
            params["temperature"] = temperature
        else:
            params["temperature"] = self._original_temperature
        print(Fore.GREEN + f" (Predicting with temperature: {params['temperature']}, max_tokens: {params['max_tokens']}) " + Style.RESET_ALL, end='')
        return prompt, self.openai.chat.completions.create(messages=prompt, **params).choices[0].message.content.strip()

    def hf_complete(self, prompt, label_set: Set[str] = set(), temperature=0.7, max_tokens=256):
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device) 
        
        # Generate text using the model
        with torch.no_grad():
            output = self.model.generate(
                inputs.input_ids,
                max_length=256,
                temperature=0,
                do_sample=False,
                top_p=0.8,  # Optionally, adjust top_p for nucleus sampling
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode the output into text
        completion = self.tokenizer.decode(output[0], skip_special_tokens=True)
        #pdb.set_trace()
        # Return the generated completion
        return prompt, completion[len(prompt):].strip()

    def palm_complete(self, prompt: str, label_set: Set[str] = set(), temperature: float = None) -> Tuple[str, str]:
        params = {
            "temperature": self._config.temperature if (temperature is None) else temperature,
            "candidate_count": 1,
            "top_p": self._config.top_p,
            "max_output_tokens": self._config.max_tokens,
        }
        model = palm.GenerativeModel(self._config.model, generation_config=params)
        #pdb.set_trace()
        response = model.generate_content(
            prompt
        )
        print(Fore.GREEN + f" (Predicting with temperature: {params['temperature']}, max_tokens: {params['max_output_tokens']}) " + Style.RESET_ALL)
        if response.text is None:
            print(Fore.RED + f"Failed to generated new instance." + Style.RESET_ALL)
            raise ValueError("Failed to generated new instance.")
        return prompt, response.text

    def set_api_key(self, key: str) -> None:
        self.openai.api_key = key

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

# for unit testing
if __name__ == "__main__":
    # model
    config = Namespace(
        model="models/text-bison-001",
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
    )
    model_api = Model(config)
    
    # tasks
    from task import TaskGenerator
    
    task_gen = TaskGenerator(
        task_input_path="./bbh/BIG-Bench-Hard/bbh/",
        task_desc_path="./bbh/bbh_task_description.json",
        batch_size=1
    )
    task_names = ["boolean_expressions", "causal_judgement", "date_understanding", "formal_fallacies", "sports_understanding"]
    
    # prompt
    for task_name in task_names:
        task = task_gen.get_task(task_name)
        task_inputs = task.get_new_inputs()
        stream_prompt = StreamPrompt(
            task_desc=task.task_desc,
            inputs=task_inputs,
            num_demos=3,
            shots=[]
        )
        
        pred_prompt = stream_prompt.gen_prediction()
        print(f"Generating prediction from label set: {task.label_set} ->")
        pred_prompt, res_text = model_api.complete(pred_prompt, task.label_set)
        full_text = pred_prompt + res_text
        print(f"full text: {full_text}")
