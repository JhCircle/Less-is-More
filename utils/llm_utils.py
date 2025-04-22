from openai import OpenAI
import transformers
from utils.prompt import *

class Pipeline:
    def __init__(self, model_id, api_key=None, base_url='https://api.openai.com/v1/',prob=False,max_tokens=4096):
        self.api = False
        self.local = False
        self.base_url = base_url
        self.model_id = model_id
        self.max_tokens=max_tokens
        self.prob=prob

        if api_key is None:
            import torch
            self.local = True
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map='auto'
            )
        else:
            self.api = True
            self.api_key = api_key

    def get_respond(self, system_prompt,user_prompt, max_tokens=None, prob=False,json_format=False):
        global logprobs
        self.prob=prob
        if max_tokens:
            self.max_tokens=max_tokens
        if self.api:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            if json_format:
                completion = client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    logprobs=self.prob,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                          "name": "cot_response",
                          "strict": True,
                          "schema": {
                            "type": "object",
                            "properties": {
                              "cot_parsing": {
                                "type": "array",
                                "items": {
                                  "type": "object",
                                  "properties": {
                                    "statement": {
                                      "type": "string"
                                    },
                                    "evidence": {
                                      "type": "string"
                                    },
                                    "Verification": {
                                      "type": "string"
                                    }
                                  },
                                  "required": ["statement", "evidence", "Verification"],
                                  "additionalProperties": False
                                }
                              }
                            },
                            "required": ["cot_parsing"],
                            "additionalProperties": False
                          }
                        }
                    },
                    temperature=0.0
                )
                # print(completion)
                response = completion.choices[0].message.content
            else:
                completion =client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.max_tokens,
                    logprobs=self.prob
                )
                # print(completion)
                response = completion.choices[0].message.content


        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            prompt = self.pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot|>")
            ]

            outputs = self.pipeline(
                prompt,
                max_new_tokens=2048,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
                logprobs=self.prob
            )

            response = outputs[0]["generated_text"][len(prompt):]
        if self.prob:
            logprobs = [token.logprob for token in completion.choices[0].logprobs.content]
            import numpy as np
            import math
            probs = [math.exp(logprob) for logprob in logprobs]
            probs=np.mean(probs)
            return response, probs
        else:
            return response

