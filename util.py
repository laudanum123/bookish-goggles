import json
from transformers import AutoTokenizer, TextIteratorStreamer
import transformers
from langchain.llms import HuggingFacePipeline
import chainlit as cl


def parse_json_from_string(string):
    start_index = string.find('{')
    end_index = string.rfind('}')
    if start_index == -1 or end_index == -1:
        return None
    print(string)
    print(start_index, end_index)
    json_string = string[start_index:end_index+1]
    #print(json_string)
    try:
        parsed_json = json.loads(json_string)
        return parsed_json
    except json.JSONDecodeError as e:
        print(e)
        return None

@cl.cache
def load_llama():
    """
    Load the LLaMA model and tokenizer.
    Returns the HuggingFacePipeline object.
    """
    model_name = "berkeley-nest/Starling-LM-7B-alpha"
    model_path = "../models"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        top_p=0.9,
        max_length=4000,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    llm = HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={"temperature": 1.2}
    )
    return llm

def get_token_count(text, model):
    """
    Calculate the token count of the given text.
    Args:
        text (str): Input text.
    Returns:
        int: Number of tokens.
    """
    return len(model.pipeline.tokenizer.encode(text))

def clean_special_tokens(text):
    """
    Remove special tokens from the given text.
    Args:
        text (str): Input text.
    Returns:
        str: Cleaned text.
    """
    return text.replace("<|end_of_turn|>", "")