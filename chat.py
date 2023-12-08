from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextIteratorStreamer
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
import tools
import util

import transformers

import chainlit as cl


template = """
GPT4 Correct User: You are a helpful AI assistent.
<|end_of_turn|>
{question}<|end_of_turn|>GPT4 Correct Assistant:
"""


# Load model and tokenizer
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
        max_length=8000,
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

def get_token_count(text):
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

model = load_llama()

@cl.on_chat_start
async def on_chat_start():
    """
    Initializes the chat session.
    Sets up the conversation history and runnable configuration.
    """
    global conversation_history
    conversation_history = []


    prompt = PromptTemplate(template=template, input_variables=["GPT4 Correct User"])
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    """
    Handles incoming messages and generates responses using the model.
    Args:
        message (cl.Message): Incoming message from the user.
    """
    global conversation_history
    runnable = cl.user_session.get("runnable")

    conversation_history.append(f"GPT4 Correct User: {clean_special_tokens(message.content)}")

    conversation_string = "<|end_of_turn|>".join(conversation_history)


    max_length = model.pipeline._forward_params["max_length"]
    
    # Fill the template and get number of tokens
    filled_template = template.format(list_of_actions=tools.list_of_tools, question=conversation_string)
    template_token_count = get_token_count(filled_template)
    
    # Remove parts of the conversation history until it fits into the model's max length
    while template_token_count > (max_length-(max_length/2)):
        print(conversation_history)
        conversation_history.pop(0)
        conversation_string = "<|end_of_turn|>".join(conversation_history)
        filled_template = template.format(list_of_actions=tools.list_of_tools, question=conversation_string)
        template_token_count = get_token_count(filled_template)

    print(filled_template)

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": conversation_string, "list_of_actions": tools.list_of_tools},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    
    json = util.parse_json_from_string(msg.content)
    if json:
        await tools.suche_artikel(json['parameter'])

    await msg.send()

    # Update conversation history with the model's response.
    conversation_history.append(f"GPT4 Correct Assistant: {clean_special_tokens(msg.content)}")