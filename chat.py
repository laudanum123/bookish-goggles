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
You are a helpful AI assistent. Give concise answers and your available tools are:{list_of_actions}.
IF YOU NEED A TOOL TO ANSWER A USER QUESTION ONLY OUTPUT THE FOLLOWING JSON OBJECT AND NOTHING ELSE:
{{"tool": "name_of_tool", "parameters": {{"function_parameter": "parameter_value"}}}}
\n
<|end_of_turn|>GPT4 Correct User:{question}<|end_of_turn|>
"""

template_test = """GPT4 Correct User: {question}<|end_of_turn|>GPT4 Correct Assistant:"""

# Load model and tokenizer

@cl.cache
def load_llama():
    model_name = "berkeley-nest/Starling-LM-7B-alpha"
    model_path = "../models"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        trust_remote_code=True,
        device_map="auto",
        max_length=600,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    llm = HuggingFacePipeline(
        pipeline=pipeline,
        model_kwargs={"temperature": 0.8}
    )
    return llm


model = load_llama()

@cl.on_chat_start
async def on_chat_start():
    global conversation_history
    conversation_history = []


    prompt = PromptTemplate(template=template, input_variables=["GPT4 Correct User"])
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    global conversation_history
    runnable = cl.user_session.get("runnable")

    conversation_history.append(f"GPT4 Correct User: {message.content}")

    conversation_string = "".join(conversation_history) + "GPT4 Correct Assistant: "

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": conversation_string, "list_of_actions": tools.list_of_tools},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    # Add the model's response to the conversation history
    conversation_history.append(f"GPT4 Correct Assistant: {chunk}")
    
    json = util.parse_json_from_string(msg.content)
    if json:
        await tools.navigate_to_url(json['parameters']['url'])

    await msg.send()