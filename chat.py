from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, TextIteratorStreamer
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.config import RunnableConfig

import transformers

import chainlit as cl


template = """
GPT4 Correct User: {question}<|end_of_turn|>GPT4 Correct Assistant:
"""

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
        max_length=4000,
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

    # Update conversation history with the user's message
    conversation_history.append(f"GPT4 Correct User: {message.content}")

    # Generate the conversation string including the history
    conversation_string = "".join(conversation_history) + "GPT4 Correct Assistant: "

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": conversation_string},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    # Add the model's response to the conversation history
    conversation_history.append(f"GPT4 Correct Assistant: {chunk}")

    await msg.send()