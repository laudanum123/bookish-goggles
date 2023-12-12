from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
import tools
from util import load_llama, get_token_count, clean_special_tokens, parse_json_from_string

import chainlit as cl


template = """
GPT4 Correct User: You are a helpful AI assistent.
<|end_of_turn|>
{question}<|end_of_turn|>GPT4 Correct Assistant:
"""


# Load model and tokenizer


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
    template_token_count = get_token_count(filled_template, model)
    
    # Remove parts of the conversation history until it fits into the model's max length
    while template_token_count > (max_length-(max_length/2)):
        print(conversation_history)
        conversation_history.pop(0)
        conversation_string = "<|end_of_turn|>".join(conversation_history)
        filled_template = template.format(list_of_actions=tools.list_of_tools, question=conversation_string)
        template_token_count = get_token_count(filled_template, model)

    print(filled_template)

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": conversation_string, "list_of_actions": tools.list_of_tools},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)
    
    json = parse_json_from_string(msg.content)
    if json:
        await tools.suche_artikel(json['parameter'])

    await msg.send()

    # Update conversation history with the model's response.
    conversation_history.append(f"GPT4 Correct Assistant: {clean_special_tokens(msg.content)}")