#!/usr/bin/env python3

import os

import gradio
from dotenv import load_dotenv
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

load_dotenv()

DOCUMENTS_FOLDER = "/app/docs"
INDEX_FILE = "/app/index.json"


def index_maker():
    print('\nPreparing index. Please, take a cup of coffee.\n')

    documents = SimpleDirectoryReader(DOCUMENTS_FOLDER).load_data()

    index = GPTSimpleVectorIndex.from_documents(documents)
    index.save_to_disk(INDEX_FILE)

    return True


def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk(INDEX_FILE)
    response = index.query(input_text)

    return response.response


gradio_interface = gradio.Interface(
    chatbot,
    gradio.Textbox(label="Question:", lines=3),
    gradio.Textbox(label="Answer:", lines=15),
    title="Custom AI Chatbot",
)


if __name__ == "__main__":
    if os.path.isfile(INDEX_FILE) is not True:
        index_maker()

    gradio_interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        show_api=False,
        inbrowser=False
    )
