#!/usr/bin/env python3

from difflib import Differ

import gradio as gr


def diff_texts(text1, text2):
    d = Differ()
    return [
        (token[2:], token[0] if token[0] != " " else None)
        for token in d.compare(text1, text2)
    ]


demo = gr.Interface(
    diff_texts,
    [
        gr.Textbox(
            label="Initial text",
            lines=3,
            value="The quick brown fox jumped over the lazy dogs.",
        ),
        gr.Textbox(
            label="Text to compare",
            lines=3,
            value="The fast brown fox jumps over lazy dogs.",
        ),
    ],
    gr.HighlightedText(
        label="Diff",
        combine_adjacent=True,
    ).style(color_map={"+": "red", "-": "green"}),
)

if __name__ == "__main__":
    demo.launch(
        server_name='0.0.0.0',
        # server_port=8080,
        show_api=False,
        inbrowser=False
    )
