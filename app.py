import gradio as gr


def greet(name):
    return name + ' лох'


title = " PuncRec - Punctuation and capitalization recovery for tatar language"
description = """
This model trained using custom dataset of public Tatarstan government documents.
It based on tatar BERT. PuncRec recovers punctuation and capitalization of ASR output for tatar (tt). 
"""

article = "article link"

textbox = gr.Textbox(
    label="Языгыз, текст башка билгеләре препинания һәм баш хәрефләр",
    placeholder="Text",
    lines=2,
)
examples = [["What are you doing?"], ["Where should we time travel to?"]]

# нәтиҗә
demo = gr.Interface(fn=greet,
                    inputs=textbox,
                    outputs="text",
                    title=title,
                    description=description,
                    article=article,
                    examples=examples,
                    )

if __name__ == '__main__':
    demo.launch(share=True)
