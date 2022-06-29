import gradio as gr

from predictor import prepare_hyperparameters, model_and_tokenizer_initialize, inference

hyperparameters = prepare_hyperparameters()
BATCH_SIZE = 64

puncRec, tokenizer = model_and_tokenizer_initialize(hyperparameters)


def predict(text):
    return inference(text, puncRec, tokenizer, hyperparameters, BATCH_SIZE)


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
examples = [["Татарстан бүгенге көндә – Россиянең үзенчәлекле һәм мөстәкыйль төбәге. Без яңа "
             "эш алымнарын һәм заманча технологияләрне, беренче чиратта, нефть чыгару, "
             "нефть эшкәртү һәм нефть химиясе, машина төзү тармакларында һәм IТ-өлкәдә "
             "кулланышка кертү буенча алдынгы урынны биләп торабыз. Татарстанда гаять куәтле "
             "производстволар ачылды һәм алар уңышлы эшләп килә. Ил картасында яңа "
             "Иннополис шәһәре пәйда булды. Казан Россиянең спорт башкаласы дигән исемгә "
             "хаклы рәвештә лаек булуын раслады, тарихи-мәдәни мирасны торгызу өлкәсендә дә "
             "җитди адымнар ясалды."]]

# нәтиҗә
demo = gr.Interface(fn=predict,
                    inputs=textbox,
                    outputs="text",
                    title=title,
                    description=description,
                    article=article,
                    examples=examples,
                    )

if __name__ == '__main__':
    demo.launch()
