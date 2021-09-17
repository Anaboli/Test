import gradio as gr
import spacy
nlp = spacy.load("g/Anaboli/Test")

def tmf632recognition(text):
    
    ausgabedict = {}
    
    doc = nlp(text)
    for ent in doc.ents:
        ausgabedict[ent.text] = ent.label_
    
    return ausgabedict

def test(text):
    
    return "Hello" + "Text"


output_text = gr.outputs.Textbox(label = "Input mapped to TMF632 Variables")
input_text = gr.inputs.Textbox(label = "Data for AI to detect TMF632 Variables")

gr.Interface(tmf632recognition,input_text, output_text).launch()
