import gradio as gr

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

gr.Interface(test,input_text, output_text).launch()
