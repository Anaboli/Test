import spacy
from spacy import displacy
import en_core_web_sm
from spacy import en_core_web_sm
nlp=spacy.load('en_core_web_sm')

TRAIN_DATA = [
              ("Männlich ist ein Geschlecht",{"entities": [(0,8, "Gender")]}),
              ("Weiblich ist ein Geschlecht",{"entities": [(0,8, "Gender")]}),
              ("weiblich ist ein Geschlecht",{"entities": [(0,8, "Gender")]}),
              ("männlich ist ein Geschlecht",{"entities": [(0,8, "Gender")]}),
              ("Mänlich ist ein Geschlecht",{"entities": [(0,7, "Gender")]}),
              ("Male is a gender",{"entities": [(0,4, "Gender")]}),
              ("male is the gender",{"entities": [(0,4, "Gender")]}),
              ("Female is a gender",{"entities": [(0,6, "Gender")]}),
              ("female is a gender",{"entities": [(0,6, "Gender")]}),
              ("Männlich ist ebenfalls ein Geschlecht",{"entities": [(0,8, "Gender")]}),
              ("Oliver Kibke ist männlich und wohnt in Lohmar 53797 am See", {"entities": [(0,12, "PERSON"),(17,25, "GENDER"),(39,45,"LOC"),(46,51, "postCode")]})
              ]

for _, annotations in TRAIN_DATA:
  for ent in annotations.get("entities"):
    print(_)
    ner.add_label(ent[2])
    print(ent[2])

pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

with nlp.disable_pipes(*unaffected_pipes):
    
    # Training for 30 iterations on the dataset
  for iteration in range(15):
    
    # shuufling examples  before every iteration
    random.shuffle(TRAIN_DATA)
    losses = {}
    
    for batch in minibatch(TRAIN_DATA,size=8):
        for text, annotations in batch:
          doc = nlp.make_doc(text)
          example = Example.from_dict(doc,annotations)
          nlp.update([example], drop=0.5,losses=losses)




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
