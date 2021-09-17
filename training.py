import spacy
from spacy import displacy
nlp=spacy.load('en_core_web_sm')

def tmf632recognition(text):
    
    ausgabedict = {}
    
    doc = nlp(text)
    for ent in doc.ents:
        ausgabedict[ent.text] = ent.label_
    
    return ausgabedict 

ner=nlp.get_pipe("ner")


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

import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.tokens import Doc
from spacy.training import Example

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


model = None
output_dir = "C:\\Users\\Olive\\Desktop\\TMF632 NER deployment"

if output_dir is not None:
    output_dir = "C:\\Users\\Olive\\Desktop\\Untitled Folder"
    nlp.to_disk(output_dir)
    print("Saved model to",output_dir)


