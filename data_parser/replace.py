import spacy
import en_core_web_md
import torch
nlp = spacy.load("en_core_web_md")
#doc = nlp("A green woman is playing the piano in the room.")
doc = nlp("this is an office chair. it is at the double cubicle pulled out from the desk.")
doc1 = []
for token in doc:
    #print(token.i)
    doc1.append(token)
#print(doc1[1])
#print(doc.noun_chunks)
a = []
for entity in doc.noun_chunks:
    #a = entity.root
    a.append(entity)
#print(a)
for entity in doc.noun_chunks:
    if entity.root.lemma_ == '-PRON-':
         #print(entity.root.i)
         doc1[entity.root.i] = a[0]
sent = " ".join('%s' %id for id in doc1)
print(sent)               
#print(token)
                #print(entity)
                #print(entity.root.head.i)
    #print('--', a)
#print(doc)
    #for x in entity.root.children:
         #print('--', x.dep_)
#a = tuple(doc.text)
#print('---', a)
#for token in doc:
    #print('-',token)
    #print('--',token.head)
    #print('---',token.head.head)
    #print('----',token.head.head.pos_)
