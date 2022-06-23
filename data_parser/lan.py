import spacy
from pprint import pprint
import sng_parser
import en_core_web_md
from sng_parser import database
# en = spacy.load('en_core_web_sm')
# parser = sng_parser.Parser('spacy', model='en')


sent = ""
if str(' bin ') in sent or str(' bin.') in sent or str(' bin,') in sent:
    nlp = spacy.load("en_core_web_lg")
else:
    nlp = spacy.load("en_core_web_md")
# nlp = spacy.load("en_core_web_lg")
doc = nlp(sent)
Doc = []
for token in doc:
    Doc.append(token)

if Doc[-1].dep_ != 'punct':
    Doc.append(".")
    sent = " ".join('%s' % id for id in Doc)

    doc = nlp(sent)
    Doc = []
    for token in doc:
        Doc.append(token)

entity_chunks = list()
for entity in doc.noun_chunks:
    if database.is_dir(entity.root.lemma_) == True and Doc[entity.end].dep_ != 'punct':
        continue
    if database.is_adj(entity.root.lemma_) == True and str(Doc[entity.root.i + 1]) == 'of':
        continue
    entity_chunks.append(entity)
sent = database.replace(sent, entity_chunks)
graph = sng_parser.parse(sent)

pprint(graph)