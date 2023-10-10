import nltk
# For Spacy:
import spacy
from spacy import displacy
from collections import Counter
from pprint import pprint
# For custom ER:
import tkinter
import re

class StanfordNER:
  def __init__(self):
    self.get_stanford_ner_location()

  def get_stanford_ner_location(self):
    print("Provide (relative/absolute) path to stanford ner package.") 
    loc = input()
    print("... Running stanford for NER; this may take some time ...")

    self.stanford_ner_tagger = nltk.tag.StanfordNERTagger(loc+'/classifiers/english.all.3class.distsim.crf.ser.gz',
    loc+'/stanford-ner.jar')

  def ner(self,doc):
    sentences = nltk.sent_tokenize(doc)
    result = []
    for sent in sentences:
      words = nltk.word_tokenize(sent)
      tagged = self.stanford_ner_tagger.tag(words)
      result.append(tagged)
    return result

  def display(self,ner):
    print(ner)
    print("\n")

#________________________________________________________________________________________________________________________________   
class SpacyNER:
  def ner(self,doc):    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(doc)
    ner =  [(X.text, X.label_) for X in doc.ents]
    return self.ner_to_dict(ner)
  
  def ner_to_dict(self,ner):
    """
    Expects ner of the form list of tuples 
    """
    ner_dict = {}
    for tup in ner:
        ner_dict[tup[0]] = tup[1]
    return ner_dict
  
  def display(self,ner):
    print(ner)
    print("\n")

#________________________________________________________________________________________________________________________________
class NltkNER:
  def ner(self,doc):
    pos_tagged = self.assign_pos_tags(doc)
    #chunks = self.split_into_chunks(pos_tagged)
    result = []
    for sent in pos_tagged:
      result.append(nltk.ne_chunk(sent))
    return result

  def assign_pos_tags(self,doc):
    sentences = nltk.sent_tokenize(doc)
    words = [nltk.word_tokenize(sent) for sent in sentences]
    pos_tagged = [nltk.pos_tag(word) for word in words]
    return pos_tagged
  
  def split_into_chunks(self,sentences):
    # This rule says that an NP chunk should be formed whenever the chunker finds an optional determiner (DT) or 
    # possessive pronoun (PRP$) followed by any number of adjectives (JJ/JJR/JJS) and 
    # then any number of nouns (NN/NNS/NNP/NNPS) {dictator/NN Kim/NNP Jong/NNP Un/NNP}. 
    # Using this grammar, we create a chunk parser.
    grammar = "NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    chunks = []
    for sent in sentences:
        chunks.append(cp.parse(sent))
    return chunks

  def display(self,ner):
    print("\n\nTagged: \n\n")
    pprint(ner)
    print("\n\nTree: \n\n ")
    for leaves in ner:
        print(leaves)
        #leaves.draw()
    print("\n")





