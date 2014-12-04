#!/usr/bin/python

import networkx as nx

# user imports
from image_search import search_image

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def textrank(document):
  text =u""
  with open(document) as f:
    text = f.read()
  sentence_tokenizer = PunktSentenceTokenizer()
  sentences = sentence_tokenizer.tokenize(text)

  bow_matrix = CountVectorizer().fit_transform(sentences)
  normalized = TfidfTransformer().fit_transform(bow_matrix)

  similarity_graph = normalized * normalized.T

  nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
  scores = nx.pagerank(nx_graph)
  return sorted(((scores[i],s) for i,s in enumerate(sentences)),
                  reverse=True)

def get_image(sentence):
  words = nltk.word_tokenize(sentence)
  tags = nltk.pos_tag(words)
  print(tags)
  nouns = [word for word, tag in tags if tag == 'NNP' or tag == 'NNS']
  print(nouns)
  search_image(nouns)

if __name__ == "__main__":
  sentences = textrank("test.txt")
  for s in sentences:
    print(s)
  get_image(sentences[0][1])
