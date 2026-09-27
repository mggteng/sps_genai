"""
Word embedding model for Assignment 1.

The functions below are moved from the Module 2 lecture notebook
(Module_2_Practical_3_Word_Embeddings): calculate_embedding and
calculate_similarity, using spaCy's large English model (300-dimensional
word vectors). EmbeddingModel wraps them so main.py can load the spaCy
model once at startup and reuse it for every request.
"""

import spacy


class EmbeddingModel:
    def __init__(self, model_name="en_core_web_lg"):
        self.nlp = spacy.load(model_name)

    def calculate_embedding(self, input_word):
        word = self.nlp(input_word)
        return word.vector

    def calculate_similarity(self, word1, word2):
        return self.nlp(word1).similarity(self.nlp(word2))
