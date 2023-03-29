import nltk

nltk.download('averaged_perceptron_tagger')

text = nltk.word_tokenize("Is it possible distinguishing cats and dogs")
print(text)