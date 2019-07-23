from nltk import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
import nltk

example = "Hello Mr. Holmes. How are you doing? The weather is nice Holmes and Python is amazing. I hope you like it too!"
sen_list = sent_tokenize(example)
sen = sen_list[2]
print(sen)
stop_words = set(stopwords.words('english'))
'''words = word_tokenize(sen)
filtered_words = []
for w in words:
    if w not in stop_words:                 tokenizing
        filtered_words.append(w)
print(filtered_words)
'''
tokenize = PunktSentenceTokenizer(sen)
tokenized = tokenize.tokenize(sen)  # Speech tagging
print(tokenized)
for i in tokenized:
    words = word_tokenize(i)
    tagged = nltk.pos_tag(words)
    # Chunking
    '''
    using regex  here . means select all characters
    ? means atleast 1 repetation.. for further info see tutorial on pythonprogrammong.net
    '''
    chunkgram = r"""Chunk: {<RB.?>*<VB.?>*<NNP><NN>?} """  # RB,VB,NNp etc are tags like VB=verb... what we are doing here is selecting certain type of words in chunk
    chunkparser = nltk.RegexpParser(chunkgram)
    chunked = chunkparser.parse(tagged)
    print(chunked)
