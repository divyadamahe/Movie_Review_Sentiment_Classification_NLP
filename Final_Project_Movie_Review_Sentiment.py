#Importing relevant libraries and packages

import os
import sys
import random
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

#Putting the stopwords in an iterable list

nltkstopwords = nltk.corpus.stopwords.words('english')

#Define a function to remove punctuations and stop words
def pre_processing_documents(document):
    word_list = re.split('\s+', document.lower())
    punctuation = re.compile(r'[-.?\%@,":;()|0-9]')
    word_list = [punctuation.sub("", word) for word in word_list]
    final_word_list = []
    for word in word_list:
        if word not in nltkstopwords:
            final_word_list.append(word)
    line = " ".join(final_word_list)
    return line

#Define a function to only get words with length greater than 3
def get_words_from_phrasedocs(docs):
    all_words = []
    for (words, sentiment) in docs:
        possible_words = [x for x in words if len(x)>=3]
        all_words.extend(possible_words)
    return all_words

#Getting all the words from the document
def get_words_from_phrasedocs_normal(docs):
    all_words = []
    for (words, sentiment) in docs:
        all_words.extend(words)
    return all_words

#Getting words from the test document
def get_words_from_test(line):
    all_words = []
    for id, words in line:
        all_words.extend(words)
    return all_words

#Define a function that returns the frequency distribution as features
def get_word_features(wordList):
    wordList = nltk.FreqDist(wordList)
    word_features = [ w for (w,c) in wordList.most_common(200)]
    return word_features

#Define a function that returns word features 
def normal_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

#Define a function that returns Part of Speech features
def POS_features(document, word_features):
    	document_words = set(document)
    	tagged_words = nltk.pos_tag(document)
    	features = {}
    	for word in word_features:
   	     features['contains({})'.format(word)] = (word in document_words)
    	numNoun = 0
    	numVerb = 0
    	numAdj = 0
    	numAdverb = 0
    	for (word, tag) in tagged_words:
    	    if tag.startswith('N'): numNoun += 1
    	    if tag.startswith('V'): numVerb += 1
    	    if tag.startswith('J'): numAdj += 1
    	    if tag.startswith('R'): numAdverb += 1
    	features['nouns'] = numNoun
    	features['verbs'] = numVerb
    	features['adjectives'] = numAdj
    	features['adverbs'] = numAdverb
    	return features
    
#Define a list with negative words and a function that returns negation features    
negationwords = ['no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 'rather',\
                 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor']

def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = False
        features['contains(NOT{})'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['contains(NOT{})'.format(document[i])] = (document[i] in word_features)
        else:
            features['contains({})'.format(word)] = (word in word_features)
    return features

#Define a function that returns sentiment lexicon as features
import sentiment_read_subjectivity

SL = sentiment_read_subjectivity.readSubjectivity("subjclueslen1-HLTEMNLP05.tff")

def SL_features(document, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos +=1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)
    return features

#Define a function that calculates precision, recall and f1 scores
def eval_measures(gold, predicted):
    # get a list of labels
    labels = list(set(gold))
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []
    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

#Open the training data set
f = open('./train.tsv', 'r')
# loop over lines in the file and use the first limit of them
phrasedata = []
for line in f:
  # ignore the first line starting with Phrase and read all lines
  if (not line.startswith('Phrase')):
    # remove final end of line character
    line = line.strip()
    # each line has 4 items separated by tabs
    # ignore the phrase and sentence ids, and keep the phrase and sentiment
    phrasedata.append(line.split('\t')[2:4])

# pick a random sample of length limit because of phrase overlapping sequences
random.shuffle(phrasedata)
phraselist = phrasedata[:50000]

print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')

#for phrase in phraselist[:10]:
 # print (phrase)

# create list of phrase documents as (list of words, label)
phrasedocs = []
phrasedocs_without = [] 
# add all the phrases
for phrase in phraselist:
    #without pre processing
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs_without.append((tokens, int(phrase[1])))
    
    #with preprocessing
    tokenizer = RegexpTokenizer(r'\w+')
    phrase[0] = pre_processing_documents(phrase[0])
    tokens = tokenizer.tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))

# # possibly filter tokens
normaltokens = get_words_from_phrasedocs_normal(phrasedocs_without)
preprocessed = get_words_from_phrasedocs(phrasedocs)

word_features = get_word_features(normaltokens)
feature_sets_without_preprocessing = [(normal_features(d, word_features), s) for (d,s) in phrasedocs_without]

#Naive Bayes without pre processing
train_set, test_set = feature_sets_without_preprocessing[1000:], feature_sets_without_preprocessing[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("The accuracy of Naive Bayes Classifier without any pre-processing is: ", nltk.classify.accuracy(classifier, test_set))

print("Top 25 most important features are: \n", classifier.most_informative_features(25))

goldlist = []
predictedlist = []
for (features, label) in test_set:
     	goldlist.append(label)
     	predictedlist.append(classifier.classify(features))
        
#print(goldlist[:30])
#print(predictedlist[:30])


eval_measures(goldlist, predictedlist)

#Naive Bayes with pre processing
train_set, test_set = feature_sets_without_preprocessing[1000:], feature_sets_without_preprocessing[:1000]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print("The accuracy of Naive Bayes Classifier with the pre-processing is: ", nltk.classify.accuracy(classifier, test_set))

print("Top 25 most important features are: \n", classifier.most_informative_features(25))

goldlist = []
predictedlist = []
for (features, label) in test_set:
     	goldlist.append(label)
     	predictedlist.append(classifier.classify(features))

eval_measures(goldlist, predictedlist)

#POS Features

POS_featuresets = [(POS_features(d, word_features), c) for (d, c) in phrasedocs]
# number of features for document 0
len(POS_featuresets[0][0].keys())

train_set, test_set = POS_featuresets[1000:], POS_featuresets[:1000]
classifier1 = nltk.NaiveBayesClassifier.train(train_set)
print("The accuracy of Naive Bayes Classifier with the POS features is: ", nltk.classify.accuracy(classifier1, test_set))

print("Top 25 most important features are: \n", classifier1.most_informative_features(25))

goldlist1 = []
predictedlist1 = []
for (features, label) in test_set:
     	goldlist1.append(label)
     	predictedlist1.append(classifier1.classify(features))
        
#print(goldlist1[:30])
#print(predictedlist1[:30])


# eval_measures(goldlist1, predictedlist1)

# Negation features

NOT_featuresets = [(NOT_features(d, word_features, negationwords), c) for (d, c) in phrasedocs]

train_set, test_set = NOT_featuresets[200:], NOT_featuresets[:200]
classifier2 = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier2, test_set))

classifier2.show_most_informative_features(30)

goldlist2 = []
predictedlist2 = []
for (features, label) in test_set:
     	goldlist2.append(label)
     	predictedlist2.append(classifier2.classify(features))
        
print(goldlist2[:30])
print(predictedlist2[:30])


eval_measures(goldlist2, predictedlist2)

#Sentiment lexicon features 

SL_featuresets = [(SL_features(d, SL), c) for (d,c) in phrasedocs]

train_set, test_set = SL_featuresets[1000:], SL_featuresets[:1000]
classifier3 = nltk.NaiveBayesClassifier.train(train_set)
print("The accuracy of Naive Bayes Classifier with the Sentiment Lexicon features is: ", nltk.classify.accuracy(classifier3, test_set))

print("Top 25 most important features are: \n", classifier3.most_informative_features(25))

#print(nltk.classify.accuracy(classifier3, test_set))

#classifier3.show_most_informative_features(30)

goldlist3 = []
predictedlist3 = []
for (features, label) in test_set:
     	goldlist3.append(label)
     	predictedlist3.append(classifier3.classify(features))
        
#print(goldlist3[:30])
#print(predictedlist3[:30])


eval_measures(goldlist3, predictedlist3)