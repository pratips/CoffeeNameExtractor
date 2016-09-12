# -*- coding: utf-8 -*-

import nltk 
import sys
import pycrfsuite

# list_sentences = ['I would like to order two hot Latte with less foam picking it up in 10 mins',
#                   'I would like to order two hot latte and a cold coffee and one cappuccino with extra sugar , will pickup in half an hour'
#                   ]
# text=nltk.word_tokenize("We are going out.Just you and me.")
# print nltk.pos_tag(text)
# [('We', 'PRP'), ('are', 'VBP'), ('going', 'VBG'), ('out.Just', 'JJ'),
#  ('you', 'PRP'), ('and', 'CC'), ('me', 'PRP'), ('.', '.')]

part3_cnames_small = []

def generate_dummy_sentences():
    part1 = ['I would like to order']
    part2_CD = ['one', 'two', 'three', 'five', 'eight', 'nine', '']
    part2_Adj = ['hot', 'cold', '']
    part3_cnames = ['Caffè Americano', 'Café Cubano', 'Caffè crema', 'Cafe Zorro', 'Doppio', 'Espresso Romano',
                    'Guillermo', 'Ristretto', 'Espresso with milk', 'Antoccino', 'Breve', 'Café bombón',
                    'Cappuccino', 'Cortado', 'Latte', 'Latte macchiato', 'Espressino', 'Flat white',
                    'Galão', 'Caffè gommosa', 'Macchiato', 'Wiener or Viennese melange', 'Coffee with milk',
                    'Café au lait', 'Ca phe sua da', 'Egg coffee', 'Eggnog latte', 'Eiskaffee', 'Kopi susu',
                    'White coffee', 'White coffee (England)', 'Coffee or espresso with whipped cream', 
                    'Vienna coffee', 'Espresso con panna', 'Black tie', 'Chai latte', 'Red tie',
                    'Yuanyang', 'Liqueur coffee', 'Irish coffee', 'Rüdesheimer Kaffee', 'Pharisäer'
                    'Carajillo']
    part4 =  ['with less foam picking it up in 10 mins', '']
    
    global part3_cnames_small
    
    for ele in part3_cnames:
        if len(ele.split())<=1:
            part3_cnames_small.append(ele)
    
    list_sentences = []        
    for ele1 in part1:
        for ele2 in part2_CD:
            for ele_adj in part2_Adj:
                for ele3 in part3_cnames_small:
                    for ele4 in part4:
                        list_sentences.append(" ".join([ele1,ele2,ele_adj,ele3,ele4]))
    #                     print list_sentences
    return list_sentences
                    
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent] 

def tag_auto(list_sentences):
    train_sents = []
    for sentence in list_sentences:
        text_tokenized = nltk.word_tokenize(sentence)
        train_sents.append(nltk.pos_tag(text_tokenized))
    return train_sents
        
def tag_manu(train_sents):
    train_sents_new = []
    for ele in train_sents:
#         print ele
        ele_new = []
        for tuple in ele:
#             print tuple
            if tuple[0] in part3_cnames_small:
#                 print tuple
                new_tuple = (tuple[0], tuple[1], 'B-COFFEE')
            else:
                new_tuple = (tuple[0], tuple[1], 'O')
            ele_new.append(new_tuple)    
#         print ele_new
        train_sents_new.append(ele_new)
    return train_sents_new

def train_test_model(train_sents):
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]
    
    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    
    
    trainer.set_params({
        'c1': 1.0,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier
    
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    
    trainer.train('coffee_tagger.crfsuite')
    
    tagger = pycrfsuite.Tagger()
    tagger.open('coffee_tagger.crfsuite')
    
    example_sent = train_sents[0]
    print example_sent
    print(' '.join(sent2tokens(example_sent)))#, end='\n\n')
    
    print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    print("Correct:  ", ' '.join(sent2labels(example_sent)))

def test_model(sentence):
    text_tokenized = nltk.word_tokenize(sentence)
    tagger = pycrfsuite.Tagger()
    tagger.open('coffee_tagger.crfsuite')
    print("Predicted:", ' '.join(tagger.tag(sent2features(nltk.pos_tag(text_tokenized)))))
    tag_list = tagger.tag(sent2features(nltk.pos_tag(text_tokenized)))
    print nltk.ne_chunk(nltk.pos_tag(text_tokenized))
    if 'B-COFFEE' in tag_list:
        
        return "Coffee : {0}".format(text_tokenized[tag_list.index('B-COFFEE')])
    else:
        return "NOT FOUND"

if __name__ == '__main__':
    list_sentences = generate_dummy_sentences()
    print list_sentences
#     sys.exit(0)
    train_sents = tag_auto(list_sentences)
    train_sents = tag_manu(train_sents)
    train_test_model(train_sents)
    
    for sentence in list_sentences:
        text_tokenized = nltk.word_tokenize(sentence)
        train_sents.append(nltk.pos_tag(text_tokenized))
    sentence = 'I would like to order two hot latte and one cappuccino'
    print test_model(sentence)
    
    