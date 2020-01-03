# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:12:02 2019

@author: Dhia
"""
#from spacy.lang.ar import Arabic
#import spacy
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import math
import io
import heapq
import re
#from stop_words import get_stop_words
#stopwords = spacy.lang.ar.stop_words.STOP_WORDS
#print('Number of stop words: %d' % len(stopwords))
#stopwords = get_stop_words('arabic')

#importer les stop_words
def stop_words():
    file = io.open("words/stop_words.txt", 'r', encoding='utf8')
    #stop_words = file.read()
    stop_words = []
    for word in file:
        regex = re.compile(r'[إأٱآا]')
        word = re.sub(regex, 'ا', word)
        regex = re.compile(r'[ئ]')
        word = re.sub(regex, 'ى', word)
        stop_words.append(word.strip())
    return stop_words

def mots_bonus():
    file = io.open("words/exp_bonus.txt", 'r', encoding='utf8')
    #stop_words = file.read()
    mots_bonus = []
    for word in file:
        regex = re.compile(r'[إأٱآا]')
        word = re.sub(regex, 'ا', word)
        regex = re.compile(r'[ئ]')
        word = re.sub(regex, 'ى', word)
        mots_bonus.append(word.strip())
    return mots_bonus

def mots_stigma():
    file = io.open("words/exp_stigma.txt", 'r', encoding='utf8')
    #stop_words = file.read()
    mots_stigma = []
    for word in file:
        regex = re.compile(r'[إأٱآا]')
        word = re.sub(regex, 'ا', word)
        regex = re.compile(r'[ئ]')
        word = re.sub(regex, 'ى', word)
        mots_stigma.append(word.strip())
    return mots_stigma

def mots_anapho():
    file = io.open("words/mot_anapho.txt", 'r', encoding='utf8')
    #stop_words = file.read()
    mots_anapho = []
    for word in file:
        regex = re.compile(r'[إأٱآا]')
        word = re.sub(regex, 'ا', word)
        regex = re.compile(r'[ئ]')
        word = re.sub(regex, 'ى', word)
        mots_anapho.append(word.strip())
    return mots_anapho

#importer fichier
def file_loader(file):
    file= io.open(file, 'r', encoding='utf8')
    return file

def title_loader(file):
    file= io.open(file, 'r', encoding='utf8')
    return file

#netoyer les phrases
def clean_sentences(file):
    lines = []
    for line in file:
        line = re.sub(r"[-،»«0123456789()\"#/@;:<>{}+=~|.?,)(]",'',line.strip())
        lines.append(line)
    return lines

#tokenizer les mots
def tokenizer(lines, stop_words):
    file = lines
    words = []
    lines = clean_sentences(file)
    for line in lines:
        for word in line.split(" "):
            if (word not in " " and word not in stop_words):
                words.append(word)
    return words

def tf(words, sentences):
    tf = {}
    for word in words:
        if word not in tf.keys():
            tf[word] = 1
        else:
            tf[word] += 1
    
    #fréquences des mots ( TF )
    n_words = len(words)
    for word in tf.keys():
        tf[word] = (tf[word]/n_words)
    
    #scores des phrases
    sentence_scores = {}
    for sent in sentences:
        if len(sent.split(' ')):
            for word in sent.split(" "):
                if word in tf.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = float(tf[word])
                    else:
                        sentence_scores[sent] += float(tf[word])
    return sentence_scores

def tf_idf(words, sentences):
    tf = {}
    for word in words:
        if word not in tf.keys():
            tf[word] = 1
        else:
            tf[word] += 1
    
    #fréquences des mots ( TF )
    n_words = len(words)
    for word in tf.keys():
        tf[word] = (tf[word]/n_words)
        
    # IDF
    idf = {}
    n_sentences = len(sentences)
    for word in words:
        n_sent_word = 0
        for sent in sentences:
            if word in sent.split(" "):
                n_sent_word += 1
        idf[word] = math.log(n_sentences/n_sent_word)
        
    #TF-IDF
    tf_idf = {}
    for key, value in tf.items():
        tf_idf[key] = value * idf[key]
    
    #scores des phrases
    sentence_scores = {}
    for sent in sentences:
        if len(sent.split(' ')):
            for word in sent.split(" "):
                if word in tf_idf.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = float(tf_idf[word])
                    else:
                        sentence_scores[sent] += float(tf_idf[word])
    return sentence_scores

def sent_pos_text(sentences):
    sent_pos_text = {}
    for sent in sentences:
        if ((sentences.index(sent)+1)<=math.ceil(len(sentences)*1/3)):
            sent_pos_text[sent] = 1
        elif((sentences.index(sent)+1)>math.ceil(len(sentences)*2/3)):
            sent_pos_text[sent] = 3
        else:
            sent_pos_text[sent] = 2
    return sent_pos_text

def sent_pos_para(sentences):
    sent_pos_para = {}
    tier1 = math.ceil(len(sentences)*1/3)
    tier2 = math.ceil(len(sentences)*2/3)
    for sent in sentences:
        if ((sentences.index(sent)+1)<=tier1):
            if ((sentences.index(sent)+1)<=math.ceil(tier1*1/3)):
                sent_pos_para[sent] = 1
            elif ((sentences.index(sent)+1)>math.ceil(tier1*2/3)):
                sent_pos_para[sent] = 3
            else:
                sent_pos_para[sent] = 2
        elif ((sentences.index(sent)+1)>tier2):
            if ((sentences.index(sent)+1)<=math.ceil((len(sentences)-tier2)*1/3+tier2)):
                sent_pos_para[sent] = 1
            elif ((sentences.index(sent)+1)>math.ceil((len(sentences)-tier2)*2/3+tier2)):
                sent_pos_para[sent] = 3
            else:
                sent_pos_para[sent] = 2
        else:
            if ((sentences.index(sent)+1)<=math.ceil((tier2-tier1)*1/3+tier1)):
                sent_pos_para[sent] = 1
            elif ((sentences.index(sent)+1)>math.ceil((tier2-tier1)*2/3+tier1)):
                sent_pos_para[sent] = 3
            else:
                sent_pos_para[sent] = 2
    return sent_pos_para
    
def n_words_title(sentences, stop_words, title):
    n_words_titles = {}
    title = tokenizer(clean_sentences(title_loader(title)), stop_words)
    for sent in sentences:
        word_title = 0
        for word in title:
            if word in sent.split(" "):
               word_title += 1
        n_words_titles[sent] = word_title
    return n_words_titles

def first_sent_text(sentences):
    first_sent_text = {}
    for sent in sentences:
        if sentences.index(sent)==0:
            first_sent_text[sent] = 1
        else:
            first_sent_text[sent] = 0
    return first_sent_text
    
def first_sent_para(sentences):
    first_sent_para = {}
    for sent in sentences:
        if ((sentences.index(sent)+1)==math.ceil(len(sentences)*1/3)):
            first_sent_para[sent] = 1
        elif((sentences.index(sent))==math.ceil(len(sentences)*2/3)):
            first_sent_para[sent] = 1
        elif (sentences.index(sent)==0):
            first_sent_para[sent] = 1
        else:
            first_sent_para[sent] = 0
    return first_sent_para


def Nb_exp_bonus(sentences, motsbonus):
    Nb_exp_bonus = {}
    for sent in sentences:
        nb_mots= 0
        for word in motsbonus:
            if word in sent.split(" "):
               nb_mots += 1
            Nb_exp_bonus[sent] = nb_mots
    return Nb_exp_bonus

def Nb_exp_stigma(sentences, motsstigma):
    Nb_exp_stigma = {}
    for sent in sentences:
        nb_mots= 0
        for word in motsstigma:
            if word in sent.split(" "):
               nb_mots += 1
            Nb_exp_stigma[sent] = nb_mots
    return Nb_exp_stigma

def Mot_anapho(sentences, motsanapho):
    Mot_anapho = {}
    for sent in sentences:
        nb_mots= 0
        for word in motsanapho:
            if word in sent.split(" "):
               nb_mots += 1
            Mot_anapho[sent] = nb_mots
    return Mot_anapho

def sent_label(tf_idf, sentences):
    sentence_scores = tf_idf    
    sent_label = {}
    summary_sentences = heapq.nlargest(math.ceil(len(sentence_scores)/3), sentence_scores, key=sentence_scores.get)
    for item in sentences:
        if item in summary_sentences:
            sent_label[item] = 1
        else:
            sent_label[item] = 0
    return sent_label
    
#features&labels
def feat_label(words, title, sentences, stop_words, mots_bonus, mots_stigma, mots_anapho):
    feat_label = pd.DataFrame()
    feat_label = feat_label.append([sent_pos_text(sentences), sent_pos_para(sentences), n_words_title(sentences, stop_words, title), 
                                    first_sent_text(sentences), first_sent_para(sentences), Nb_exp_bonus(sentences, mots_bonus),
                                    Nb_exp_stigma(sentences, mots_stigma), Mot_anapho(sentences, mots_anapho),
                                    tf_idf(words, sentences), tf(words, sentences), sent_label(tf_idf(words, sentences), sentences)] ,ignore_index=True).T
    feat_label.columns = ['Position_ph_texte','Position_ph_parag','Nb_mot_titre','Pos_ph_texte','Pos_ph_parag','Nb_exp_bonus','Nb_exp_stigma',
                          'Mot_anapho','tf-idf','tf','label']
    return feat_label

#SVM Model
def SVM_model(feat_label):
    X = feat_label.drop('label', axis=1)
    y = feat_label['label']    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print('\n')
    print(classification_report(y_test,y_pred))
    modelname = 'SVM.sav'
    pickle.dump(model, open(modelname, 'wb'))
    
def load_model(modelname, features):
    load_model = pickle.load(open(modelname, 'rb'))
    y_pred = load_model.predict(features)
    print("\nles predictions sont: \n")
    print(y_pred)

def extract_features(words, title, sentences, stop_words, mots_bonus, mots_stigma, mots_anapho):
    feat_label = pd.DataFrame()
    feat_label = feat_label.append([sent_pos_text(sentences), sent_pos_para(sentences), n_words_title(sentences, stop_words, title), 
                                    first_sent_text(sentences), first_sent_para(sentences), Nb_exp_bonus(sentences, mots_bonus),
                                    Nb_exp_stigma(sentences, mots_stigma), Mot_anapho(sentences, mots_anapho),
                                    tf_idf(words, sentences), tf(words, sentences)] ,ignore_index=True).T
    feat_label.columns = ['Position_ph_texte','Position_ph_parag','Nb_mot_titre','Pos_ph_texte','Pos_ph_parag','Nb_exp_bonus','Nb_exp_stigma',
                          'Mot_anapho','tf-idf','tf']
    return feat_label
#résumé 
def resume(words, sentences, filename):                   
    tf = {}
    for word in words:
        if word not in tf.keys():
            tf[word] = 1
        else:
            tf[word] += 1
    
    #fréquences des mots ( TF )
    n_words = len(words)
    for word in tf.keys():
        tf[word] = (tf[word]/n_words)
        
    # IDF
    idf = {}
    n_sentences = len(sentences)
    for word in words:
        n_sent_word = 0
        for sent in sentences:
            if word in sent.split(" "):
                n_sent_word += 1
        idf[word] = math.log(n_sentences/n_sent_word)
        
    #TF-IDF
    tf_idf = {}
    for key, value in tf.items():
        tf_idf[key] = value * idf[key]
    
    len_sents = 0
    for sent in sentences:
        len_sents += len(sent.split(' '))
    m_sent = math.ceil(len_sents/n_sentences)
    
    #scores des phrases
    sentence_scores = {}
    for sent in sentences:
        if len(sent.split(' ')) <= m_sent:
            for word in sent.split(" "):
                if word in tf_idf.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = float(tf_idf[word])
                    else:
                        sentence_scores[sent] += float(tf_idf[word])
    
    summary_sentences = heapq.nlargest(math.ceil(len(sentence_scores)/3), sentence_scores, key=sentence_scores.get)
    resume = io.open(filename, 'w+',encoding='utf8')
    for item in sentences:
        if item in summary_sentences:
            resume.write("%s\n" %item)
    resume.close()

def main():
    stopwords = stop_words()
    motsbonus = mots_bonus()
    motsstigma = mots_stigma()
    motsanapho = mots_anapho()
    file = file_loader("text1.txt")
    title = "title1.txt"
    sentences = clean_sentences(file)
    words = tokenizer(sentences, stopwords)
    features_labels = feat_label(words, title, sentences, stopwords, motsbonus, motsstigma, motsanapho)
    features_labels = features_labels.append(feat_label(words, title, sentences, stopwords, motsbonus, motsstigma, motsanapho))
    features_labels.to_csv("features_labels.csv", sep='\t', encoding ='utf-8-sig')
    #resume(words, sentences, "résumé.txt")
    SVM_model(features_labels)
    test_file = file_loader("text2.txt")
    title_test = "title2.txt"
    test_sentences = clean_sentences(test_file)
    test_words = tokenizer(test_sentences, stopwords)
    test_features = extract_features(test_words, title_test, test_sentences, stopwords, motsbonus, motsstigma, motsanapho)
    model = load_model("SVM.sav", test_features)
    resume(test_words, test_sentences, 'test_résumé.txt')
    test_features.to_csv("test_features_labels.csv", sep='\t', encoding='utf-8-sig')
    test_features.to_excel("output.xlsx")
    #resume(test_words, test_sentences, "test_résumé.txt")


if __name__ == '__main__':
    main()