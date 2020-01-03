#!/usr/bin/python
# -*- coding:utf-8 -*-
import script
import sys
import re
import io
def break_into_sentences(corpus_file):
    f = io.open(corpus_file, 'r', encoding='utf8')
    Sentence_Size = 0
    Max_Size = 4000  # Max_Size of the sentences
    paragraph = f.read()
	
    resultFile = io.open("SEG_Outputt.txt", 'w',encoding='utf8')
    sentences = list()
    temp_sentence = list()
    flag = False
    for ch in paragraph.strip():
        if ch in [u'؟', u'!', u'.', u'؛',u'?']:
            Sentence_Size = 0
            flag = True
        elif flag:
            sentences.append(''.join(temp_sentence).strip())
            temp_sentence = []
            flag = False
        regex = re.compile(r'[إأٱآا]')
        ch = re.sub(regex, 'ا', ch)
        regex = re.compile(r'[ئ]')
        ch = re.sub(regex, 'ى', ch)
        #remove_non_arabic_symbols fct
        ch = re.sub(r'[^\u0600-\u06FF]', ' ', ch)		
        temp_sentence.append(ch)
        if ch.isspace():
           Sentence_Size = Sentence_Size + 1
           if Sentence_Size > Max_Size:
              Sentence_Size = 0
              flag = True
              
    else:
        sentences.append(''.join(temp_sentence).strip())
        for item in sentences:
            resultFile.write("%s\n" % re.sub(' +', ' ', item))

if __name__ == '__main__':
    break_into_sentences(sys.argv[1])