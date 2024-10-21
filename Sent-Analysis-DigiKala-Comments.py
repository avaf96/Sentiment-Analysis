# -*- coding: utf-8 -*-
"""
Created on Wed Jun 4 10:11:05 2020

@author: Ava Fgh
"""


import re
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras import utils
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential, load_model
from keras.preprocessing import sequence



data_path = 'C:\\Users\\Digikala\\bin\\data'
model_path = 'C:\\Users\\Digikala\\bin\\model\\model.h5'


def open_and_filter_data(base_path):
    comment  = []
    label    = []
    user_id  = []
    comments_by_filter_none = []
    label_by_filter_none    = []
    user_id_filter_none  = []
    for i in range(2, 66):
        print(i)
        if i == 3: continue
        print('             ' + str(i))
        address = base_path + '\\split_' + str(i) + '.xlsx'
        data = pd.read_excel(address, converters={ 'user_id': float ,'comment': str, 'label': int})
        user_id = data['user_id'].tolist()
        comment = data['comment'].tolist()
        label = data['label'].tolist()
        for c in comment: comments_by_filter_none.append(c)
        for l in label: label_by_filter_none.append(l)
        for u in user_id: user_id_filter_none.append(u)


    size_of_comment = len(comments_by_filter_none)

    indexs_for_remove_none = []
    for i in range(size_of_comment):
        if isinstance(comments_by_filter_none[i], str):
            if comments_by_filter_none[i] is None or len(comments_by_filter_none[i]) == 0 or label_by_filter_none[i] is None:
                indexs_for_remove_none.append(i)
        else:
                indexs_for_remove_none.append(i)


    for i in range(len(indexs_for_remove_none)):
        key = indexs_for_remove_none[i]
        if not isinstance(comments_by_filter_none[key], str) and math.isnan(comments_by_filter_none[key]):
            if key <= 1000:
                print('in file 2 comment error in line ' + str(key))
            else:
                p = int(key / 10)
                p1 = int(p / 10)
                p2 = int(p1 / 10)

                p3 = key + p2 + 2
                a = list(map(int, ' '.join(str(p3)).split()))
                if key <= 10000:
                    a.pop(0)
                else:
                    a.pop(0)
                    a.pop(0)

                print('in file  ' + str(p2 + 3) + '  comment error in line  ' + str(a[0]) + str(a[1]) + str(a[2]))

    for m in range(len(indexs_for_remove_none)):
        key = indexs_for_remove_none[m]
        if math.isnan(label_by_filter_none[key]):
            if key <= 1000:
                print('in file 2 label error in line ' + str(key))
            else:
                u = int(key / 10)
                u1 = int(u / 10)
                u2 = int(u1 / 10)

                u3 = key + u2 + 2
                b = list(map(int, ' '.join(str(u3)).split()))
                if key <= 10000:
                    b.pop(0)
                else:
                    b.pop(0)
                    b.pop(0)

                print('in file  ' + str(u2 + 3) + '  label error in line  ' + str(b[0]) + str(b[1]) + str(b[2]))


    text = input('To continue and replace non-comment entries with a space and non-label entries with 0, enter 1. To exit, enter 2: ')

    if int(text) == 1 :
        for row in range(len(comments_by_filter_none)):
            if isinstance(comments_by_filter_none[row], str):
                if len(comments_by_filter_none[row]) == 0 or math.isnan(label_by_filter_none[row]) or \
                        (label_by_filter_none[row] != -1 and label_by_filter_none[row] != 0 and label_by_filter_none[row] != 1 ):
                   comments_by_filter_none[row] = ''
                   label_by_filter_none[row] = 0
            else:
                comments_by_filter_none[row] = ''
                label_by_filter_none[row] = 0

    elif text == 2 :  exit()

    return  user_id_filter_none ,comments_by_filter_none, label_by_filter_none



tokenizer = Tokenizer(num_words=1000, split=' ')

def replace_additions_of_comment_and_tokenizer(comment ,labels ):
    comment = [re.compile("[0-9A_Za-z۰-۹]").sub("", str(line)) for line in comment]
    comment = [re.compile("[.`;:!\'?,\"()\[\]،؛ًٌٍَُِّ]").sub("", str(line)) for line in comment]
    comment = [re.sub(r"[\s]", " ", str(line)) for line in comment]
    comment = [cm.replace('.', ' ') for cm in comment]
    tokenize_comment = tokenizer.texts_to_sequences(comment)
    return tokenize_comment , labels





def create_model(vsize):
    model = Sequential()
    model.add(Embedding(vsize , 64))
    model.add(Bidirectional(LSTM(64, dropout=0.2)))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model



print('opening data ')
user_id ,comments, labels = open_and_filter_data(data_path)




while True :
    #test   =    6671599
    text = input('To insert a comment, press 1 - To analyze human, press 2 : ')
    if int(text) == 2:
        id_search = input('Enter user id  : ')
        po = 0
        ne = 0
        pone = 0
        count = 0
        for m in range(len(user_id)):
            if float(user_id[m]) == float(id_search):
                if   labels[m] == 1 :
                    po = po +1
                    count = count + 1
                elif  labels[m] == 0 :
                    pone = pone +1
                    count = count + 1
                elif  labels[m] == -1 :
                    ne = ne + 1
                    count = count + 1
        if count != 0 :
           if po > ne : x = 'positive'
           elif ne > po : x = 'negative'
           else: x= 'neutral'

           print(  str(id_search ) + ' have a '  + str(count) +' comment and have ' + str(po) +" positive  comment and " + str(ne) +
                 " negative  comment and "+ str(pone) +" neutral  comment and human is : " + x )
        elif count == 0 :
            print(str(id_search ) + ' not found ' )


    elif int(text) == 1:

        if os.path.isfile(model_path):
            model_is_ready = True
        else:
            model_is_ready = False

        print('clean and tokeniz data')

        tokenizer.fit_on_texts(comments)
        x , y = replace_additions_of_comment_and_tokenizer(comments, labels)
        y = np.asarray(y)

        comment_train, comment_test, label_train, label_test = train_test_split(x, y, test_size=0.2, random_state=42)
        comment_train = sequence.pad_sequences(comment_train, maxlen=128)
        comment_test = sequence.pad_sequences(comment_test, maxlen=128)

        label_test = utils.to_categorical(label_test, 3)
        label_train = utils.to_categorical(label_train, 3)

        if model_is_ready:
            print("model exists. ")
            model = load_model(model_path)

        else:
            print("model does not exist, please wait for training ...")
            model = create_model(10000)
            his_model = model.fit(comment_train, label_train, batch_size=10, epochs=2, validation_split=0.2)
            model.save(model_path)
            print(his_model.history)
            history = his_model.history

            # acc = his.history['accuracy']
            # val_acc = his.history['val_accuracy']
            # loss = his.history['loss']
            # val_loss = his.history['val_loss']
            # epochs = range(1, len(acc) + 1)
            # plt.plot(epochs, acc, 'bo', label='Traning acc')
            # plt.plot(epochs, val_acc, 'b', label='Validation acc')
            # plt.title('Traning and validation accuracy')
            # plt.legend()
            # plt.figure()
            # plt.plot(epochs, loss, 'bo', label='Traning loss')
            # plt.plot(epochs, val_loss, 'b', label='Validation loss')
            # plt.title('Traning and validation loss')
            # plt.legend()
            # plt.show()


        accuracy, score = model.evaluate(comment_test, label_test, batch_size=10)
        print('Test score is    :', score)
        print('Test accuracy is :', accuracy)

        while True:
            input_comment = input('please insert comment : ')
            tokens = tokenizer.texts_to_sequences([input_comment])
            p = sequence.pad_sequences(tokens, maxlen=128)
            result = model.predict(p)
            print('positive   + : ', str(round(result[0][1] * 100, 4)) + '%')
            print('neutral    o : ', str(round(result[0][0] * 100, 4)) + '%')
            print('negative   - : ', str(round(result[0][2] * 100, 4)) + '%')