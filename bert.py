import os
import tensorflow as tf
import tensorflow_text
import numpy as np
import re
import glob
from natsort import natsorted

num_regex = re.compile(r'[a-zA-Z0-9 ,$\'\-]+\? *|[a-zA-Z0-9 ,$\'\-]+! *|[a-zA-Z0-9 ,$\'\-]+\. *|[a-zA-Z0-9 ,$\'\-]+ *')
saved_model_path = './my_models/glue_cola_bert_en_uncased_L-12_H-768_A-12'

with tf.device('/job:localhost'):
  reloaded_model = tf.saved_model.load(saved_model_path) #モデルのロード

files = natsorted(glob.glob('./cb_aug_ri/eda_1_*.txt')) #読み込みファイルリスト

with tf.device('/job:localhost'):
    acceptable = 0
    unacceptable = 0
    
    for infile in files:
        lines = open(infile, 'r').readlines() #１行ごと
        print(infile)
        for line in lines:
            sentences = num_regex.findall(line) #行を文単位に分割
            for sentence in sentences:
                test_data = tf.constant([sentence]) #１行をモデルの入力タイプに変換
                result = reloaded_model(test_data) #テスト
                bert_result_class = tf.argmax(result, axis=1)[0] #結果のクラス分け
                if bert_result_class == 1:
                    acceptable += 1
                else:
                    unacceptable += 1
    
    outfile = open('./cb_aug_ri/cb_ri_cola1.txt', 'w') #
    outfile.write('acceptable = ' + str(acceptable) + '\n')
    outfile.write('unacceptable = ' + str(unacceptable) + '\n')
    print([acceptable, unacceptable])
    outfile.close()
    