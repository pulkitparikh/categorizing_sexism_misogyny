import sys
import numpy as np
import os, sys, pickle, csv, sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from evalMeasures import *
from loadPreProc import *
from string import punctuation
from word_embed import *
import nltk
from collections import Counter
from doc2vec_embed import *
from nltk.tokenize import TweetTokenizer
import time

def classification_model(X_train, X_test, y_train, y_tested, model_type):
    # print ("Model Type:", model_type)
    # print (X_train.shape)
    # print (y_train)
    # print (np.array(y_train).shape)
    model = get_model(model_type)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_tested
	    
def get_model(m_type):
    if m_type == 'logistic_regression':
        logreg = LogisticRegression()
    elif m_type == "random_forest":
        logreg = RandomForestClassifier(n_estimators=conf_dict_com['n_estimators'], n_jobs=-1, class_weight=conf_dict_com['class_weight'])
    elif m_type == "svm":
        logreg = LinearSVC(C=conf_dict_com['c_linear_SVC'],class_weight = conf_dict_com['class_weight'])
    elif m_type == "GBT":
        logreg = GradientBoostingClassifier(n_estimators= conf_dict_com['n_estimators'])
    else:
        print ("ERROR: Please specify a correct model")
        return None
    return logreg

def tf_idf(input_train,input_test,count_vec):
    tfidf_transformer = TfidfTransformer(norm = 'l2')
    bow_transformer_train= count_vec.fit_transform(input_train)
    bow_transformer_test =count_vec.transform(input_test)
    train_features = tfidf_transformer.fit_transform(bow_transformer_train).toarray()
    test_features= tfidf_transformer.transform(bow_transformer_test).toarray()
    return train_features,test_features

def feat_concat(features_word,features_char,features_POS,doc_feat,len_post,adj,text):
    features = []
    for i in range(len(text)): 
        features_text = np.append(features_word[i], features_char[i])
        features_text = np.append(features_text, features_POS[i])
        features_text = np.append(features_text, doc_feat[i])
        features_text = np.append(features_text, [len_post[i], adj[i]])
        features.append(features_text)
    return features

def train(data_dict,conf_dict_com):
    print (conf_dict_com['feat_type'])
    if conf_dict_com['feat_type']== "wordngrams":
        print("Using word based features")
        tfidf_transformer = TfidfTransformer(norm = 'l2')
        count_vec = CountVectorizer(analyzer="word",max_features = conf_dict_com['MAX_FEATURES'],stop_words='english',ngram_range = (1,2))
        bow_transformer_train = count_vec.fit_transform(data_dict['text'][0:data_dict['train_en_ind']])
        bow_transformer_test =count_vec.transform(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        train_features = tfidf_transformer.fit_transform(bow_transformer_train)
        test_features= tfidf_transformer.transform(bow_transformer_test )
    elif conf_dict_com['feat_type'] == "charngrams": 
        print("Using char n-grams based features")
        tfidf_transformer = TfidfTransformer(norm = 'l2')
        count_vec = CountVectorizer(analyzer="char",max_features = conf_dict_com['MAX_FEATURES'], ngram_range = (1,5))
        bow_transformer_train = count_vec.fit_transform(data_dict['text'][0:data_dict['train_en_ind']])
        bow_transformer_test =count_vec.transform(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        train_features = tfidf_transformer.fit_transform(bow_transformer_train)
        test_features= tfidf_transformer.transform(bow_transformer_test )         
    elif conf_dict_com['feat_type'] == "elmo":
        emb_size = conf_dict_com['poss_word_feats_emb_dict']['elmo']
        print("using elmo")
        train_features=[]
        test_features =[]
        for i in range(len(data_dict['text'][0:data_dict['train_en_ind']])):
            arr = np.load(conf_dict_com['filepath']+ str(i) + '.npy')
            avg_words = np.mean(arr, axis=0)
            train_features.append(avg_words)
        train_features = np.asarray(train_features)
        print (train_features.shape)
        inc = data_dict['test_st_ind']
        for i in range(len(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])):
            arr = np.load(conf_dict_com['filepath'] + str(inc) + '.npy')
            inc = inc + 1
            avg_words = np.mean(arr, axis=0)
            test_features.append(avg_words)
        test_features = np.asarray(test_features)
        print (test_features.shape)       
    elif conf_dict_com['feat_type'] == "ling_feat":
        len_post_train = []
        len_post_test = []
        POS_train = []
        POS_test = []
        adj_train = []
        adj_test =[]
        # doc2vec  features
        doc_feat_train = doc2vec_feat(data_dict['text'][0:data_dict['train_en_ind']])
        doc_feat_test = doc2vec_feat(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        # word ngrams and char ngrams
        count_vec_word = CountVectorizer(analyzer="word",max_features = conf_dict_com['MAX_FEATURES'],stop_words='english',ngram_range = (1,3))
        train_features_word, test_features_word = tf_idf(data_dict['text'][0:data_dict['train_en_ind']],data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']],count_vec_word)
        count_vec_char = CountVectorizer(analyzer="char",max_features = conf_dict_com['MAX_FEATURES'], ngram_range = (3,5))
        train_features_char, test_features_char = tf_idf(data_dict['text'][0:data_dict['train_en_ind']],data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']],count_vec_char)
        # linguistic features and POS
        for i in range(len(data_dict['text'][0:data_dict['train_en_ind']])):
            rep_te = data_dict['text'][i].replace(" ","")
            len_post_train.append(len(rep_te))
            adjectives_train =[token for token, pos in nltk.pos_tag(nltk.word_tokenize(data_dict['text'][i])) if pos.startswith('JJ')]
            counts_train = Counter(adjectives_train)
            adj_train.append(len(dict(counts_train.items())))
            POS_train.append([token + "_" + pos for token, pos in nltk.pos_tag(nltk.word_tokenize(data_dict['text'][i]))])
        ind = data_dict['test_st_ind']
        for j in range(len(data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])):
            rep_tex = data_dict['text'][ind].replace(" ","")
            len_post_test.append(len(rep_tex))
            adjectives_test =[token for token, pos in nltk.pos_tag(nltk.word_tokenize(data_dict['text'][ind])) if pos.startswith('JJ')]
            counts_test = Counter(adjectives_test)
            adj_test.append(len(dict(counts_test.items())))
            POS_test.append([token + "_" + pos for token, pos in nltk.pos_tag(nltk.word_tokenize(data_dict['text'][ind]))])
            ind = ind + 1
        POS_traindata = [' '.join(i) for i in POS_train]
        POS_testdata = [' '.join(i) for i in POS_test]
        train_features_POS, test_features_POS = tf_idf(POS_traindata,POS_testdata,count_vec_word)
        train_features = feat_concat(train_features_word,train_features_char,train_features_POS,doc_feat_train,len_post_train,adj_train,data_dict['text'][0:data_dict['train_en_ind']])
        test_features = feat_concat(test_features_word,test_features_char,test_features_POS,doc_feat_test,len_post_test,adj_test,data_dict['text'][data_dict['test_st_ind']:data_dict['test_en_ind']])
        print (np.shape(train_features))
        print (np.shape(test_features))

    return train_features, test_features

startTIme = time.time()
conf_dict_list, conf_dict_com = load_config(sys.argv[1])
data_dict = load_data(conf_dict_com['filename'], conf_dict_com['data_path'], conf_dict_com['save_path'], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com['test_mode'],conf_dict_com["filename_map"])
train_feat, test_feat = train(data_dict,conf_dict_com)
print (conf_dict_com["prob_trans_types"][0])
if conf_dict_com["prob_trans_types"][0] == "lp": 
    labels_lp,n_class,bac_map,for_map =fit_trans_labels_powerset(data_dict['lab'][:data_dict['train_en_ind']], data_dict['NUM_CLASSES'])
    for model_name in conf_dict_com['models']:
        print (model_name)
        metr_dict = init_metr_dict(data_dict['prob_type'])
        for run_ind in range(conf_dict_com["num_runs"]):
            pred, true = classification_model(train_feat, test_feat, labels_lp, data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']], model_name)
            y_predict = powerset_vec_to_label_lists(pred,bac_map, data_dict['NUM_CLASSES'])
            metr_dict = calc_metrics_print(y_predict, true, metr_dict, data_dict['NUM_CLASSES'], data_dict['prob_type'])
        metr_dict = aggregate_metr(metr_dict, conf_dict_com["num_runs"], data_dict['prob_type'])
        print_results(metr_dict,data_dict['prob_type'])
else:
    labels_bin =  trans_labels_bin_classi(data_dict['lab'][:data_dict['train_en_ind']])
    for model_name in conf_dict_com['models']:
        print (model_name)
        metr_dict = init_metr_dict(data_dict['prob_type'])
        for run_ind in range(conf_dict_com["num_runs"]):
            pred, true = classification_model(train_feat, test_feat, labels_bin[0], data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']], model_name)
            metr_dict = calc_metrics_print(pred, true, metr_dict, data_dict['NUM_CLASSES'], data_dict['prob_type'])
        metr_dict = aggregate_metr(metr_dict, conf_dict_com["num_runs"], data_dict['prob_type'])
        print_results(metr_dict,data_dict['prob_type'])
			
timeLapsed = int(time.time() - startTime + 0.5)
t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))
print(t_str)