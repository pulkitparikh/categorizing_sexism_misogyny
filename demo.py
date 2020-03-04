import re
from loadPreProc import load_map, di_list_op_to_label_lists
from sent_enc_embed import sent_enc_featurize
from word_embed import word_featurize
from nltk import sent_tokenize
from keras.models import load_model
from gen_batch_keras import TrainGenerator, TestGenerator
import numpy as np
from dlModels import attLayer_hier, multi_binary_loss
import os
import sys

sys.setrecursionlimit(10000)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

filename_accounts = 'data/accounts_of_sexism.txt'

def evaluate_model(mod_op_list, data_dict, classi_probs_label_info):
    y_pred_list = []
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val
        y_pred = np.rint(mod_op).astype(int)
        y_pred_list.append(y_pred)

        pred_vals = di_list_op_to_label_lists(y_pred_list, mod_op_list, data_dict['NUM_CLASSES'], classi_probs_label_info)
    return pred_vals

def gen_raw_output(num_saved_mods, saved_mod_path, test_generator, num_classes):
    mod_op_list = []
    for m_ind in range(num_saved_mods):
        model = load_model("%smod~%d.h5" % (saved_mod_path, m_ind), custom_objects={'attLayer_hier': attLayer_hier, 'multi_binary_of': multi_binary_loss(np.empty([num_classes, 2]))})
        # model.summary()
        # exit()
        mod_op = model.predict_generator(generator=test_generator, verbose=1, use_multiprocessing=False, workers=1)
        mod_op_list.append((mod_op, None))
    return mod_op_list

def load_data(list_posts, max_num_sent, max_words_sent, max_post_length, filename_map):
    conf_map = load_map(filename_map)
    r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
    r_white = re.compile(r'[\s.(?)!]+')
    text_sen = []
    text = []
    for post in list_posts:
        se_list = []
        for se in sent_tokenize(post):
            se_cl = r_white.sub(' ', r_anum.sub('', str(se).lower())).strip()
            if se_cl == "":
                continue
            words = se_cl.split(' ')
            while len(words) > max_words_sent:
                se_list.append(' '.join(words[:max_words_sent]))
                words = words[max_words_sent:]
            se_list.append(' '.join(words))
        text_sen.append(se_list)
        text.append(' '.join(se_list))

    data_dict = {}  
    data_dict['text_sen'] = text_sen
    data_dict['text'] = text

    data_dict['max_num_sent'] = max_num_sent
    data_dict['max_words_sent'] = max_words_sent
    data_dict['max_post_length'] = max_post_length

    data_dict['test_st_ind'] = 0
    data_dict['test_en_ind'] = len(text_sen)

    data_dict['FOR_LMAP'] = conf_map['FOR_LMAP']
    # data_dict['LABEL_MAP'] = conf_map['LABEL_MAP']
    data_dict['NUM_CLASSES'] = len(data_dict['FOR_LMAP'])
    data_dict['prob_type'] = conf_map['prob_type']

    return data_dict

def predict_main(list_posts, data_folder_path, saved_model_path, temp_save_path):
    model_type = 'hier_fuse'
    word_feats_raw = [{'emb': 'elmo', 's_enc': 'rnn', 'm_id': '11'}, {'emb': 'glove', 's_enc': 'rnn', 'm_id': '21'}]
    sent_enc_feats_raw = [{'emb': 'bert_pre', 'm_id': '1'}]
    conf_dict_com = {'poss_sent_enc_feats_emb_dict': {'bert_pre': 768}, 'poss_word_feats_emb_dict': {'glove': 300, 'ling': 33, 'elmo': 3072}}
    classi_probs_label_info = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13],[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

    data_dict = load_data(list_posts, 16, 35, 198, data_folder_path + 'esp_class_maps.txt')
    for cl_post in data_dict['text_sen']:
        print(cl_post)
    word_feats, word_feat_str = word_featurize(word_feats_raw, model_type, data_dict, conf_dict_com['poss_word_feats_emb_dict'], False, False, data_folder_path, temp_save_path, True)
    sent_enc_feats, sent_enc_feat_str = sent_enc_featurize(sent_enc_feats_raw, model_type, data_dict, conf_dict_com['poss_sent_enc_feats_emb_dict'], False, False, data_folder_path, temp_save_path, True)
    # print(word_feats)
    # print(sent_enc_feats)
    # print(sent_enc_feats[0]['feats'].shape)
    # print(np.arange(data_dict['test_st_ind'], data_dict['test_en_ind']))
    # exit()
    test_generator = TestGenerator(np.arange(data_dict['test_st_ind'], data_dict['test_en_ind']), word_feats, sent_enc_feats, data_dict, 64)
    mod_op_list = gen_raw_output(2, saved_model_path, test_generator, data_dict['NUM_CLASSES'])
  
    pred_vals = evaluate_model(mod_op_list, data_dict, classi_probs_label_info)
    
    pred_categories = []
    for pred_val_set in pred_vals:
        pred_categories.append([data_dict['FOR_LMAP'][x] for x in pred_val_set])
    return pred_categories

data_folder_path = 'data/'
saved_model_path = 'saved/hier_fuse~elmo~rnn~11~~glove~rnn~21~~~~~~~~~~bert_pre~1~~~~~0_1_2_3_4_5_6_7_8_9_10_11_12_13+0_1_2_3_4_5_6_7_8_9_10_11_12_13~di~True~100~2_3_4~lstm~300~500~1~False~True/'
temp_save_path = 'temp_saved/'

with open(filename_accounts) as file_in:
    list_posts = []
    for post in file_in:
        list_posts.append(post)

# list_posts = ['I am x. you are y', 'You are y']
# print(list_posts)
pred_categories = predict_main(list_posts, data_folder_path, saved_model_path, temp_save_path)

for pred_categ in pred_categories:
    print(pred_categ)