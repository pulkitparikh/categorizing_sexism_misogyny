import csv 
from sklearn.utils import shuffle
from ast import literal_eval
import numpy as np   
import os
import re
import pickle
from nltk import sent_tokenize

def is_model_hier(model_type):
  if model_type.startswith('hier'):
    return True
  return False

def load_data(filename, data_path, save_path, test_ratio, valid_ratio, rand_state, max_words_sent, test_mode, filename_map, use_saved_data_stuff, save_data_stuff):
  data_dict_filename = ("%sraw_data~%s~%s~%s~%s~%s~%s.pickle" % (save_path, filename[:-4], test_ratio, valid_ratio, rand_state, max_words_sent, test_mode))
  if use_saved_data_stuff and os.path.isfile(data_dict_filename):
    print("loading input data dict")
    with open(data_dict_filename, 'rb') as f_data:
        data_dict = pickle.load(f_data)
  else:      
    cl_in_filename = ("%sraw_data~%s~%s.pickle" % (save_path, filename[:-4], max_words_sent))
    if use_saved_data_stuff and os.path.isfile(cl_in_filename):
      print("loading cleaned unshuffled input")
      with open(cl_in_filename, 'rb') as f_cl_in:
          text, text_sen, label_lists, conf_map = pickle.load(f_cl_in)
    else:
      conf_map = load_map(data_path + filename_map)
      r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
      r_white = re.compile(r'[\s.(?)!]+')
      text = []; label_lists = []; text_sen = []
      with open(data_path + filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter = '\t')
        for row in reader:
          post = str(row['post'])
          row_clean = r_white.sub(' ', r_anum.sub('', post.lower())).strip()
          text.append(row_clean)

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

          cat_list = str(row['labels']).split(',')
          label_ids = list(set([conf_map['LABEL_MAP'][cat] for cat in cat_list]))
          label_lists.append(label_ids)

      if save_data_stuff:
        print("saving cleaned unshuffled input")
        with open(cl_in_filename, 'wb') as f_cl_in:
          pickle.dump([text, text_sen, label_lists, conf_map], f_cl_in)

    data_dict = {}  
    data_dict['text'], data_dict['text_sen'], data_dict['lab'] = shuffle(text, text_sen, label_lists, random_state = rand_state)
    train_index = int((1 - test_ratio - valid_ratio)*len(text)+0.5)
    val_index = int((1 - test_ratio)*len(text)+0.5)

    data_dict['max_num_sent'] = max([len(post_sen) for post_sen in data_dict['text_sen'][:val_index]])
    data_dict['max_post_length'] = max([len(post.split(' ')) for post in data_dict['text'][:val_index]])
    data_dict['max_words_sent'] = max_words_sent

    if test_mode:
      data_dict['train_en_ind'] = val_index
      data_dict['test_en_ind'] = len(text)
    else:
      data_dict['train_en_ind'] = train_index
      data_dict['test_en_ind'] = val_index
    data_dict['test_st_ind'] = data_dict['train_en_ind']

    data_dict['FOR_LMAP'] = conf_map['FOR_LMAP']
    data_dict['LABEL_MAP'] = conf_map['LABEL_MAP']
    data_dict['NUM_CLASSES'] = len(data_dict['FOR_LMAP'])
    data_dict['prob_type'] = conf_map['prob_type']

    if save_data_stuff:
      print("saving input data dict")
      with open(data_dict_filename, 'wb') as f_data:
        pickle.dump(data_dict, f_data)

  return data_dict

def load_map(filename):
  conf_sep = "----------"
  content = ''
  with open(filename, 'r') as f:
    for line in f:
      line = line.strip()
      if line != '' and line[0] != '#':
        content += line

  items = content.split(conf_sep)
  conf_map = {}
  for item in items:
    parts = [x.strip() for x in item.split('=')]
    conf_map[parts[0]] = literal_eval(parts[1])
  # print(conf_map)
  return conf_map

def load_config(filename):
  print("loading config")
  conf_sep_1 = "----------\n"
  conf_sep_2 = "**********\n"
  conf_dict_list = []
  conf_dict_com = {}
  with open(filename, 'r') as f:
    content = f.read()
  break_ind = content.find(conf_sep_2)  

  nested_comps = content[:break_ind].split(conf_sep_1)
  for comp in nested_comps:
    pairs = comp.split(';')
    conf_dict = {}
    for pair in pairs:
      pair = ''.join(pair.split())
      if pair == "" or pair[0] == '#': 
        continue
      parts = pair.split('=')
      conf_dict[parts[0]] = literal_eval(parts[1])
    conf_dict_list.append(conf_dict)

  lines = content[break_ind+len(conf_sep_2):].split('\n')
  for pair in lines:
    pair = ''.join(pair.split())
    if pair == "" or pair[0] == '#': 
      continue
    parts = pair.split('=')
    conf_dict_com[parts[0]] = literal_eval(parts[1])

  print("config loaded")
  return conf_dict_list, conf_dict_com

def binary_to_decimal(b_list):
  out1 = 0
  for bit in b_list:
    out1 = (out1 << 1) | bit
  return out1

def num_to_label_list(num, NUM_CLASSES):
  f_str = ("%sb" % NUM_CLASSES)
  return [ind for ind, x in enumerate(format(num, f_str)) if x == '1']

def powerset_vec_to_label_lists(vec, bac_map, NUM_CLASSES):
  return [num_to_label_list(bac_map[x], NUM_CLASSES) for x in vec]

# def powerset_vec_to_single_label(vec, bac_map, NUM_CLASSES):
#   return [num_to_label_list(bac_map[x], NUM_CLASSES)[0] for x in vec]

def br_op_to_label_lists(vecs):
  NUM_CLASSES = len(vecs)
  op_list = []
  for i in range(len(vecs[0])):
    cat_l = []
    for j in range(NUM_CLASSES):
      if vecs[j][i] == 1:
        cat_l.append(j)  
    op_list.append(cat_l)    
  return op_list

def di_op_to_label_lists(vecs):
  NUM_CLASSES = len(vecs[0])
  op_list = []
  for i in range(len(vecs)):
    cat_l = []
    for j in range(NUM_CLASSES):
      if vecs[i][j] == 1:
        cat_l.append(j)  
    op_list.append(cat_l)    
  return op_list

def di_list_op_to_label_lists(vecs_list, mod_op_list, NUM_CLASSES, classi_probs_label_info):
  label_arr = np.zeros([len(vecs_list[0]), NUM_CLASSES], dtype=np.int64)
  label_nn = np.zeros([len(vecs_list[0]), NUM_CLASSES], dtype=np.int64)
  sum_arr = np.zeros([len(vecs_list[0]), NUM_CLASSES])
  count_arr = np.zeros([len(vecs_list[0]), NUM_CLASSES])
  for ind, vecs in enumerate(vecs_list):
    for i, vec in enumerate(vecs):
      for j, val in enumerate(vec):
        act_ind = classi_probs_label_info[ind][j]
        if val == 1:
          label_nn[i, act_ind] += 1 
        else:
          label_nn[i, act_ind] -= 1 
        sum_arr[i, act_ind] += mod_op_list[ind][0][i][j]
        count_arr[i, act_ind] += 1
  for i in range(len(label_arr)):
    for j in range(NUM_CLASSES):
      if label_nn[i,j] > 0:
        label_arr[i,j] = 1
      elif label_nn[i,j] == 0:
        label_arr[i,j] = np.rint(sum_arr[i,j]/count_arr[i,j]).astype(int)

    if sum(label_arr[i]) == 0:
      max_val = 0
      for c in range(NUM_CLASSES):
        cur_val = sum_arr[i, c]/count_arr[i, c]
        if cur_val > max_val:
          max_val = cur_val
          max_ind = c
      label_arr[i,max_ind] = 1
  return di_op_to_label_lists(label_arr)

def map_labels_to_num(label_ids, NUM_CLASSES):
  arr = [0] * NUM_CLASSES
  for label_id in label_ids:
    arr[label_id] = 1
  num = binary_to_decimal(arr) 
  return num

def fit_trans_labels_powerset(org_lables, NUM_CLASSES):
  ind = 0
  for_map = {}
  bac_map = {}
  new_labels = np.empty(len(org_lables), dtype=np.int64)
  for s_ind, label_ids in enumerate(org_lables):
    l = map_labels_to_num(label_ids, NUM_CLASSES)
    if l not in for_map:
      for_map[l] = ind
      bac_map[ind] = l
      ind += 1
    new_labels[s_ind] = for_map[l]
  num_lp_classes = ind
  return new_labels, num_lp_classes, bac_map, for_map

def trans_labels_multi_hot(org_lables, NUM_CLASSES):
  label_arr = np.zeros([len(org_lables), NUM_CLASSES], dtype=np.int64)
  for sample_ind, label_ids in enumerate(org_lables):
    for label_id in label_ids:
      label_arr[sample_ind][label_id] = 1
  return label_arr 

def trans_labels_multi_hot_list(org_lables, classi_probs_label_info):
  label_arr_list = []
  num_classes_list = []
  for l_list in classi_probs_label_info:
    label_arr = np.zeros([len(org_lables), len(l_list)], dtype=np.int64)
    map_dict = {}
    for ind, l in enumerate(l_list):
      map_dict[l] = ind
    for sample_ind, label_ids in enumerate(org_lables):
      for label_id in label_ids:
        if label_id in map_dict:
          label_arr[sample_ind][map_dict[label_id]] = 1
    label_arr_list.append(label_arr)
    num_classes_list.append(len(l_list))
  return num_classes_list, label_arr_list

def trans_labels_BR(org_lables, NUM_CLASSES):
  label_lists_br = [np.zeros(len(org_lables), dtype=np.int64) for i in range(NUM_CLASSES)]
  for sample_ind, label_ids in enumerate(org_lables):
    for label_id in label_ids:
      label_lists_br[label_id][sample_ind] = 1
  return label_lists_br

def trans_labels_bin_classi(org_lables):
  return [np.array([l[0] for l in org_lables], dtype=np.int64)]

def weights_cat(org_lables):
  NUM_CLASSES = len(org_lables[0])
  w_arr = np.empty(NUM_CLASSES)
  scores = np.zeros(NUM_CLASSES)
  for lab_ids in org_lables:
    row_sum = 0
    inds_to_update = []
    for lab_id, lab_val in enumerate(lab_ids):
      if lab_val == 1:
        row_sum += 1
        inds_to_update.append(lab_id)
    for lab_id in inds_to_update:
      # scores[lab_id] += 1
      scores[lab_id] += 1/row_sum
  for i in range(NUM_CLASSES):
    w_arr[i] = len(org_lables)/scores[i]
  return w_arr