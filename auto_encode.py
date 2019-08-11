import time
import sys
import os
import re
import numpy as np
import json
import h5py
from sent_embed import use_embed_posts, bert_embed_posts, infersent_embed_posts
from keras import optimizers
from keras import backend as K
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from loadPreProc import load_config
from nltk import sent_tokenize, word_tokenize
import pickle
import csv

def shortlist_words(num_sample, filename, op_name, data_path):
	raw_fname = data_path + filename
	post_list = []
	w_c_list = []
	for post_org in open(raw_fname):
		post = post_org.rstrip()
		l = len(post.split())
		if l < 7:
			continue
		post_list.append(post)
		w_c_list.append(l)
	arg_arr = np.argsort(np.array(w_c_list))

	s_name = data_path + op_name
	with open(s_name, 'w') as wfile:
		for i, ind in enumerate(arg_arr):
			wfile.write("%s\n" % (post_list[ind]))
			# wfile.write("%d\n" % (w_c_list[ind]))
			if i == num_sample-1:
				break

shortlist_words(70000, 'unlab_minus_lab.txt', 'unlab_minus_lab_shortest_n.txt', 'data/')
exit(1)

def remove_lab_from_unlab(filename, op_name, all_data_file, data_path):
	raw_fname = data_path + filename
	all_data_fname = data_path + all_data_file
	s_name = data_path + op_name
	r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
	r_white = re.compile(r'[\s.(?)!]+')

	lab_id_hash = {}
	lab_content_list = []
	with open(all_data_fname, 'r') as allfile:
		reader = csv.DictReader(allfile, delimiter = '\t')
		for row in reader:
			a_clean = r_white.sub(' ', r_anum.sub('', row['post'].lower())).strip()
			lab_content_list.append(a_clean)
			lab_id_hash[row['post id']] = True

	num_remove = 0
	num_remain = 0
	with open(raw_fname, 'r') as txtfile:
		with open(s_name, 'w') as wfile:
			reader = csv.DictReader(txtfile, delimiter = '\t')
			for row in reader:
				row_clean = r_white.sub(' ', r_anum.sub('', row['post'].lower())).strip()
				present_flag = False
				for lab_post in lab_content_list:
					if lab_post in row_clean:
						present_flag = True
						break
				if present_flag or row['id'][5:] in lab_id_hash:
					num_remove += 1
				else:
					wfile.write("%s\n" % row['post'])
					num_remain += 1
	print("removed: %s, remaining %s" % (num_remove, num_remain))

remove_lab_from_unlab('unlab_data_postids.csv', 'unlab_minus_lab.txt', 'data.csv', 'data/')
exit(1)

def remove_test_from_unlab(filename, op_name, fin_data_name, all_data_file, data_path, save_path):
	raw_fname = data_path + filename
	all_data_fname = data_path + all_data_file
	s_name = data_path + op_name
	data_fname = save_path + fin_data_name
	r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
	r_white = re.compile(r'[\s.(?)!]+')

	with open(data_fname, 'rb') as f_data:
		data_dict = pickle.load(f_data)
	# removing validation and test data
	test_data = data_dict['text'][data_dict['test_st_ind']:]

	all_id_dict = {}
	with open(all_data_fname, 'r') as allfile:
		reader = csv.DictReader(allfile, delimiter = '\t')
		for row in reader:
			a_clean = r_white.sub(' ', r_anum.sub('', row['post'].lower())).strip()
			all_id_dict[a_clean] = row['post id']
	print('all_id_dict done')

	test_id_dict= {}
	for t_post in test_data:
		test_id_dict[all_id_dict[t_post]] = True
	print('test_id_dict done')

	num_remove = 0
	num_remain = 0
	with open(raw_fname, 'r') as txtfile:
		with open(s_name, 'w') as wfile:
			reader = csv.DictReader(txtfile, delimiter = '\t')
			for row in reader:
				# print(row)
				# if "seeing online articles being published" in row['post']:
				# 	print(row)
				row_clean = r_white.sub(' ', r_anum.sub('', row['post'].lower())).strip()
				test_flag = False
				for test_post in test_data:
					if test_post in row_clean:
						test_flag = True
						break
				if test_flag or row['id'][5:] in test_id_dict:
					num_remove += 1
				else:
					wfile.write("%s\n" % row['post'])
					num_remain += 1
	print("removed: %s, remaining %s" % (num_remove, num_remain))
	print("test data size: %s" % (len(test_data)))

# remove_test_from_unlab('unlab_data_postids.csv', 'unlab_sans_test.txt', 'data_0.15_0.15_22_35_False.pickle', 'data.csv', 'data/', 'saved/')
# exit(1)

def bert_pretraining_data(filename, data_path, save_path):
	max_seq_len = 1000
	raw_fname = data_path + filename
	s_name = save_path + filename[:-4] + '_bert_pre.txt'
	if os.path.isfile(s_name):
		print("already exists")
	else:
		with open(raw_fname, 'r') as txtfile:
			with open(s_name, 'w') as wfile:
				post_cnt = 0
				for post in txtfile.readlines():
					list_sens = []
					post_has_big_sens = False
					for se in sent_tokenize(post):
						if len(word_tokenize(se)) > max_seq_len:
							post_has_big_sens = True
							break
						list_sens.append(se)
					if post_has_big_sens:
						continue

					if post_cnt > 0:
						wfile.write("\n")
					for se in list_sens:
						wfile.write("%s\n" % se)

					post_cnt += 1
		print("saved %d bert pretraining data" % post_cnt)
# bert_pretraining_data('unlab_sans_test.txt', "data/", "../bert/tmp/")
# exit(1)
# screen -L python create_pretraining_data.py \
#   --input_file=tmp/unlab_sans_test_bert_pre.txt \
#   --output_file=tmp/tf_examples.tfrecord \
#   --vocab_file=../bert/uncased_L-12_H-768_A-12/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=120 \
#   --max_predictions_per_seq=18 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=5

# screen -L python run_pretraining.py \
#   --input_file=tmp/tf_examples.tfrecord \
#   --output_dir=tmp/pretraining_output \
#   --do_train=True \
#   --do_eval=True \
#   --bert_config_file=../bert/uncased_L-12_H-768_A-12/bert_config.json \
#   --init_checkpoint=../bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
#   --train_batch_size=25 \
#   --max_seq_length=120 \
#   --max_predictions_per_seq=18 \
#   --num_train_steps=100000 \
#   --num_warmup_steps=10000 \
#   --learning_rate=2e-5

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.setrecursionlimit(10000)
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def batch_add(sen, all_sens, batch_sens, batch_size_data):
	if len(batch_sens) == batch_size_data:
		all_sens.append(batch_sens)
		batch_sens = [sen]
	else:
		batch_sens.append(sen)
	return batch_sens, all_sens

def load_sen_data(filename, data_path, save_path, max_words_sent, batch_size_data, use_saved_sent_feats):
	raw_fname = data_path + filename
	s_name = save_path + filename[:-4] + '.pickle'
	if use_saved_sent_feats and os.path.isfile(s_name):
		print("loading unlabled sens")
		with open(s_name, 'r') as f:
			all_sens, num_samps = json.load(f)
	else:
		r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
		r_white = re.compile(r'[\s.(?)!]+')
		all_sens = []
		batch_sens = []
		with open(raw_fname, 'r') as txtfile:
			for post in txtfile.readlines():
				for se in sent_tokenize(post):
					se_cl = r_white.sub(' ', r_anum.sub('', str(se).lower())).strip()
					if se_cl == "":
						continue
					words = se_cl.split(' ')
					while len(words) > max_words_sent:
						batch_sens, all_sens = batch_add(' '.join(words[:max_words_sent]), all_sens, batch_sens, batch_size_data)
						words = words[max_words_sent:]
					batch_sens, all_sens = batch_add(' '.join(words), all_sens, batch_sens, batch_size_data)
		num_samps = batch_size_data*len(all_sens)+len(batch_sens)
		all_sens.append(batch_sens)
		print("saving unlabled sens")
		with open(s_name, 'w') as f:
			json.dump([all_sens, num_samps], f)
	return all_sens, num_samps

def calc_layer_widths(layer_percs, emb_dim):
	layer_widths = []
	for layer_perc in layer_percs:
		layer_widths.append(np.rint(layer_perc*emb_dim).astype(int))
	return layer_widths

def gen_sent_feats(feat_name, data_sen, emb_dim, use_saved_sent_feats, save_sent_feats, data_fold_path, save_fold_path, batch_size_data, num_samps):
	print("computing %s unlabeled sent feats" % feat_name)
	s_filename = ("%sunlab_sent_feat~%s.h5" % (save_fold_path, feat_name))
	if use_saved_sent_feats and os.path.isfile(s_filename):
		print("loading %s unlabeled sent feats" % feat_name)
		with h5py.File(s_filename, "r") as hf:
			feats = hf['feats'][:]
	else:
		if feat_name == 'bert':
			feats = bert_embed_posts(data_sen, batch_size_data, emb_dim, data_fold_path)
		elif feat_name == 'use':
			feats = use_embed_posts(data_sen, batch_size_data, emb_dim, data_fold_path)
		elif feat_name == 'infersent':
			feats = inferSent_embed_posts(data_sen, batch_size_data, emb_dim, data_fold_path)
		feats = np.reshape(feats, (-1, emb_dim))[:num_samps]
		if save_sent_feats:
			print("saving %s unlabeled sent feats" % feat_name)
			with h5py.File(s_filename, "w") as hf:
				hf.create_dataset('feats', data=feats)
	print(feats.shape)
	return feats

conf_dict_list, conf_dict_com = load_config(sys.argv[1])
data_sen, num_samps = load_sen_data(conf_dict_com['filename'], conf_dict_com['data_folder_name'], conf_dict_com['save_folder_name'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com['BATCH_SIZE_DATA'], conf_dict_com['use_saved_sent_feats'])
print("num of unlab sens: %s" % num_samps)

for conf_dict in conf_dict_list:
	for feat_name, layer_percs in conf_dict['layer_info']:
		feats = gen_sent_feats(feat_name, data_sen, conf_dict_com['poss_sent_feats_emb_dict'][feat_name], conf_dict_com['use_saved_sent_feats'], conf_dict_com['save_sent_feats'], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['BATCH_SIZE_DATA'], num_samps)
		layer_widths = calc_layer_widths(layer_percs, conf_dict_com['poss_sent_feats_emb_dict'][feat_name])
		for loss_func in conf_dict['loss_funcs']:
			for drop1 in  conf_dict['drop1']:
				for drop2 in  conf_dict['drop2']:
					startTime = time.time()
					info_str = "feat: %s, layer_percs = %s, loss_func = %s, drop1 = %s, drop2 = %s" % (feat_name,layer_percs,loss_func,drop1,drop2)
					fname_mod = "%smod_aut~%s+%s+%s+%s+%s.h5" % (conf_dict_com['save_folder_name'], feat_name,'_'.join([str(x) for x in layer_percs]),loss_func,drop1,drop2)
					print(info_str)

					if conf_dict_com['use_saved_model'] and os.path.isfile(fname_mod):
						print("Model file exists")
					else:
						h = input_vec = Input(shape=(conf_dict_com['poss_sent_feats_emb_dict'][feat_name],))
						h = Dropout(drop1)(h)
						for layer_width in layer_widths:
							h = Dense(layer_width, activation='relu')(h)
						encoder = Model(input_vec, h)

						for layer_width in reversed(layer_widths[:-1]):
							h = Dense(layer_width, activation='relu')(h)
						h = Dropout(drop2)(h)
						h = Dense(conf_dict_com['poss_sent_feats_emb_dict'][feat_name])(h)

						autoencoder = Model(input_vec, h)
						adam = optimizers.Adam(lr=conf_dict_com['LEARN_RATE'])
						autoencoder.compile(loss=loss_func, optimizer=adam)

						autoencoder.summary()
						autoencoder.fit(feats, feats, epochs=conf_dict_com["EPOCHS"], shuffle=True, batch_size=conf_dict_com['BATCH_SIZE_AUTO'], verbose=1)
						encoder.save(fname_mod)

						timeLapsed = int(time.time() - startTime + 0.5)
						hrs = timeLapsed/3600.
						t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))
						print(t_str)

						K.clear_session()
						# set_session(tf.set_sessionn(config=config))
					print("encod dim: %d" % layer_widths[-1])                

