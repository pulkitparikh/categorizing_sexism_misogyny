import json
import csv
from os import listdir
from os.path import isfile, join
from loadPreProc import di_op_to_label_lists, load_map
from evalMeasures import *
import numpy as np
import re
from nltk import sent_tokenize
import pickle

# att_fold_name = "/home/pulkit/research/sexismclassification/results/att_info/hier_fuse_elm_rnn_11+glo_rnn_21+lin_rnn_31_bert_pre_1_di_True_100_234_lstm_300_600_1_False_True_0/"

att_fold_name = "/home/pulkit/research/extented_sexismclassification/results/att_info/hier_fuse_elm_rnn_11+glo_rnn_21_bert_pre_1_di_True_100_234_lstm_100_200_1_False_True_"
# att_fold_name = "/home/pulkit/research/extented_sexismclassification/results/att_info/hier_fuse~elmo~rnn~11~~glove~rnn~21~~fasttext~rnn~31~~ling~rnn~41~~bert_pre~1~use~1~infersent~1~0_1_2_3_4_5_6_7_8_9_10_11_12_13~di~True~100~2_3_4~lstm~200~400~1~False~True~"
att_fold_best_name = "/home/pulkit/research/extented_sexismclassification/results/att_info/hier_fuse~elmo~rnn~11~~glove~rnn~21~~~~~~~~~~bert_pre~1~~~~~0_1_2_3_4_5_6_7_8_9_10_11_12_13+0_1_2_3_4_5_6_7_8_9_10_11_12_13~di~True~100~2_3_4~lstm~300~500~1~False~True~"
base_fname = "/home/pulkit/research/extented_sexismclassification/results/inst/flat_fuse_elm_rnn_1__di_True_100_234_lstm_200_300_1_False_True.txt"

n_test_samp = 1953
# word_1_att_ind = 0
# word_2_att_ind = 1
# sent_att_ind = 2
# thresh = 0.5
k_top = 2
min_samples = 10
top_k_freq = 10

conf_map = load_map('data/esp_class_maps.txt')

def build_post_dict(use_saved_data, save_data):
	filename = 'saved/cl_orig_dict.pickle'
	if use_saved_data and os.path.isfile(filename):
		print("loading cl_orig_dict")
		with open(filename, 'rb') as f:
			cl_orig_dict = pickle.load(f)
	else:
		r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
		r_white = re.compile(r'[\s.(?)!]+')
		max_words_sent = 35
		cl_orig_dict = {}
		with open('data/data_trans.csv', 'r') as csvfile:
			reader = csv.DictReader(csvfile, delimiter = '\t')
			for row in reader:
				post = str(row['post'])

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
				cl_orig_dict['***'.join(se_list)] = post
		if save_data:
			print("saving cl_orig_dict")
			with open(filename, 'wb') as f:
				pickle.dump(cl_orig_dict, f)

	return cl_orig_dict

def cooccur_ana(att_fold_name, n_test_samp, sent_att_ind, FOR_LMAP, NUM_CLASSES):
	f_summary = open("cooccur_ana.txt", 'w')
	co_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
	co_co_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
	# w_summary = csv.DictWriter(f_summary, fieldnames = ['index','text','att words','label','comp pred'], delimiter = '\t')
	# w_summary.writeheader()
	for i in range(n_test_samp):
		f_name = "%s%d~s%d.json" % (att_fold_name, i, sent_att_ind)
		with open(f_name, 'r') as f:
			sent_att_dict =json.load(f)[0]
		# if sent_att_dict['label'] == sent_att_dict['prediction']:
		# 	continue
		t_cats = di_op_to_label_lists([sent_att_dict['label']])[0]
		p_cats = di_op_to_label_lists([sent_att_dict['prediction']])[0]

		for i in range(len(t_cats)-1):
			for j in range(i+1, len(t_cats)):
				co_mat[t_cats[i], t_cats[j]] += 1
				if t_cats[i] in p_cats and t_cats[j] in p_cats:
					co_co_mat[t_cats[i], t_cats[j]] += 1

	co_dict = {}
	co_co_dict = {}			
	for i in range(NUM_CLASSES-1):
		for j in range(i+1, NUM_CLASSES):
			co_dict[(i,j)] = co_mat[i,j] + co_mat[j,i]
			co_co_dict[(i,j)] = co_co_mat[i,j] + co_co_mat[j,i]

	t = sorted(co_dict.items(), key = lambda x : x[1], reverse=True)
	for i in range(top_k_freq):
		f_summary.write("%s, %s - %d\n" % (FOR_LMAP[t[i][0][0]], FOR_LMAP[t[i][0][1]], t[i][1]))		
		# f_summary.write("%s, %s\n" % (FOR_LMAP[t[i][0][0]], FOR_LMAP[t[i][0][1]]))		
	f_summary.write("\n------------\n\n")	

	t = sorted(co_co_dict.items(), key = lambda x : x[1], reverse=True)
	for i in range(top_k_freq):
		# f_summary.write("%s, %s\n" % (FOR_LMAP[t[i][0][0]], FOR_LMAP[t[i][0][1]]))		
		f_summary.write("%s, %s - %d\n" % (FOR_LMAP[t[i][0][0]], FOR_LMAP[t[i][0][1]], t[i][1]))		
	f_summary.write("\n------------\n\n")	

	f_summary.close()
# cooccur_ana(att_fold_name, n_test_samp, sent_att_ind, conf_map['FOR_LMAP'], conf_map['FOR_LMAP'], len(conf_map['FOR_LMAP']))
# exit(1)

def error_ana(att_fold_name, n_test_samp, sent_att_ind, num_runs, conf_map):
	min_num_labs = 1
	max_mum_labs = 7
	dict_list = []
	for n_r in range(num_runs):
		num_lab_dict = {}
		for i in range(min_num_labs, max_mum_labs+1):
			num_lab_dict[i] = {'pred_vals': [], 'true_vals': []}
		dict_list.append(num_lab_dict)
	print("dict list loaded")
	# count = 0
	f_summary = open("error_ana.txt", 'w')
	# w_summary = csv.DictWriter(f_summary, fieldnames = ['index','text','att words','label','comp pred'], delimiter = '\t')
	# w_summary.writeheader()
	for n_r in range(num_runs):
		for i in range(n_test_samp):
			f_name = "%s%d/%d~s%d.json" % (att_fold_name, n_r, i, sent_att_ind)
			with open(f_name, 'r') as f:
				sent_att_dict =json.load(f)[0]
			# if sent_att_dict['label'] == sent_att_dict['prediction']:
			# 	continue
			t_cats = di_op_to_label_lists([sent_att_dict['label']])[0]
			p_cats = di_op_to_label_lists([sent_att_dict['prediction']])[0]

			dict_list[n_r][len(t_cats)]['true_vals'].append(t_cats)
			dict_list[n_r][len(t_cats)]['pred_vals'].append(p_cats)
			# count = count + 1

	metr_dict = init_metr_dict(conf_map['prob_type'])
	for num_labs in range(min_num_labs, max_mum_labs+1):
		if len(dict_list[0][num_labs]['true_vals']) < min_samples:
			continue 		
		print(num_labs)
		f_summary.write("%s\n" % num_labs)
		for n_r in range(num_runs): 
			metr_dict = calc_metrics_print(dict_list[n_r][num_labs]['pred_vals'], dict_list[n_r][num_labs]['true_vals'], metr_dict, len(conf_map['FOR_LMAP']), conf_map['prob_type'])
		metr_dict = aggregate_metr(metr_dict, num_runs, conf_map['prob_type'])

		write_print_results_no_tsv(metr_dict, f_summary, conf_map['prob_type']) 
		f_summary.write("----------------------\n")                                                           

	# c = 0		
	# for num_lab, v in num_lab_dict.items():	
	# 	if len(v['pred_vals']) < min_samples:
	# 		continue
	# 	print(num_lab)
	# 	print(len(v['pred_vals']))
	# 	f_summary.write("%s\n" % num_lab)
	# 	metr_dict = init_metr_dict()
	# 	metr_dict = calc_metrics_print(v['pred_vals'], v['true_vals'], metr_dict)
	# 	metr_dict = aggregate_metr(metr_dict, 1)

	# 	write_results(metr_dict, f_summary) 
	# 	print("Precision Instance: %.3f" % metr_dict['avg_pi'])
	# 	print("Recall Instance: %.3f" % metr_dict['avg_ri'])
	# 	print("----------------------")                                                           
	# 	f_summary.write("----------------------\n")                                                           
	# 	c += len(v['pred_vals'])

	# print(count)
	# print(c)
	f_summary.close()
# error_ana(att_fold_best_name, n_test_samp, 2, 3, conf_map)

def better_results(att_fold_name, base_fname, n_test_samp, sent_att_ind, k_top, FOR_LMAP, sep_char):
	cl_orig_dict = build_post_dict(True, True)
	# base_dict = {}
	with open(base_fname,'r') as f:
		reader = csv.DictReader(f, delimiter = '\t')
		base_rows = list(reader)
		# for row in reader:
		# 	if row['post'] in base_dict:
		# 		print('problem')
		# 	else:
		# 		base_dict[row['post']] = [int(st.strip()) for st in row['pred cats'].split(',')]

	count = 0
	f_summary = open("att_info_2wv_avg.txt", 'w')
	w_summary = csv.DictWriter(f_summary, fieldnames = ['index','text','att words','label','comp pred'], delimiter = '\t')
	w_summary.writeheader()
	for i in range(n_test_samp):
		w_att_dict_list_list = []
		for w_att_ind in range(sent_att_ind):
			f_name = "%s0/%d%sw%d.json" % (att_fold_name, i, sep_char, w_att_ind)
			with open(f_name, 'r') as f:
				w_att_dict_list_list.append(json.load(f))
		f_name = "%s0/%d%ss%d.json" % (att_fold_name, i, sep_char, sent_att_ind)
		with open(f_name, 'r') as f:
			sent_att_dict =json.load(f)[0]
			# print(sent_att_dict['text'])
			# print(di_op_to_label_lists([sent_att_dict['prediction']])[0])
			# print("***")
			# print(di_op_to_label_lists([sent_att_dict['label']])[0])
			# print(base_dict["._ ".	join(sent_att_dict['text'])])
			# "--------------"
			# input()
		# if "being told i should take cat calls as compliments by my father"	not in sent_att_dict['text'][0]:
		# 	continue

		if len(di_op_to_label_lists([sent_att_dict['label']])[0]) >= 1 and len(sent_att_dict['text']) >= 1 and sent_att_dict['label'] == sent_att_dict['prediction']:
			assert "._ ".join(sent_att_dict['text']) == base_rows[i]['post']
			# pred_list_num = di_op_to_label_lists([sent_att_dict['prediction']])[0]
			comp_list_num = [int(st.strip()) for st in base_rows[i]['pred cats'].split(',')]
			# comp_list_num = base_dict["._ ".join(sent_att_dict['text'])]
			lab_list_num = di_op_to_label_lists([sent_att_dict['label']])[0]
			if lab_list_num == comp_list_num:
				continue
			lab_list = [FOR_LMAP[x] for x in lab_list_num]

			if 'Body shaming' not in lab_list:
				continue

			comp_list = [FOR_LMAP[x] for x in comp_list_num]
			row = {}
			row['index'] = i
			row['text'] = cl_orig_dict['***'.join(sent_att_dict['text'])]
			row['label'] = ','.join(lab_list)
			row['comp pred'] = ','.join(comp_list)

			# print(row['text'])
			att_w_list = []
			for si in range(len(w_att_dict_list_list[0])):
				# print(w_att_dict_list_list[0][si]['text'])
				# print(w_att_dict_list_list[0][si]['attention'])
				# print(w_att_dict_list_list[1][si]['attention'])

				compo_att = np.zeros(len(w_att_dict_list_list[0][si]['text']))
				for wi in range(len(w_att_dict_list_list[0][si]['text'])):
					for w_vec_ind in range(sent_att_ind):
						compo_att[wi] += w_att_dict_list_list[w_vec_ind][si]['attention'][wi]
						# compo_att[wi] = max(compo_att[wi], w_att_dict_list_list[w_vec_ind][si]['attention'][wi])
				s_indices = np.argsort(-compo_att)
				# print(type(s_indices))
				# print(s_indices[:k_top])
				# print(w1_att_dict['text'])
				att_words = [w_att_dict_list_list[0][si]['text'][ii] for ii in s_indices[:k_top]]
				att_w_list.append(', '.join(att_words))
				# print(att_words)
			row['att words'] = '(' + '), ('.join(att_w_list) + ')'
			# att_inds = [x[0] for x in enumerate(w1_att_dict['attention']) if x[1] > thresh]
			# print(att_inds)

			count = count + 1
			w_summary.writerow(row)
	print(count)
	f_summary.close()
# better_results(att_fold_name, base_fname, n_test_samp, 4, k_top, conf_map['FOR_LMAP'], '~')
better_results(att_fold_name, base_fname, n_test_samp, 2, k_top, conf_map['FOR_LMAP'], '_')
# for clust_ind, att_arr in enumerate(mod_op_list[0][1]):
#     att_list = []
#     if len(att_arr.shape) == 3:
#         fname_att = ("%satt_info/%d/%d~w%d.json" % (save_folder_name, ind, ind, clust_ind))
#         for ind_sen, sen in enumerate(post):
#             my_sen_dict = {}
#             my_sen_dict['text'] = sen.split(' ')
#             my_sen_dict['label'] = true_vals_multi_hot[ind].tolist()
#             my_sen_dict['prediction'] = y_pred_list[0][ind].tolist()
#             my_sen_dict['posterior'] = mod_op_list[0][0][ind].tolist()
#             my_sen_dict['attention'] = att_arr[ind, ind_sen, :].tolist()
#             my_sen_dict['id'] = "%d~w%d~%d" % (ind, clust_ind, ind_sen)
#             att_list.append(my_sen_dict)
#     else:
#         fname_att = ("%satt_info/%d/%d~s%d.json" % (save_folder_name, ind, ind, clust_ind))
#         my_sen_dict = {}
#         my_sen_dict['text'] = post
#         my_sen_dict['label'] = true_vals_multi_hot[ind].tolist()
#         my_sen_dict['prediction'] = y_pred_list[0][ind].tolist()
#         my_sen_dict['posterior'] = mod_op_list[0][0][ind].tolist()
#         my_sen_dict['attention'] = att_arr[ind, :].tolist()
#         my_sen_dict['id'] = "%d~s%d" % (ind, clust_ind)
#         att_list.append(my_sen_dict)

