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

att_fold_name = "/home/pulkit/research/extented_sexismclassification/results/att_info/hier_fuse_elm_rnn_11+glo_rnn_21_bert_pre_1_di_True_100_234_lstm_100_200_1_False_True_"

base_fname = "/home/pulkit/research/extented_sexismclassification/results/inst/flat_fuse_elm_rnn_1__di_True_100_234_lstm_200_300_1_False_True.txt"
ours_fname = "/home/pulkit/research/extented_sexismclassification/results/inst/hier_fuse~elmo~rnn~11~~glove~rnn~21~~~~~~~~~~bert_pre~1~~~~~0_1_2_3_4_5_6_7_8_9_10_11_12_13+0_1_2_3_4_5_6_7_8_9_10_11_12_13~di~True~100~2_3_4~lstm~300~500~1~False~True.txt"

mis_att_fold_name = "/home/pulkit/research/extented_sexismclassification/results_mis/att_info/flat_fuse~elmo~rnn~1~~~~~~~~~~~~~~bert_pre~1~~~~~lp~True~100~234~lstm~200~500~1~False~True~"
mis_base_fname = "/home/pulkit/research/extented_sexismclassification/results_mis/inst/flat_fuse~elmo~rnn~1~~~~~~~~~~~~~~~~~~~~lp~True~100~234~lstm~100~300~1~False~True.txt"

k_top = 2
min_samples = 10
top_k_freq = 10

def hist_num_label_per_post():
	conf_map = load_map('data/esp_class_maps.txt')
	data_file = 'data/data_trans.csv'
	hist_vecor = np.zeros(10)
	with open(data_file, 'r') as csvfile:
		reader = csv.DictReader(csvfile, delimiter = '\t')
		for row in reader:
			cats = list(set([conf_map['LABEL_MAP'][cat] for cat in row['labels'].split(',')]))
			num_cats = len(cats)
			hist_vecor[num_cats] += 1
	print(hist_vecor)
hist_num_label_per_post()
exit()

def build_post_dict(task, use_saved_data, save_data):
	filename = 'saved/cl_orig_dict~%s.pickle' % task
	if use_saved_data and os.path.isfile(filename):
		print("loading cl_orig_dict")
		with open(filename, 'rb') as f:
			cl_orig_dict = pickle.load(f)
	else:
		if task == 'sexism_classi':
			task_filename = 'data/data_trans.csv'
		elif task == 'misogyny_classi':
			task_filename = 'data/ami_data.txt'
		r_anum = re.compile(r'([^\sa-z0-9.(?)!])+')
		r_white = re.compile(r'[\s.(?)!]+')
		max_words_sent = 35
		cl_orig_dict = {}
		with open(task_filename, 'r') as csvfile:
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
				cl_orig_dict[' '.join(se_list)] = post
		if save_data:
			print("saving cl_orig_dict")
			with open(filename, 'wb') as f:
				pickle.dump(cl_orig_dict, f)

	return cl_orig_dict

def cooccur_ana(att_fold_name, n_test_samp, sent_att_ind, FOR_LMAP, NUM_CLASSES):
	f_summary = open("cooccur_ana.txt", 'w')
	co_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
	co_co_mat = np.zeros((NUM_CLASSES, NUM_CLASSES))
	for i in range(n_test_samp):
		f_name = "%s%d~s%d.json" % (att_fold_name, i, sent_att_ind)
		with open(f_name, 'r') as f:
			sent_att_dict =json.load(f)[0]
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
	f_summary.write("\n------------\n\n")	

	t = sorted(co_co_dict.items(), key = lambda x : x[1], reverse=True)
	for i in range(top_k_freq):
		f_summary.write("%s, %s - %d\n" % (FOR_LMAP[t[i][0][0]], FOR_LMAP[t[i][0][1]], t[i][1]))		
	f_summary.write("\n------------\n\n")	

	f_summary.close()

def write_error_ana(dict_list, conf_map, num_runs, min_num_labs, max_mum_labs, pred_freq_list, f_summary):
	metr_dict = init_metr_dict(conf_map['prob_type'])
	actu_freqs = {num_labs: 0 for num_labs in range(min_num_labs, max_mum_labs+1)}
	pred_freqs = {num_labs: 0 for num_labs in range(min_num_labs, max_mum_labs+1)}
	for num_labs in range(min_num_labs, max_mum_labs+1):
		for n_r in range(num_runs): 
			actu_freqs[num_labs] += len(dict_list[n_r][num_labs]['true_vals'])
			pred_freqs[num_labs] += pred_freq_list[n_r][num_labs]
		actu_freqs[num_labs] = round(actu_freqs[num_labs]/num_runs)
		pred_freqs[num_labs] = round(pred_freqs[num_labs]/num_runs)

		if len(dict_list[0][num_labs]['true_vals']) < min_samples:
			continue 		
		print("number of labels per post: %s" % num_labs)
		f_summary.write("number of labels per post: %s\n" % num_labs)
		for n_r in range(num_runs): 
			metr_dict = calc_metrics_print(dict_list[n_r][num_labs]['pred_vals'], dict_list[n_r][num_labs]['true_vals'], metr_dict, len(conf_map['FOR_LMAP']), conf_map['prob_type'])
		metr_dict = aggregate_metr(metr_dict, num_runs, conf_map['prob_type'])

		write_print_results_no_tsv(metr_dict, f_summary, conf_map['prob_type'])
		f_summary.write("----------------------\n")                                                           
		print("----------------------")                                                           

	f_summary.write("# of labels per post\tfrequency wrt the true labels\tfrequency wrt the predicted labels\t\n")	
	for num_labs in range(min_num_labs, max_mum_labs+1):
		f_summary.write("%d\t%d\t%d\t\n" % (num_labs, actu_freqs[num_labs], pred_freqs[num_labs]))                                                           

def error_ana(ours_fname, base_fname, sent_att_ind, num_runs, conf_map):
	min_num_labs = 1
	max_mum_labs = 7
	header_strings = ["Best proposed method", "Best baseline"]
	f_summary = open("error_ana.txt", 'w')
	for file_ind, filename in enumerate([ours_fname, base_fname]):
		dict_list = []
		pred_freq_list = []
		for n_r in range(num_runs):
			num_lab_dict = {}
			num_lab_freq_dict = {}
			for i in range(min_num_labs, max_mum_labs+1):
				num_lab_dict[i] = {'pred_vals': [], 'true_vals': []}
				num_lab_freq_dict[i] = 0
			dict_list.append(num_lab_dict)
			pred_freq_list.append(num_lab_freq_dict)

		for n_r in range(num_runs):
			with open(filename,'r') as f:
				reader = csv.DictReader(f, delimiter = '\t')
				rows = list(reader)
			for i in range(len(rows)):

				t_cats = [int(st.strip()) for st in rows[i]['actu cats'].split(',')]
				p_cats = [int(st.strip()) for st in rows[i]['pred cats'].split(',')]

				dict_list[n_r][len(t_cats)]['true_vals'].append(t_cats)
				dict_list[n_r][len(t_cats)]['pred_vals'].append(p_cats)
				pred_freq_list[n_r][len(p_cats)] += 1
				# count = count + 1
		f_summary.write("%s\n\n" % header_strings[file_ind])
		print("%s\n" % header_strings[file_ind])
		write_error_ana(dict_list, conf_map, num_runs, min_num_labs, max_mum_labs, pred_freq_list, f_summary)
		f_summary.write("****************************\n")
		print("****************************")

	f_summary.close()

def better_results(att_fold_name, base_fname, model_name, task, use_saved_data, sent_att_ind, k_top, FOR_LMAP, sep_char):
	cl_orig_dict = build_post_dict(task, use_saved_data, True)
	with open(base_fname,'r') as f:
		reader = csv.DictReader(f, delimiter = '\t')
		base_rows = list(reader)
	n_test_samp = len(base_rows)	

	count = 0
	f_summary = open("att_info_2wv_sum_" + task + ".txt", 'w')
	w_summary = csv.DictWriter(f_summary, fieldnames = ['index','text','att words','label','comp pred'], delimiter = '\t')
	w_summary.writeheader()
	for i in range(n_test_samp):
		w_att_dict_list_list = []
		if model_name == 'hier_fuse':
			for w_att_ind in range(sent_att_ind):
				f_name = "%s0/%d%sw%d.json" % (att_fold_name, i, sep_char, w_att_ind)
				with open(f_name, 'r') as f:
					w_att_dict_list_list.append(json.load(f))
			f_name = "%s0/%d%ss%d.json" % (att_fold_name, i, sep_char, sent_att_ind)
			with open(f_name, 'r') as f:
				sent_att_dict =json.load(f)[0]
			num_sentences_capped = len(sent_att_dict['text'])
			labels_multi_hot = sent_att_dict['label']
			prediction_multi_hot = sent_att_dict['prediction']
			post_text = " ".join(sent_att_dict['text'])
		else:
			for w_att_ind in range(sent_att_ind):
				f_name = "%s0/%d%sw%d.json" % (att_fold_name, i, sep_char, w_att_ind)
				with open(f_name, 'r') as f:
					t_list = json.load(f)
					if type(t_list[0]['text']) != list:
						modified_list = []
						for t in t_list:
							t['text'] = t['text'].split(' ')
							modified_list.append(t)
						w_att_dict_list_list.append(modified_list)
					else:
						w_att_dict_list_list.append(t_list)
			num_sentences_capped = 1
			labels_multi_hot = w_att_dict_list_list[0][0]['label']
			prediction_multi_hot = w_att_dict_list_list[0][0]['prediction']
			post_text = " ".join(w_att_dict_list_list[0][0]['text'])

		labels_list_num = di_op_to_label_lists([labels_multi_hot])[0]
		if len(labels_list_num) >= 1 and labels_multi_hot == prediction_multi_hot:# and num_sentences_capped >= 0:
			assert post_text == base_rows[i]['post'].replace("._ ", " ")
			assert sorted(labels_list_num) == sorted([int(st.strip()) for st in base_rows[i]['actu cats'].split(',')])
			comp_list_num = [int(st.strip()) for st in base_rows[i]['pred cats'].split(',')]
			if labels_list_num == comp_list_num:
				continue
			lab_list = [FOR_LMAP[x] for x in labels_list_num]

			comp_list = [FOR_LMAP[x] for x in comp_list_num]
			row = {}
			row['index'] = i
			row['text'] = cl_orig_dict[post_text]
			row['label'] = ','.join(lab_list)
			row['comp pred'] = ','.join(comp_list)

			att_w_list = []
			for si in range(len(w_att_dict_list_list[0])):
				len_min = min(len(w_att_dict_list_list[0][si]['attention']), len(w_att_dict_list_list[0][si]['text']))
				compo_att = np.zeros(len_min)
				for wi in range(len_min):
					for w_vec_ind in range(sent_att_ind):
						compo_att[wi] += w_att_dict_list_list[w_vec_ind][si]['attention'][wi]
				s_indices = np.argsort(-compo_att)
				att_words = [w_att_dict_list_list[0][si]['text'][ii] for ii in s_indices[:k_top]]
				att_w_list.append(', '.join(att_words))
			row['att words'] = '(' + '), ('.join(att_w_list) + ')'

			count = count + 1
			w_summary.writerow(row)
	print(count)
	f_summary.close()


# # better_results(att_fold_name, base_fname, 4, k_top, load_map('data/esp_class_maps.txt')['FOR_LMAP'], '~')
# better_results(att_fold_name, base_fname, 'hier_fuse', 'sexism_classi', False, 2, k_top, load_map('data/esp_class_maps.txt')['FOR_LMAP'], '_')
better_results(mis_att_fold_name, mis_base_fname, 'flat_fuse', 'misogyny_classi', False, 1, k_top, load_map('data/mis_class_maps.txt')['FOR_LMAP'], '~')