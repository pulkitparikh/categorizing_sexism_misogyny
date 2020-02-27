import sys
import csv
# f_name1 = "mul_esp.tsv"
# f_name1 = "det_mis.tsv"
f_name1 = "cla_mis.tsv"
f_path = 'results/latex/'

wf_dict = {'elmo': 'ELMo', 'glove': 'GloVe', 'fasttext': 'fastText', 'ling': 'Ling'}
sf_dict = {'bert_pre': 'tBERT', 'use': 'USE', 'infersent': 'InferSent', 'bert': 'BERT', 'bert_0.9_0.8_0.7_0.6_0.5_mse_0.25_0.25': 'BERT\_0.9\_0.8\_0.7\_0.6\_0.5', 'bert_0.9_0.8_0.7_mse_0.25_0.25': 'BERT\_0.9\_0.8\_0.7', 'bert_0.95_0.9_0.85_0.8_0.75_mse_0.25_0.25': 'BERT\_0.95\_0.9\_0.85\_0.8\_0.75', 'bert_0.95_0.9_0.85_mse_0.25_0.25': 'BERT\_0.95\_0.9\_0.85'}
w_alias_dict = {'elm': 'elmo', 'glo': 'glove', 'fas': 'fasttext', 'lin': 'ling'}
alg_dict = {'cnn': 'c', 'rnn': 'l', 'c_rnn': 'cl'}
h_baseline_dict = {'sh(wch(ELMo))': 'CNN-biLSTM-Attention', 'sh(wlh(ELMo))': 'Hierarchical-biLSTM-Attention', 'InferSent': 'InferSent-biLSTM-Attention', 'BERT': 'BERT-biLSTM-Attention', 'USE': 'USE-biLSTM-Attention'}
f_baseline_dict = {'wclf(ELMo)': 'C-biLSTM', 'wcf(ELMo)': 'CNN-Kim', 'wlf(ELMo)': 'biLSTM', 'InferSent': 'InferSent', 'BERT': 'BERT', 'USE': 'USE'}

max_num_word_feats = 4
max_num_word_attributes = 4
max_num_sent_enc_feats = 3
max_num_sent_enc_attributes = 2

def load_val_data(filename):
	with open(filename, 'r') as res_tsv:
		reader = csv.DictReader(res_tsv, delimiter = '\t')
		rows = list(reader)

	model_dic = {}
	for row in rows:
		if row['model'] == "#####":
			continue
		att_d = int(row['att dim'])	
		att_flag = True if att_d > 0 else False 
		classi_probs_labels_val = row['classi probs labels'] if 'classi probs labels' in row else ''
		model_key = (row['model'], row['word feats'], row['sent feats'], classi_probs_labels_val, row['trans'], row['class imb'].lower(), att_flag)
		if model_key in model_dic:
			model_dic[model_key][0].append(row['cnn filters'])
			model_dic[model_key][1].append(row['max pool k'])
			model_dic[model_key][2].append(row['rnn dim'])
			model_dic[model_key][3].append(row['att dim'])
		else:
			model_dic[model_key] = [[row['cnn filters']], [row['max pool k']], [row['rnn dim']], [row['att dim']]]

	val_stats_dict = {}		
	for key, list_list in model_dic.items():
		# print(key)
		val_stats_dict[key] = {}
		val_stats_dict[key]['cnn filters'] = len(set(list_list[0]))
		val_stats_dict[key]['max pool k'] = len(set(list_list[1]))
		val_stats_dict[key]['rnn dim'] = len(set(list_list[2]))
		val_stats_dict[key]['att dim'] = len(set(list_list[3]))
	return val_stats_dict

def gen_hyper_str(model_str, word_feats_str, sent_feats_str, classi_probs_labels_str, trans_str, class_imb_str, att_dim_sr, cnn_filters_str, max_pool_k_str, rnn_dim_str, att_dim_str):
	att_d = int(att_dim_str)	
	att_flag = True if att_d > 0 else False 
	model_key = (model_str, word_feats_str, sent_feats_str, classi_probs_labels_str, trans_str, class_imb_str.lower(), att_flag)
	if model_key not in val_stats_dict:
		return 'N.A.', 'N.A.', 'N.A.', 'N.A.'
	cnn_fil = cnn_filters_str if val_stats_dict[model_key]['cnn filters'] > 1 else 'N.A.'
	max_pool_k = max_pool_k_str if val_stats_dict[model_key]['max pool k'] > 1 else 'N.A.'
	rnn_dim = rnn_dim_str if val_stats_dict[model_key]['rnn dim'] > 1 else 'N.A.'
	att_dim = att_dim_str if val_stats_dict[model_key]['att dim'] > 1 else 'N.A.'
	return cnn_fil, max_pool_k, rnn_dim, att_dim

def split_word_feats(word_feats):
	if '~' in word_feats:
		return word_feats.split('~')
	arr = word_feats.split('+')
	wf_list = []
	if arr != ['']:
		for entry in arr:
			a = entry.split('_')
			wf_list.extend([w_alias_dict[a[0]], a[1], a[2], ''])
	wf_list.extend([''] * (max_num_word_feats* max_num_word_attributes - len(wf_list)))
	return wf_list

def split_sent_feats(sent_feats):
	if '~' in sent_feats:
		return sent_feats.split('~')
	arr = sent_feats.split('+')
	sf_list = []
	if arr != ['']:
		for entry in arr:
			sf_list.extend([entry[:-2], entry[-1]])
	sf_list.extend([''] * (max_num_sent_enc_feats* max_num_sent_enc_attributes - len(sf_list)))
	return sf_list

def trans_notation(model, word_feats, sent_feats, att_dim):
	out_str = ''
	wf_list = split_word_feats(word_feats)
	# print(wf_list)
	w_clusts = {str(i): [] for i in range(1, max_num_word_feats+1, 1)}
	for i in range(0, max_num_word_feats*max_num_word_attributes, max_num_word_attributes):
		if wf_list[i] != '':
			w_clusts[wf_list[i+2][0]].append(i)
	s_clusts = {str(i): [] for i in range(1, max_num_sent_enc_feats+1, 1)}		
	for k,v_list in w_clusts.items():
		if v_list == []:
			continue
		w_str = ''
		for w_ind in v_list:
			w_str += (wf_dict[wf_list[w_ind]] + ', ')
		w_ind = v_list[0]
		if model == 'hier_fuse':
			if wf_list[w_ind+1].startswith('sep'):
				s_clusts[wf_list[w_ind + 2][1]].append("wch(%s)" % (w_str[:-2]))
				s_clusts[wf_list[w_ind + 2][2]].append("wlh(%s)" % (w_str[:-2]))
			else:
				s_clusts[wf_list[w_ind + 2][1]].append("w%sh(%s)" % (alg_dict[wf_list[w_ind + 1]], w_str[:-2]))
		else:
			if wf_list[w_ind+1].startswith('com'):
				out_str += "wcf(%s), " % (w_str[:-2])
				out_str += "wlf(%s), " % (w_str[:-2])
			else:
				out_str += "w%sf(%s), " % (alg_dict[wf_list[w_ind + 1]], w_str[:-2])
	sf_list = split_sent_feats(sent_feats)
	# print(sf_list)
	for i in range(0, max_num_sent_enc_feats*max_num_sent_enc_attributes, max_num_sent_enc_attributes):
		if sf_list[i] != '':
			if model == 'hier_fuse':
				s_clusts[sf_list[i+1]].append(sf_dict[sf_list[i]])
			else:
				out_str += ("%s, " % sf_dict[sf_list[i]])	
	if model == 'hier_fuse':
		for k,v_list in s_clusts.items():
			if v_list == []:
				continue
			s_cl = ''
			for s in v_list:
				s_cl += (s + ', ')
			out_str += "sh(%s), " % s_cl[:-2]
		if out_str[:-2] in h_baseline_dict:
			return h_baseline_dict[out_str[:-2]], True
		else:
			return out_str[:-2], False
	else:
		if out_str[:-2] in f_baseline_dict:
			b = f_baseline_dict[out_str[:-2]]
			if b == "biLSTM" and att_dim != '0':
				b += "-Attention"
			return b, True
		else:
			return out_str[:-2], False

with open(f_path + f_name1, 'r') as res_tsv:
	reader = csv.DictReader(res_tsv, delimiter = '\t')
	rows = list(reader)

if sys.argv[1] == 'hyper':
	val_stats_dict = load_val_data(f_path + 'val_' + f_name1)
	with open(f_path + 'hyper_start_table.txt', 'r') as f:
		start_lines = f.readlines()
else:
	with open(f_path + f_name1[:-4] + '_start_table.txt', 'r') as f:
		start_lines = f.readlines()

no_b_start_lines = start_lines.copy()
no_b_start_lines[6] = no_b_start_lines[6].replace('|c|p{', '|p{')
no_b_start_lines[8] = no_b_start_lines[8][1:]

with open(f_path + 'end_table.txt', 'r') as f:
	end_lines = f.readlines()

dict_d = {('di', True, True, False): [], ('di', False, True, False): [], ('dc', True, True, False): [], ('lp', True, False, False): [], ('lp', True, True, False): [], ('br', True, True, False): [], ('di', True, True, True): []}
for row in rows:
	auto_enc_flag = True if '_mse_' in row['sent feats'] else False
	key = (row['trans'], ('classi probs labels' not in row or row['classi probs labels'] == '0_1_2_3_4_5_6_7_8_9_10_11_12_13'), row['class imb'].lower() == 'true', auto_enc_flag)
	dict_d[key].append(row)

if f_name1.startswith("det_mis"):
	cline_str = "\cline{2-4}\n"
else:
	cline_str = "\cline{2-6}\n"

for key, val_list in dict_d.items():
	if val_list == []:
		continue
	key_str = "_".join([str(x) for x in key])
	f_name = ("%s%s_%s_%s.txt" % (f_path, f_name1[:-4], key_str, sys.argv[1]))
	with open(f_name, 'w') as f_fin:
		base_list = []
		prop_list = []
		f_fin.write("%%%s\n" % key_str)
		# input()
		for ind, row in enumerate(val_list):
			# print(row)
			# print(trans_notation(row['model'], row['word feats'], row['sent feats']))
			# input()
			meth_name, is_baseline = trans_notation(row['model'], row['word feats'], row['sent feats'], row['att dim'])
			if key[1] == True:
				mod_id = meth_name
				# ("%s_%s_%s" % (row['model'], row['word feats'], row['sent feats']))
			else:
				mod_id = ("%s %s" % (meth_name, row['classi probs labels'].replace('_', '\_').replace('+', ' + ')))

			if sys.argv[1] == 'hyper':
				classi_probs_labels_val = row['classi probs labels'] if 'classi probs labels' in row else ''
				# print(row)
				cnn_fil, max_pool_k, rnn_dim, att_dim = gen_hyper_str(row['model'], row['word feats'], row['sent feats'], classi_probs_labels_val, row['trans'], row['class imb'], row['att dim'], row['cnn filters'], row['max pool k'], row['rnn dim'], row['att dim'])
				st = ("%s & %s & %s & %s & %s\\\\\n" % (mod_id, rnn_dim, att_dim, cnn_fil, max_pool_k))
			else:	
				if f_name1.startswith("mul_esp"):
					st = ("%s & %.3f & %.3f & %.3f & %.3f\\\\\n" % (mod_id, float(row['f1-Inst']), float(row['f1-Macro']), float(row['Jaccard']), float(row['f1-Micro'])))
				elif f_name1.startswith("cla_mis"):
					st = ("%s & %.3f & %.3f & %.3f & %.3f\\\\\n" % (mod_id, float(row['f-We']), float(row['f-Ma']), float(row['acc']), float(row['f-Mi'])))
				elif f_name1.startswith("det_mis"):
					st = ("%s & %.3f & %.3f\\\\\n" % (mod_id, float(row['f']), float(row['acc'])))

			if is_baseline:
				base_list.append(st)
			else:
				prop_list.append(st)

		if base_list != []:
			f_fin.writelines(start_lines)
			# f_fin.write('\\begin{tabular}{|c|p{2.8in}|c|c|c|c|}\n\hline\n&Approach & $F_I$ & $F_{macro}$ & $Acc_I$ & $F_{micro}$\\\n\hline\n\hline\n')
			f_fin.write("\multirow{%s}{*}{Baselines} " % len(base_list))
			for i in range(len(base_list)-1):
				f_fin.write('& ' + base_list[i])
				f_fin.write(cline_str)	
			f_fin.write('& ' + base_list[len(base_list)-1])
			f_fin.write("\hline\n")	
			f_fin.write("\multirow{%s}{*}{Proposed methods} " % len(prop_list))
			s_add = '& '
			st_line = cline_str
		else:
			f_fin.writelines(no_b_start_lines)
			# f_fin.write('\begin{tabular}{p{2.8in}|c|c|c|c|}\n\hline\nApproach & $F_I$ & $F_{macro}$ & $Acc_I$ & $F_{micro}$\\\n\hline\n\hline\n')
			s_add = ''
			st_line = "\hline\n"

		for i in range(len(prop_list)-1):
			f_fin.write(s_add + prop_list[i])
			f_fin.write(st_line)	
		f_fin.write(s_add + prop_list[len(prop_list)-1])

		f_fin.writelines(end_lines)
		# print("------------")


