import csv

fl_only_report_best = False
only_inc_best_list = [
]
#('hier_fuse', 'elm_rnn_11+glo_rnn_11', 'bert_pre_1+use_1', 'di', 'True'), ('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1+use_1', 'di', 'True'), ('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1', 'di', 'True'), ('hier_fuse', 'elm_rnn_11+glo_rnn_11', 'bert_pre_1', 'di', 'True'), ('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_2', 'di', 'True'), ('hier_fuse', 'elm_cnn_11+glo_cnn_21', 'bert_pre_1', 'di', 'True'), ('hier_fuse', 'elm_cnn_11+glo_cnn_11', 'bert_pre_1', 'di', 'True'), ('hier_fuse', 'elm_rnn_11', '', 'di', 'True'), ('hier_fuse', 'elm_cnn_11', '', 'di', 'True'), ('hier_fuse', 'glo_rnn_11', '', 'di', 'True'), ('hier_fuse', 'glo_cnn_11', '', 'di', 'True'), ('hier_fuse', 'fas_rnn_11', '', 'di', 'True'), ('hier_fuse', 'fas_cnn_11', '', 'di', 'True'), ('uni_sent', '', 'bert_pre_1', 'di', 'True'), ('uni_sent', '', 'bert_1', 'di', 'True'), ('uni_sent', '', 'use_1', 'di', 'True'), ('uni_sent', '', 'infersent_1', 'di', 'True')]
remove_list = [
	('flat_fuse', 'elmo~rnn~1~~glove~rnn~2~~ling~rnn~3~~~~~', 'bert_pre~1~~~~', 'di', 'True', True),
	('flat_fuse', 'elmo~rnn~1~~glove~rnn~2~~~~~~~~~', 'bert_pre~1~use~1~~', 'di', 'True', True),
	('flat_fuse', 'elmo~rnn~1~~glove~rnn~2~~~~~~~~~', 'bert_pre~1~~~~', 'di', 'True', True),
	('flat_fuse', 'elmo~rnn~1~~~~~~~~~~~~~', 'bert_pre~1~~~~', 'di', 'True', True),
	('flat_fuse', 'elmo~rnn~1~~glove~rnn~1~~~~~~~~~', 'bert_pre~1~~~~', 'di', 'True', True),
	# ('flat_fuse', 'elmo~cnn~1~~glove~cnn~2~~~~~~~~~', 'bert_pre~1~use~1~~', 'di', 'True', True),
	# ('flat_fuse', 'elmo~cnn~1~~~~~~~~~~~~~', 'bert_pre~1~~~~', 'di', 'True', True),

	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_2+use_3', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_22', 'bert_pre_3', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_2', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_2+use_2', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_11', 'bert_pre_2', 'di', 'True', True),

	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1+use_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21+lin_rnn_31', 'bert_pre_1', 'di', 'True', True),
	('hier_fuse', 'elm_sep_111+glo_sep_211', 'bert_pre_1', 'di', 'True', True),
	('hier_fuse', 'elm_cnn_11+glo_cnn_21', 'bert_pre_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_11', 'bert_pre_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11', 'bert_pre_1', 'di', 'True', True),

	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_0.9_0.8_0.7_0.6_0.5_mse_0.25_0.25_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_0.9_0.8_0.7_mse_0.25_0.25_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_0.95_0.9_0.85_mse_0.25_0.25_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_0.95_0.9_0.85_0.8_0.75_mse_0.25_0.25_1', 'di', 'True', True),

	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1', 'lp', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1', 'lp', 'False', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1', 'dc', 'True', True),

	('hier_fuse', 'elm_rnn_11', '', 'dc', 'True', True),
	('flat_fuse', 'elm_rnn_1', '', 'dc', 'True', True),
	('flat_fuse', 'elm_rnn_1', '', 'lp', 'False', True),
	('flat_fuse', 'elm_rnn_1', '', 'lp', 'True', True),
	('hier_fuse', 'elm_rnn_11', '', 'lp', 'False', True),
	('hier_fuse', 'elm_rnn_11', '', 'lp', 'True', True),
	('flat_fuse', 'elm_cnn_1', '', 'di', 'True', True),
	('flat_fuse', 'elm_rnn_1', '', 'di', 'True', True),
	('flat_fuse', 'elm_rnn_1', '', 'di', 'True', False),

	('flat_fuse', 'glo_rnn_1', '', 'di', 'True', True),
	('flat_fuse', 'fas_rnn_1', '', 'di', 'True', True),

	('c_bilstm', 'elm_rnn_1', '', 'di', 'True', False),

	('uni_sent', '', 'bert_pre_1', 'di', 'True', True),
	('uni_sent', '', 'bert_1', 'di', 'True', True),
	('uni_sent', '', 'use_1', 'di', 'True', True),
	('uni_sent', '', 'infersent_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11', '', 'di', 'True', True),
	('hier_fuse', 'glo_rnn_11', '', 'di', 'True', True),
	('hier_fuse', 'fas_rnn_11', '', 'di', 'True', True),
	('hier_fuse', 'elm_cnn_11', '', 'di', 'True', True),
	('hier_fuse', 'glo_cnn_11', '', 'di', 'True', True),
	('hier_fuse', 'fas_cnn_11', '', 'di', 'True', True)]

	# ('hier_fuse', 'elm_sep_111+glo_sep_111', 'bert_pre_1', 'di', 'True'),
	# ('hier_fuse', 'elm_rnn_11+glo_rnn_22', 'bert_pre_3', 'di', 'True'),
	# ('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1', 'lp', 'False'),
	# ('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1', 'dc', 'True'),
	# ('hier_fuse', 'elm_cnn_11', 'bert_pre_1', 'di', 'True'),
	# ('hier_fuse', 'elm_rnn_11', 'bert_pre_1', 'di', 'True'),
	# ('hier_fuse', 'elm_rnn_11+glo_rnn_21+lin_rnn_31', 'bert_pre_1', 'di', 'True'),
	# ('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_2', 'di', 'True'),
	# ('hier_fuse', 'elm_rnn_11+glo_rnn_11', 'bert_pre_1+use_1', 'di', 'True'),
	# ('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1', 'di', 'True'),
	# ('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_pre_1+use_1', 'di', 'True'),
	# ('hier_fuse', 'elm_cnn_11+glo_cnn_21', 'bert_pre_1', 'di', 'True'),
	# ('hier_fuse', 'elm_cnn_11+glo_cnn_11', 'bert_pre_1', 'di', 'True'),
useless_list = [
	# ('hier_fuse', 'elm_rnn_11+glo_rnn_11', 'bert_pre_2', 'di', 'True', True),
	('hier_fuse', 'elm_cnn_11', '', 'lp', 'False', True),
	('hier_fuse', 'elm_cnn_11', '', 'dc', 'True', True),
	# ('hier_fuse', 'elm_rnn_11', '', 'lp', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_11', 'bert_2', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_2', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_22', 'bert_3', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_22', 'use_3', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_1+use_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'use_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_11', 'bert_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21', 'bert_1+use_1', 'di', 'True', True),
	('hier_fuse', 'elm_rnn_11+glo_rnn_21+fas_rnn_31+lin_rnn_41', 'bert_1', 'di', 'True', True),
	('hier_fuse', 'elm_sep_111+glo_sep_211', 'bert_1', 'di', 'True', True),
	('hier_fuse', 'elm_cnn_11+glo_cnn_21', 'bert_1', 'di', 'True', True)]
input_res_file_1 = 'results/tsv1.txt'
input_res_file_2 = 'results/tsv2.txt'
val_res_file = 'results/val_tsv.txt'
test_res_file = 'results/test_tsv.txt'
remove_list.extend(useless_list)

def rem_dupli_text(f_name_1, f_name_2):
	with open(f_name_2,'r') as res_tsv:
		reader = csv.DictReader(res_tsv, delimiter = '\t')
		rows2_org = list(reader)

	p_best_dict_2 = {}
	p_remov_inds_2 = [False]*len(rows2_org)
	for ind, row in enumerate(rows2_org):
		post_cl = tuple(sorted(row.items(), key=lambda kv: kv[1]))
		if post_cl in p_best_dict_2:
			p_remov_inds_2[p_best_dict_2[post_cl]] = True
		p_best_dict_2[post_cl] = ind

	with open(f_name_1,'r') as res_tsv:
		reader = csv.DictReader(res_tsv, delimiter = '\t')
		rows1_org = list(reader)

	p_best_dict_1 = {}
	p_remov_inds_1 = [False]*len(rows1_org)
	for ind, row in enumerate(rows1_org):
		post_cl = tuple(sorted(row.items(), key=lambda kv: kv[1]))
		if post_cl in p_best_dict_1:
			p_remov_inds_1[p_best_dict_1[post_cl]] = True
		if post_cl in p_best_dict_2:
			p_remov_inds_2[p_best_dict_2[post_cl]] = True
		p_best_dict_1[post_cl] = ind

	rows2 = []
	for ind, row in enumerate(rows2_org):
		if p_remov_inds_2[ind] == False:
			rows2.append(row)
	
	rows1 = []
	for ind, row in enumerate(rows1_org):
		if p_remov_inds_1[ind] == False:
			rows1.append(row)

	return rows1, rows2, reader.fieldnames + ['GPU']

# rev_input_res_file_1 = 'results/rev_tsv1.txt'
# rev_input_res_file_2 = 'results/rev_tsv2.txt'
# def revise_input(input_res_file, op_name):
# 	with open(input_res_file,'r') as res_tsv:
# 		reader = csv.DictReader(res_tsv, delimiter = '\t')
# 		rows = list(reader)

# 	with open(op_name, 'w') as f_fin:
# 		w_fin = csv.DictWriter(f_fin, fieldnames = reader.fieldnames, delimiter = '\t')
# 		w_fin.writeheader()
# 		for row in rows:
# 			if 'bert_pre' not in row['sent feats']:
# 				w_fin.writerow(row)
# revise_input(input_res_file_1, rev_input_res_file_1)
# revise_input(input_res_file_2, rev_input_res_file_2)
# exit(1)

rows1_raw, rows2_raw, my_fields = rem_dupli_text(input_res_file_1, input_res_file_2)
rows1 = []
for row in rows1_raw:
	row['GPU'] = '0'
	rows1.append(row)
rows2 = []
for row in rows2_raw:
	row['GPU'] = '1'
	rows2.append(row)
rows = rows2 + rows1
# rows = list({tuple(d.items()) for d in rows})

# with open('test.txt', 'w') as f_fin:
# 	w_fin = csv.DictWriter(f_fin, fieldnames = my_fields, delimiter = '\t')
# 	w_fin.writeheader()
# 	for row in rows:
# 		w_fin.writerow(row)
# exit()

model_dic_test = {}
model_dic = {}
for row in rows:
	if row['rnn dim'] == '400':
		continue

	att_d = int(row['att dim'])	
	att_flag = True if att_d > 0 else False 
	model_key = (row['model'], row['word feats'], row['sent feats'], row['trans'], row['class imb'], att_flag)

	if row['test mode'] == "True":
		if model_key in model_dic_test:
			print("multiple hyper params run in the test mode!")
			exit()
		else:
			model_dic_test[model_key] = row
		# if model_key in model_dic_test:
		# 	if row['GPU'] == '0':
		# 		model_dic_test[model_key] = row
		# 	else:
		# 		if model_dic_test[model_key]['GPU'] == '1':
		# 			model_dic_test[model_key] = row
		# else:
		# 	model_dic_test[model_key] = row
	else:
		if model_key not in remove_list:
			if model_key in model_dic:
				model_dic[model_key].append(row)
			else:
				model_dic[model_key] = [row]
#####################################################
mod_best_dic = {}
model_dic_sorted = {}
for key, row_list in model_dic.items():
	model_dic_sorted[key] = sorted(row_list, key = lambda i: float(i['f_I+f_Ma']), reverse=True)
	mod_best_dic[key] = float(model_dic_sorted[key][0]['f_I+f_Ma'])

# print(model_dic.keys())
# print("---------------")

mod_best_sorted = sorted(mod_best_dic.items(), key = lambda x : x[1], reverse=True)
# for x in mod_best_sorted:
# 	print(x)
# print("-----------------------")

# with open(val_res_file,'r') as f:
# 	reader = csv.DictReader(f, delimiter = '\t')
# 	rows_prev = [sorted(tuple(d.items())) for d in list(reader)]

# for mod_key, val in mod_best_sorted:
# 	for row in model_dic_sorted[mod_key]:
# 		if sorted(tuple(row.items())) not in rows_prev:
# 			print(row)
# print("-----------------------")

sep_row = {}
for f in my_fields:
	sep_row[f] = '#####'
with open(val_res_file, 'w') as f_fin:
	w_fin = csv.DictWriter(f_fin, fieldnames = my_fields, delimiter = '\t')
	w_fin.writeheader()
	if fl_only_report_best:
		for mod_key, val in mod_best_sorted:
			w_fin.writerow(model_dic_sorted[mod_key][0])
	else:
		for mod_key, val in mod_best_sorted:
			if mod_key in only_inc_best_list:
				w_fin.writerow(model_dic_sorted[mod_key][0])
			else:
				for row in model_dic_sorted[mod_key]:
					w_fin.writerow(row)
				w_fin.writerow(sep_row)

#####################################################
try:
	with open(test_res_file,'r') as f:
		reader = csv.DictReader(f, delimiter = '\t')
		rows_prev = [sorted(tuple(d.items())) for d in list(reader)]
except:
	rows_prev = []

mod_best_dic_test = {}
# model_dic_sorted_test = {}
for key, row in model_dic_test.items():
	# model_dic_sorted_test[key] = sorted(row_list, key = lambda i: float(i['f_I+f_Ma']), reverse=True)
	mod_best_dic_test[tuple(list(key)+[row['GPU']])] = float(row['f_I+f_Ma'])

mod_best_sorted_test = sorted(mod_best_dic_test.items(), key = lambda x : x[1], reverse=True)
for x in mod_best_sorted_test:
	print(x)
print("-----------------------")
for mod_key, val in mod_best_sorted_test:
	row = model_dic_test[mod_key[:-1]]
	if sorted(tuple(row.items())) not in rows_prev:
		print(sorted(tuple(row.items())))

with open(test_res_file, 'w') as f_fin:
	w_fin = csv.DictWriter(f_fin, fieldnames = my_fields, delimiter = '\t')
	w_fin.writeheader()
	for mod_key, val in mod_best_sorted_test:
		# for row in model_dic_sorted_test[mod_key]:
		w_fin.writerow(model_dic_test[mod_key[:-1]])
