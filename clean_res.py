import csv
import os

fl_only_report_best = False
only_inc_best_list = []
exclude_files = []
remove_list = []
useless_list = []

# output_fold = 'results_mis/'
# best_metr = 'f-We'

# output_fold = 'results_det/'
# best_metr = 'f'

output_fold = 'results_mul/'
best_metr = 'f_I+f_Ma'

remove_list.extend(useless_list)

val_res_file = output_fold + 'val_tsv.txt'
test_res_file = output_fold + 'test_tsv.txt'

def rem_dupli_text_new(f_name_list):
	rows_org = []
	for f_name in f_name_list:
		with open(f_name,'r') as res_tsv:
			reader = csv.DictReader(res_tsv, delimiter = '\t')
			rows_org += list(reader)

	p_remov_inds = [False]*len(rows_org)
	p_best_dict = {}
	for ind, row in enumerate(rows_org):
		row_key = tuple(sorted(row.items(), key=lambda kv: kv[0]))
		if row_key in p_best_dict:
			p_remov_inds[p_best_dict[row_key]] = True
		p_best_dict[row_key] = ind

	rows = []
	for ind, row in enumerate(rows_org):
		if p_remov_inds[ind] == False:
			row['GPU'] = '0'
			rows.append(row)

	return rows, reader.fieldnames + ['GPU']

file_list = []
for file in os.listdir(output_fold):
	if file[-4:] == '.txt' and file.startswith('tsv') and file not in exclude_files:
		file_list.append(output_fold + file)
rows, my_fields = rem_dupli_text_new(file_list)

model_dic_test = {}
model_dic = {}
for row in rows:
	if row['rnn dim'] == '400' or row['model'] == 'random':
		continue
	att_d = int(row['att dim'])	
	att_flag = True if att_d > 0 else False 
	classi_probs_labels_val = row['classi probs labels'] if 'classi probs labels' in row else ''
	model_key = (row['model'], row['word feats'], row['sent feats'], classi_probs_labels_val, row['trans'], row['class imb'], att_flag)

	if row['test mode'] == "True":
		if model_key in model_dic_test:
			print("multiple hyper params run in the test mode!")
			exit()
		else:
			model_dic_test[model_key] = row
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
	model_dic_sorted[key] = sorted(row_list, key = lambda i: float(i[best_metr]), reverse=True)
	mod_best_dic[key] = float(model_dic_sorted[key][0][best_metr])

mod_best_sorted = sorted(mod_best_dic.items(), key = lambda x : x[1], reverse=True)

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

mod_best_dic_test = {}
for key, row in model_dic_test.items():
	mod_best_dic_test[key] = float(row[best_metr])

mod_best_sorted_test = sorted(mod_best_dic_test.items(), key = lambda x : x[1], reverse=True)

with open(test_res_file, 'w') as f_fin:
	w_fin = csv.DictWriter(f_fin, fieldnames = my_fields, delimiter = '\t')
	w_fin.writeheader()
	for mod_key, val in mod_best_sorted_test:
		w_fin.writerow(model_dic_test[mod_key])
