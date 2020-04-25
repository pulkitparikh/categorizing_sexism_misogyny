import numpy as np
from sklearn.model_selection import train_test_split
from loadPreProc import *
from evalMeasures import *
import sys

conf_dict_list, conf_dict_com = load_config(sys.argv[1])

os.makedirs(conf_dict_com["output_folder_name"], exist_ok=True)
os.makedirs(conf_dict_com["save_folder_name"], exist_ok=True)
data_dict = load_data(conf_dict_com["filename"], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"], conf_dict_com["filename_map"], conf_dict_com['use_saved_data_stuff'], conf_dict_com['save_data_stuff'])

res_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_filename"]
if os.path.isfile(res_path):
    f_res = open(res_path, 'a')
else:
    f_res = open(res_path, 'w')

tsv_path = conf_dict_com["output_folder_name"] + conf_dict_com["res_tsv_filename"]
if os.path.isfile(tsv_path):
    f_tsv = open(tsv_path, 'a')
else:
    f_tsv = open(tsv_path, 'w')
    if data_dict['prob_type'] == 'multi-label':
        f_tsv.write("model\tword feats\tsent feats\tclassi probs labels\ttrans\tclass imb\tcnn filters\tmax pool k\trnn dim\tatt dim\tf_I+f_Ma\tstd_d\tf1-Inst\tf1-Macro\tsum_4\tJaccard\tf1-Micro\tExact\tI-Ham\trnn type\tl rate\tb size\tdr1\tdr2\ttest mode\n") 
    elif data_dict['prob_type'] == 'multi-class':
        f_tsv.write("model\tword feats\tsent feats\ttrans\tclass imb\tcnn fils\tcnn kernerls\tthresh\trnn dim\tatt dim\tpool k\tstack RNN\tf-We\tf-Ma\tf-Mi\tacc\tp-We\tp_Ma\tp_Mi\tr_We\tr_Ma\tr_Mi\tstd_f_we\trnn type\tl rate\tb size\tdr1\tdr2\ttest mode\tclassi probs labels\n") 
    elif data_dict['prob_type'] == 'binary':
        f_tsv.write("model\tword feats\tsent feats\ttrans\tclass imb\tcnn fils\tcnn kernerls\tthresh\trnn dim\tatt dim\tpool k\tstack RNN\tf\tp\tr\tacc\tstd_f\trnn type\tl rate\tb size\tdr1\tdr2\ttest mode\tclassi probs labels\n") 

def train_evaluate_model(trainY_list, true_vals, metr_dict, prob_type, NUM_CLASSES):
	train_coverage = np.zeros(NUM_CLASSES)
	for lset in trainY_list:
		for l in lset:
			train_coverage[l] += 1.0
	train_coverage /= float(len(trainY_list))

	if prob_type == 'multi-label':
		r_op = np.empty((len(true_vals), NUM_CLASSES), dtype=int)
		for i in range(len(true_vals)):
			while(True):
				r_op[i, :] = 0
				for j in range(NUM_CLASSES):
					r_num = np.random.uniform()
					# if r_num < 1:
					# if r_num < 0.5:
					if r_num < train_coverage[j]:
						r_op[i,j] = 1			
				if sum(r_op[i, :]) > 0:
					break

		pred_vals = di_op_to_label_lists(r_op)
	else:
		pred_vals = []
		for i in range(len(true_vals)):
			r_num = np.random.uniform()
			start_range = 0
			for j in range(NUM_CLASSES):
				end_range = start_range + train_coverage[j]
				if r_num >= start_range and r_num < end_range:
					pred_vals.append([j])
					break
				start_range = end_range	
	assert(len(pred_vals) == len(true_vals))			
	return pred_vals, true_vals, calc_metrics_print(pred_vals, true_vals, metr_dict, NUM_CLASSES, prob_type)

f_res = open(conf_dict_com["output_folder_name"] + '/' + conf_dict_com["res_filename"], 'a')
metr_dict = init_metr_dict(data_dict['prob_type'])
model_type = conf_dict_list[0]['model_types'][0]
info_str = "model: %s, test mode = %s" % (model_type, conf_dict_com["test_mode"])
for run_ind in range(conf_dict_com["num_runs"]):
	pred_vals, true_vals, metr_dict = train_evaluate_model(data_dict['lab'][:data_dict['train_en_ind']], data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']], metr_dict, data_dict['prob_type'], data_dict['NUM_CLASSES'])

f_res.write("%s\n\n" % info_str)
print("%s\n" % info_str)

metr_dict = aggregate_metr(metr_dict, conf_dict_com["num_runs"], data_dict['prob_type'])
write_results(metr_dict, f_res, f_tsv, data_dict['prob_type'], model_type,'','','','','','','','','','','','','',conf_dict_com)

f_res.close()
f_tsv.close()