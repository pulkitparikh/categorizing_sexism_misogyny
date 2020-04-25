import numpy as np   
from load_pre_proc import *
import sys
import time

conf_dict_list, conf_dict_com = load_config(sys.argv[1])
data_dict = load_data(conf_dict_com["filename"], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"], conf_dict_com["filename_map"], conf_dict_com['use_saved_data_stuff'], conf_dict_com['save_data_stuff'])

max_group_id = 2
r = 1
labs_in_all = [11,7,12,5,3,10,6,13]
# labs_in_all = []

res_path = "label_info_%s.txt" % (max_group_id+1)

def sort_by_train_coverage(trainY_list, num_labs):
	train_coverage = np.zeros(num_labs)
	for lset in trainY_list:
		for l in lset:
			train_coverage[l] += 1.0
	train_coverage /= np.sum(train_coverage)
	s_indices = np.argsort(train_coverage)
	for class_ind in s_indices:
		print("%d - %.3f" % (class_ind, train_coverage[class_ind]))		
	print("--------------")

def train_coverage(trainY_list, map_dict, num_labs):
	train_coverage = np.zeros(num_labs)
	for lset in trainY_list:
		for l in lset:
			if l in map_dict:
				train_coverage[map_dict[l]] += 1.0
	train_coverage /= np.sum(train_coverage)
	return train_coverage

def score(conf, train_c, max_group_id, out_l):
	s_arr = np.zeros(max_group_id + 1)
	for c_ind, l_id in enumerate(conf):
		for g_id in out_l[l_id]:
			s_arr[g_id] += train_c[c_ind]
	return -np.std(s_arr), s_arr

def comb_rec(arr, out_a, ind, a_ind, n, r, out_l):
	if r == 0:
		# print(out_a)
		out_l.append(list(out_a))
		return
	for i in range(a_ind, n-r+1):
		out_a[ind] = arr[i]
		comb_rec(arr, out_a, ind+1, i+1, n, r-1, out_l)
	return

startTime = time.time()

arr = [i for i in range(max_group_id+1)]
out_a = [0 for i in range(r)]
out_l = []	
comb_rec(arr, out_a, 0, 0, len(arr), r, out_l)
max_list_id = len(out_l)-1
print(out_l)

rem_labs = list(set(range(data_dict['NUM_CLASSES'])) - set(labs_in_all))
map_dict = {}
for ind, l in enumerate(rem_labs):
  map_dict[l] = ind
print(map_dict)
num_labs = len(rem_labs)

train_c = train_coverage(data_dict['lab'][:data_dict['train_en_ind']], map_dict, num_labs)
print(train_c)
print("-----------")
conf = np.zeros(num_labs, dtype=np.int64)

max_sc = -np.inf
while True:
	sc, s_arr = score(conf, train_c, max_group_id, out_l)
	if sc > max_sc:
		max_sc = sc
		best_conf = list(conf)
		best_s_arr =s_arr
	for i in range(num_labs-1, -1, -1):
		if conf[i] == max_list_id:
			conf[i] = 0
		else:
			conf[i]+= 1
			break
	if i == 0 and conf[i] == 0:
		break

with open(res_path, 'a') as f:
	sort_by_train_coverage(data_dict['lab'][:data_dict['train_en_ind']], data_dict['NUM_CLASSES'])
	classi_probs_label_info = [list(labs_in_all) for i in range(max_group_id+1)]
	for class_ind, l_id in enumerate(best_conf):
		for g_id in out_l[l_id]:
			classi_probs_label_info[g_id].append(rem_labs[class_ind])

	classi_probs_label_str = str(classi_probs_label_info)[2:-2].replace('], [', '+').replace(', ','_')

	timeLapsed = int(time.time() - startTime + 0.5)
	hrs = timeLapsed/3600.
	t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))

	s1 = "%s\n%s\n%s\n-----------\n%s\n%s\n%s" % (best_conf, max_sc, best_s_arr, classi_probs_label_info, classi_probs_label_str, t_str)

	print(s1)                
	f.write(s1)
