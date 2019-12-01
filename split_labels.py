import numpy as np   
from loadPreProc import *
import sys
import time

conf_dict_list, conf_dict_com = load_config(sys.argv[1])
data_dict = load_data(conf_dict_com["filename"], conf_dict_com["data_folder_name"], conf_dict_com["save_folder_name"], conf_dict_com['TEST_RATIO'], conf_dict_com['VALID_RATIO'], conf_dict_com['RANDOM_STATE'], conf_dict_com['MAX_WORDS_SENT'], conf_dict_com["test_mode"], conf_dict_com["filename_map"], conf_dict_com['use_saved_data_stuff'], conf_dict_com['save_data_stuff'])

max_group_id = 4
res_path = "label_info_%s.txt" % (max_group_id+1)

def train_coverage(trainY_list, NUM_CLASSES):
	train_coverage = np.zeros(NUM_CLASSES)
	for lset in trainY_list:
		for l in lset:
			train_coverage[l] += 1.0
	# train_coverage /= float(len(trainY_list))
	train_coverage /= np.sum(train_coverage)
	return train_coverage

def score(conf, train_c, max_group_id, train_labs):
	s_dict = {c:0 for c in range(max_group_id + 1)}
	for c_ind, g_id in enumerate(conf):
		s_dict[g_id] += train_c[c_ind]
	s_arr = np.array(list(s_dict.values()))
	# s_arr_norm = s_arr/np.sum(s_arr)
	return -np.std(s_arr), s_arr

# def score(conf, train_c, max_group_id, train_labs):
# 	s_dict = {c:0 for c in range(max_group_id + 1)}
# 	mem_dict = {c:{} for c in range(max_group_id + 1)}
# 	for c_ind, g_id in enumerate(conf):
# 		s_dict[g_id] += train_c[c_ind]
# 		mem_dict[g_id][c_ind] = None
# 	s_arr = np.array(list(s_dict.values()))
# 	s_arr_norm = s_arr/np.sum(s_arr)

# 	non_emp_count_arr = np.zeros(max_group_id+1)
# 	for l_list in train_labs:
# 		for g_id in range(max_group_id + 1):
# 			for c in l_list:
# 				if c in mem_dict[g_id]:
# 					non_emp_count_arr[g_id] += 1
# 					break
# 	non_emp_count_arr_norm = non_emp_count_arr/np.sum(non_emp_count_arr)					
# 	# print(non_emp_count_arr_norm)
# 	# print(s_arr_norm)
# 	# print(np.std(non_emp_count_arr_norm))
# 	# print(np.std(s_arr_norm))
# 	s1 = -np.std(non_emp_count_arr_norm)
# 	s2 = -np.std(s_arr_norm)
# 	return (s1+s2)/2, s1, s2, non_emp_count_arr, s_arr

startTime = time.time()

train_c = train_coverage(data_dict['lab'][:data_dict['train_en_ind']], data_dict['NUM_CLASSES'])
print(train_c)
print("-----------")
conf = np.zeros(data_dict['NUM_CLASSES'], dtype=np.int64)

max_sc = -np.inf
while True:
	# print(conf)
	# sc, s1, s2, non_emp_count_arr, s_arr = score(conf, train_c, max_group_id, data_dict['lab'][:data_dict['train_en_ind']])
	sc, s_arr = score(conf, train_c, max_group_id, data_dict['lab'][:data_dict['train_en_ind']])
	if sc > max_sc:
		max_sc = sc
		best_conf = list(conf)
		best_s_arr =s_arr
		# best_count_arr=non_emp_count_arr
	for i in range(data_dict['NUM_CLASSES']-1, -1, -1):
		if conf[i] == max_group_id:
			conf[i] = 0
		else:
			conf[i]+= 1
			break
	if i == 0 and conf[i] == 0:
		break

with open(res_path, 'a') as f:
	# print(best_conf)
	# print(max_sc)
	# print(best_s_arr)
	# # print(best_count_arr)
	# print("-----------")
	classi_probs_label_info = [[] for i in range(max_group_id+1)]
	for class_ind, g_id in enumerate(best_conf):
		classi_probs_label_info[g_id].append(class_ind)
	# print(classi_probs_label_info)

	classi_probs_label_str = str(classi_probs_label_info)[2:-2].replace('], [', '+').replace(', ','_')

	timeLapsed = int(time.time() - startTime + 0.5)
	hrs = timeLapsed/3600.
	t_str = "%.1f hours = %.1f minutes over %d hours\n" % (hrs, (timeLapsed % 3600)/60.0, int(hrs))

	s1 = "%s\n%s\n%s\n-----------\n%s\n%s\n%s" % (best_conf, max_sc, best_s_arr, classi_probs_label_info, classi_probs_label_str, t_str)

	print(s1)                
	f.write(s1)
