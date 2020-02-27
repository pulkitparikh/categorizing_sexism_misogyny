import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

metrics = {
	'multi-label': ['em', 'ji', 'ihl', 'pi', 'ri', 'fi', 'pl_mi', 'rl_mi', 'fl_mi', 'pl_ma', 'rl_ma', 'fl_ma'],
	'multi-class': ['acc', 'p_mi', 'r_mi', 'f_mi', 'p_ma', 'r_ma', 'f_ma', 'p_we', 'r_we', 'f_we'],
	'binary': ['acc', 'p', 'r', 'f'],
}
  
def rec_label(tp, fn):
	if (tp + fn) == 0:
		# print "denominator zero"
		return -np.inf
	return tp/float(tp + fn)

def prec_label(tp, fp):
	if (tp + fp) == 0:
		# print "denominator zero"
		return -np.inf
	return tp/float(tp + fp)

def f_label(tp, fp, fn):
	if (2*tp + fn + fp) == 0:
		# print "denominator zero"
		return -np.inf
	return (2*tp)/float(2*tp + fn + fp)

def components_F(pred, act, NUM_CLASSES):
	TP = np.zeros(NUM_CLASSES)
	FP = np.zeros(NUM_CLASSES)
	# TN = np.zeros(NUM_CLASSES)
	FN = np.zeros(NUM_CLASSES)
	for l_id in range(NUM_CLASSES):
		for pr, ac in zip(pred, act):
			if l_id in ac:
				if l_id in pr:
					TP[l_id] += 1
				else:
					FN[l_id] += 1
			else:
				if l_id in pr:
					FP[l_id] += 1
				# else:
				# 	TN[l_id] += 1
	return TP, FP, FN			

def jaccard_similarity(list1, list2):
    intersection = len(set(list1).intersection(list2))
    union = (len(list1) + len(list2)) - intersection
    if union == 0:
    	return 1
    return float(intersection) / union

def set_diff(list1, list2):
    intersection = len(set(list1).intersection(list2))
    return (len(list1) + len(list2)) - (2*intersection)

def hamming_loss(pred, act, NUM_CLASSES):
	cnt = 0.0
	for pr, ac in zip(pred, act):
		cnt += set_diff(pr, ac)
	return cnt/(len(pred)*NUM_CLASSES)		

def insights_results(pred_vals, true_vals, posts, sen_posts, dyn_fname_part, out_fold):
	inst_fold_str = ("%sinst/" % (out_fold))
	os.makedirs(inst_fold_str, exist_ok=True)
	dyn_fname_inst = ("%s%s.txt" % (inst_fold_str, dyn_fname_part))
	f_err_inst = open(dyn_fname_inst, 'w')
	f_err_inst.write("post\t# sents\t# words\t# words/sen\tpred cats\tactu cats\tJaccard\n")
	for pr, ac, post, sen_post in zip(pred_vals, true_vals, posts, sen_posts):
		js = jaccard_similarity(pr, ac)
		post_reco = "._ ".join(sen_post)
		num_words = len(post.split(" "))
		num_words_sen = float(num_words)/len(sen_post)
		pr_str = str(pr)[:-1][1:]
		ac_str = str(ac)[:-1][1:]
		f_err_inst.write("%s\t%d\t%d\t%.2f\t%s\t%s\t%.3f\n" % (post_reco, len(sen_post), num_words, num_words_sen, pr_str, ac_str, js))
	f_err_inst.close()

# insights_results_lab(pred_vals, true_vals, data_dict['lab'][0:data_dict['train_en_ind']], fname_part, conf_dict_com["output_folder_name"])
def insights_results_lab(pred_vals, true_vals, train_labels, dyn_fname_part, out_fold, NUM_CLASSES, FOR_LMAP):
	TP, FP, FN = components_F(pred_vals, true_vals, NUM_CLASSES)
	train_coverage = np.zeros(NUM_CLASSES)
	for lset in train_labels:
		for l in lset:
			train_coverage[l] += 1.0
	train_coverage /= float(len(train_labels))

	dyn_fname_lab = ("%slab/%s.txt" % (out_fold, dyn_fname_part))
	f_err_lab = open(dyn_fname_lab, 'w')
	f_err_lab.write("lab id\tlabel\ttrain cov\tPrec\tRecall\tF score\n")
	class_ind = 0
	for tp, fp, fn in zip(TP,FP,FN):
		P = prec_label(tp, fp)
		R = rec_label(tp, fn)
		F = f_label(tp, fp, fn)
		f_err_lab.write("%d\t%s\t%.2f\t%.3f\t%.3f\t%.3f\n" % (class_ind, FOR_LMAP[class_ind],train_coverage[class_ind]*100,P,R,F))
		class_ind += 1
	f_err_lab.close()

def aggregate_metr(metr_dict, num_vals, prob_type):
	global metrics
	for key in metrics[prob_type]:
		s = 0
		s_sq = 0
		print(metr_dict[key])
		for v in metr_dict[key]:
			s += v
			s_sq += v**2
		avg_v = s/num_vals
		metr_dict['avg_' + key] = avg_v
		# print(avg_v)
		# print(num_vals)
		# print(s_sq)
		metr_dict['std_' + key] = np.sqrt(s_sq/num_vals - avg_v**2)
		metr_dict[key] = []
	return metr_dict

def init_metr_dict(prob_type):
	global metrics
	metr_dict = {}
	for key in metrics[prob_type]:
		metr_dict[key] = []
	return metr_dict

def prep_write_lines(metr_dict, prob_type):
	if prob_type == 'multi-label':
		lines = [
			"f1 Instance: %.3f, std: %.3f" % (metr_dict['avg_fi'], metr_dict['std_fi']),
			"f1 Lable Macro: %.3f, std: %.3f" % (metr_dict['avg_fl_ma'], metr_dict['std_fl_ma']),
			"Jaccard Index: %.3f" % metr_dict['avg_ji'],
			"f1 Lable Micro: %.3f" % metr_dict['avg_fl_mi'],
			"Exact Match: %.3f" % metr_dict['avg_em'],
			"Inverse Hamming Loss: %.3f" % metr_dict['avg_ihl']
		]
	elif prob_type == 'multi-class':
		lines = [
			"f1 Weighted: %.3f, std: %.3f" % (metr_dict['avg_f_we'], metr_dict['std_f_we']),
			"f1 Macro: %.3f" % metr_dict['avg_f_ma'],
			"f1 Micro: %.3f" % metr_dict['avg_f_mi'],
			"P Weighted: %.3f" % metr_dict['avg_p_we'],
			"R Weighted: %.3f" % metr_dict['avg_r_we'],
			"Accuracy: %.3f" % metr_dict['avg_acc'],
		]
	elif prob_type == 'binary':
		lines = [
			"f1: %.3f, std: %.3f" % (metr_dict['avg_f'], metr_dict['std_f']),
			"P: %.3f" % metr_dict['avg_p'],
			"R: %.3f" % metr_dict['avg_r'],
			"Accuracy: %.3f" % metr_dict['avg_acc'],
		]
	return lines

def print_results(metr_dict,prob_type):
	lines = prep_write_lines(metr_dict, prob_type)
	for line in lines:
		print(line)

def write_print_results_no_tsv(metr_dict, f_res, prob_type):
	lines = prep_write_lines(metr_dict, prob_type)
	for line in lines:
		print(line)
		f_res.write(line + '\n')

def write_results(metr_dict, f_res, f_tsv, prob_type, model_type,word_feat_str,sent_enc_feat_str,classi_probs_label_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,thresh,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag,rnn_type,conf_dict_com):
	if prob_type == 'multi-label':
		f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\n" % (model_type,word_feat_str,sent_enc_feat_str,classi_probs_label_str,prob_trans_type,class_imb_flag,num_cnn_filters,max_pool_k_val,rnn_dim,att_dim,(metr_dict['avg_fl_ma']+metr_dict['avg_fi'])/2,(metr_dict['std_fl_ma']+metr_dict['std_fi'])/2,metr_dict['avg_fi'],metr_dict['avg_fl_ma'],(metr_dict['avg_fl_ma']+metr_dict['avg_fi']+metr_dict['avg_ji']+metr_dict['avg_fl_mi'])/4,metr_dict['avg_ji'],metr_dict['avg_fl_mi'],metr_dict['avg_em'],metr_dict['avg_ihl'],rnn_type,conf_dict_com["LEARN_RATE"],conf_dict_com["BATCH_SIZE"],conf_dict_com["dropO1"],conf_dict_com["dropO2"], conf_dict_com["test_mode"]))
	elif prob_type == 'multi-class':
		f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,thresh,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag,metr_dict['avg_f_we'],metr_dict['avg_f_ma'],metr_dict['avg_f_mi'],metr_dict['avg_acc'],metr_dict['avg_p_we'],metr_dict['avg_p_ma'],metr_dict['avg_p_mi'],metr_dict['avg_r_we'],metr_dict['avg_r_ma'],metr_dict['avg_r_mi'],metr_dict['std_f_we'],rnn_type,conf_dict_com["LEARN_RATE"],conf_dict_com["BATCH_SIZE"],conf_dict_com["dropO1"],conf_dict_com["dropO2"], conf_dict_com["test_mode"], classi_probs_label_str))
	elif prob_type == 'binary':
		f_tsv.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (model_type,word_feat_str,sent_enc_feat_str,prob_trans_type,class_imb_flag,num_cnn_filters,cnn_kernel_set_str,thresh,rnn_dim,att_dim,max_pool_k_val,stack_rnn_flag,metr_dict['avg_f'],metr_dict['avg_p'],metr_dict['avg_r'],metr_dict['avg_acc'],metr_dict['std_f'],rnn_type,conf_dict_com["LEARN_RATE"],conf_dict_com["BATCH_SIZE"],conf_dict_com["dropO1"],conf_dict_com["dropO2"], conf_dict_com["test_mode"], classi_probs_label_str))
	lines = prep_write_lines(metr_dict, prob_type)
	for line in lines:
		print(line)
		f_res.write(line + '\n')

def calc_metrics_print(pred_vals, true_vals, metr_dict, NUM_CLASSES, prob_type):
	if prob_type == 'multi-label':
		metr_dict['em'].append(exact_match(pred_vals, true_vals))
		metr_dict['ji'].append(jaccard_index_avg(pred_vals, true_vals))
		metr_dict['ihl'].append(inverse_hamming_loss(pred_vals, true_vals, NUM_CLASSES))
		pi, ri, fi = F_metrics_instance(pred_vals, true_vals)
		metr_dict['pi'].append(pi)
		metr_dict['ri'].append(ri)
		metr_dict['fi'].append(fi)
		TP, FP, FN = components_F(pred_vals, true_vals, NUM_CLASSES)
		pl_mi, rl_mi, fl_mi = F_metrics_label_micro_from_comp(TP, FP, FN)
		metr_dict['pl_mi'].append(pl_mi)
		metr_dict['rl_mi'].append(rl_mi)
		metr_dict['fl_mi'].append(fl_mi)
		pl_ma, rl_ma, fl_ma = F_metrics_label_macro_from_comp(TP, FP, FN, NUM_CLASSES)
		metr_dict['pl_ma'].append(pl_ma)
		metr_dict['rl_ma'].append(rl_ma)
		metr_dict['fl_ma'].append(fl_ma)
	elif prob_type == 'multi-class':
		pred_vals_sc = [labels[0] for labels in pred_vals]
		true_vals_sc = [labels[0] for labels in true_vals]
		metr_dict['p_mi'].append(precision_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='micro'))
		metr_dict['r_mi'].append(recall_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='micro'))
		metr_dict['f_mi'].append(f1_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='micro'))
		metr_dict['p_ma'].append(precision_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='macro'))
		metr_dict['r_ma'].append(recall_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='macro'))
		metr_dict['f_ma'].append(f1_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='macro'))
		metr_dict['p_we'].append(precision_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='weighted'))
		metr_dict['r_we'].append(recall_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='weighted'))
		metr_dict['f_we'].append(f1_score(true_vals_sc, pred_vals_sc, labels=np.arange(NUM_CLASSES), average='weighted'))
		metr_dict['acc'].append(accuracy_score(true_vals_sc, pred_vals_sc))
	elif prob_type == 'binary':
		metr_dict['p'].append(precision_score(true_vals, pred_vals))
		metr_dict['r'].append(recall_score(true_vals, pred_vals))
		metr_dict['f'].append(f1_score(true_vals, pred_vals))
		metr_dict['acc'].append(accuracy_score(true_vals, pred_vals))
	return metr_dict

# actual metric-related functions ------------------------------------
def exact_match(pred, act):
	cnt = 0.0
	for pr, ac in zip(pred, act):
		if set(pr) == set(ac):
			cnt += 1
	return cnt/len(pred)		

# also known as 'accuracy'
def jaccard_index_avg(pred, act):
	cnt = 0.0
	for pr, ac in zip(pred, act):
		cnt += jaccard_similarity(pr, ac)
	return cnt/len(pred)		

def inverse_hamming_loss(pred, act, NUM_CLASSES):
	cnt = 0.0
	for pr, ac in zip(pred, act):
		cnt += set_diff(pr, ac)
	# print ((len(pred)*NUM_CLASSES)-cnt)/(len(pred)*NUM_CLASSES)		
	# print 1-hamming_loss(pred, act, NUM_CLASSES)
	return ((len(pred)*NUM_CLASSES)-cnt)/(len(pred)*NUM_CLASSES)

def F_metrics_instance(pred, act):
	prec = 0.0
	rec = 0.0
	for pr, ac in zip(pred, act):
		intersection = float(len(set(pr).intersection(ac)))
		prec += intersection/len(pr)
		rec += intersection/len(ac)
	prec = prec/len(pred)
	rec = rec/len(pred)
	if (prec+rec) == 0:
		f_sc = 0.0
	else:	
		f_sc = 2*prec*rec/(prec+rec) 
	return prec, rec, f_sc	

def F_metrics_label_macro(pred, act, NUM_CLASSES):
	TP, FP, FN = components_F(pred, act, NUM_CLASSES)
	return F_metrics_label_macro_from_comp(TP, FP, FN, NUM_CLASSES)

def F_metrics_label_macro_from_comp(TP, FP, FN, NUM_CLASSES):
	# print TP
	# print FP
	# print FN
	avgF = 0.0
	avgP = 0.0
	avgR = 0.0
	for tp, fp, fn in zip(TP,FP,FN):
		avgP += prec_label(tp, fp)
		avgR += rec_label(tp, fn)
		avgF += f_label(tp, fp, fn)

	# f_scores = [f_label(tp, fp, fn) for tp, fp, fn in zip(TP,FP,FN)]
	# precs = [prec_label(tp, fp) for tp, fp in zip(TP,FP)]
	# recs = [rec_label(tp, fn) for tp, fn in zip(TP,FN)]
	return avgP/NUM_CLASSES, avgR/NUM_CLASSES, avgF/NUM_CLASSES

def F_metrics_label_micro(pred, act, NUM_CLASSES):
	TP, FP, FN = components_F(pred, act, NUM_CLASSES)
	return F_metrics_label_micro_from_comp(TP, FP, FN)

def F_metrics_label_micro_from_comp(TP, FP, FN):
	# print TP
	# print FP
	# print FN
	avgTP = np.mean(TP)
	avgFP = np.mean(FP)
	avgFN = np.mean(FN)
	return prec_label(avgTP, avgFP), rec_label(avgTP, avgFN), f_label(avgTP, avgFP, avgFN)

# act = [[1, 6, 16], [18], [1, 5], [3, 15]]
# pred = [[16, 1, 6], [3, 6, 17], [1, 5, 2], [3, 15]]
# print len(act)

# print "exact_match"
# print exact_match(pred, act)

# print "jaccard_index_avg"
# print jaccard_index_avg(pred, act)

# print "inverse_hamming_loss"
# print inverse_hamming_loss(pred, act)

# print "F_metrics_instance"
# p,r,f = F_metrics_instance(pred, act)
# print p
# print r
# print f

# print "F_metrics_label_macro"
# p,r,f = F_metrics_label_macro(pred, act)
# print p
# print r
# print f

# print "F_metrics_label_micro"
# p,r,f = F_metrics_label_micro(pred, act)
# print p
# print r
# print f
