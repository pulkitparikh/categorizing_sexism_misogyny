import os
import numpy as np
from dlModels import get_model, attLayer_hier, multi_binary_loss, br_binary_loss, lp_categ_loss, multi_cat_w_loss, multi_cat_loss, kmax_pooling
from sklearn.utils import class_weight
from loadPreProc import *
from evalMeasures import *
# from keras.models import load_model
# from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras import backend as K
from gen_batch_keras import TrainGenerator, TestGenerator
import pickle
import json

# mod_op[i] is the probablity of the ith test sample's class being 1 (verified)
def evaluate_model(mod_op_list, data_dict, bac_map, prob_trans_type, metr_dict, thresh, att_flag, output_folder_name, fname_part_r_ind, classi_probs_label_info):
    y_pred_list = []
    true_vals = data_dict['lab'][data_dict['test_st_ind']:data_dict['test_en_ind']]
    num_test_samp = len(true_vals)
    sum_br_lists = np.zeros(num_test_samp, dtype=np.int64)
    arg_max_br_lists = np.empty(num_test_samp, dtype=np.int64)
    max_br_lists = np.zeros(num_test_samp)
    for cl_ind, tup_val in enumerate(mod_op_list):
        mod_op, att_op = tup_val
        if prob_trans_type == 'lp':
            y_pred  = np.argmax(mod_op, 1)
        elif prob_trans_type == "di":        
            y_pred = np.rint(mod_op).astype(int)
        elif prob_trans_type == "dc":
            y_pred = np.zeros((mod_op.shape), dtype=np.int64)
            for ind_row, row in enumerate(mod_op):
                s_indices = np.argsort(-row)
                row_s = row[s_indices]
                dif = row_s[:len(row)-1] - row_s[1:]
                m_ind = dif.argmax()
                y_pred[ind_row, s_indices[:m_ind+1]] = 1
                # y_pred[ind_row, row > thresh*max(row)] = 1  
        elif prob_trans_type == "br":
            mod_op = np.squeeze(mod_op, -1)
            y_pred = np.rint(mod_op).astype(int)
            if data_dict['prob_type'] == 'multi-label':
                sum_br_lists += y_pred
                for i in range(num_test_samp):
                    if mod_op[i] > max_br_lists[i]:
                        max_br_lists[i] = mod_op[i]
                        arg_max_br_lists[i] = cl_ind
        y_pred_list.append(y_pred)

    if prob_trans_type == 'lp':
        pred_vals = powerset_vec_to_label_lists(y_pred_list[0], bac_map, data_dict['NUM_CLASSES'])
    elif prob_trans_type == "di":
        pred_vals = di_list_op_to_label_lists(y_pred_list, mod_op_list, data_dict['NUM_CLASSES'], classi_probs_label_info)
    elif prob_trans_type == "dc":
        pred_vals = di_op_to_label_lists(y_pred_list[0])
    elif prob_trans_type == "br":
        if data_dict['prob_type'] == 'multi-label':
            for i in range(len(true_vals)):
                if sum_br_lists[i] == 0:
                    y_pred_list[arg_max_br_lists[i]][i] = 1
            pred_vals = br_op_to_label_lists(y_pred_list)
        elif data_dict['prob_type'] == 'binary':
            pred_vals = bin_classi_op_to_label_lists(y_pred_list[0])
  
    if att_flag and (mod_op_list[0][1] is not None):
        true_vals_multi_hot = trans_labels_multi_hot(true_vals, data_dict['NUM_CLASSES'])
        pred_vals_multi_hot = trans_labels_multi_hot(pred_vals, data_dict['NUM_CLASSES'])
        for ind, data_ind in enumerate(range(data_dict['test_st_ind'], data_dict['test_en_ind'])):
        # for ind, post in enumerate(data_dict['text_sen'][data_dict['test_st_ind']:data_dict['test_en_ind']]):
            att_path = "%satt_info/%s/" % (output_folder_name, fname_part_r_ind)
            os.makedirs(att_path, exist_ok=True)
            true_vals_multi_hot_ind = true_vals_multi_hot[ind].tolist()
            y_pred_list_0_ind = pred_vals_multi_hot[ind].tolist()
            mod_op_list_0_0_ind = mod_op_list[0][0][ind].tolist()
            post_sens = data_dict['text_sen'][data_ind][:data_dict['max_num_sent']]
            post_sens_split = []
            for sen in post_sens:
                post_sens_split.append(sen.split(' '))
            for clust_ind, att_arr in enumerate(mod_op_list[0][1]):
                att_list = []
                if len(att_arr.shape) == 3:
                    fname_att = ("%s%d~w%d.json" % (att_path, ind, clust_ind))
                    for ind_sen, split_sen in enumerate(post_sens_split):
                        my_sen_dict = {}
                        my_sen_dict['text'] = split_sen#[:data_dict['max_words_sent']]
                        my_sen_dict['label'] = true_vals_multi_hot_ind
                        my_sen_dict['prediction'] = y_pred_list_0_ind
                        my_sen_dict['posterior'] = mod_op_list_0_0_ind
                        my_sen_dict['attention'] = att_arr[ind, ind_sen, :].tolist()
                        my_sen_dict['id'] = "%d~w%d~%d" % (ind, clust_ind, ind_sen)
                        att_list.append(my_sen_dict)
                else:
                    my_sen_dict = {}
                    my_sen_dict['label'] = true_vals_multi_hot_ind
                    my_sen_dict['prediction'] = y_pred_list_0_ind
                    my_sen_dict['posterior'] = mod_op_list_0_0_ind
                    my_sen_dict['attention'] = att_arr[ind, :].tolist()
                    if fname_part_r_ind.startswith('hier_fuse'):
                        fname_att = ("%s%d~s%d.json" % (att_path, ind, clust_ind))
                        my_sen_dict['text'] = data_dict['text_sen'][data_ind]#[:data_dict['max_num_sent']]
                        my_sen_dict['id'] = "%d~s%d" % (ind, clust_ind)
                    else:
                        fname_att = ("%s%d~w%d.json" % (att_path, ind, clust_ind))
                        my_sen_dict['text'] = data_dict['text'][data_ind].split(' ')#[:data_dict['max_post_length']]
                        my_sen_dict['id'] = "%d~w%d" % (ind, clust_ind)
                    att_list.append(my_sen_dict)
                with open(fname_att, 'w') as f:
                    json.dump(att_list, f)

    return pred_vals, true_vals, calc_metrics_print(pred_vals, true_vals, metr_dict, data_dict['NUM_CLASSES'], data_dict['prob_type'])

def train_predict(word_feats, sent_enc_feats, trainY, data_dict, model_type, num_cnn_filters, rnn_type, fname_part, loss_func, class_w, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, cnn_kernel_set, m_ind, run_ind, save_folder_name, use_saved_model, gen_att, learn_rate, dropO1, dropO2, batch_size, num_epochs, save_model, save_trained_mod):
    att_op = None
    fname_mod_op = ("%s%s/iop~%d~%d.pickle" % (save_folder_name, fname_part, m_ind, run_ind))
    if use_saved_model and os.path.isfile(fname_mod_op):
        print("loading model o/p")
        with open(fname_mod_op, 'rb') as f:
            mod_op = pickle.load(f)
        if gen_att:
            fname_att_op = ("%s%s/att_op~%d~%d.pickle" % (save_folder_name, fname_part, m_ind, run_ind))
            if os.path.isfile(fname_att_op):
                with open(fname_att_op, 'rb') as f:
                    att_op = pickle.load(f)
    else:
        model, att_mod = get_model(model_type, data_dict['max_post_length'], data_dict['max_num_sent'], data_dict['max_words_sent'], word_feats, sent_enc_feats, learn_rate, dropO1, dropO2, num_cnn_filters, rnn_type, loss_func, nonlin, out_vec_size, rnn_dim, att_dim, max_pool_k_val, stack_rnn_flag, cnn_kernel_set)
        training_generator = TrainGenerator(np.arange(0, data_dict['train_en_ind']), trainY, word_feats, sent_enc_feats, data_dict, batch_size)
        model.fit_generator(generator=training_generator, epochs=num_epochs, shuffle=False, verbose=1, use_multiprocessing=False, workers=1)
        test_generator = TestGenerator(np.arange(data_dict['test_st_ind'], data_dict['test_en_ind']), word_feats, sent_enc_feats, data_dict, batch_size)
        mod_op = model.predict_generator(generator=test_generator, verbose=1, use_multiprocessing=False, workers=1)
        if gen_att and (att_mod is not None):
            att_op = att_mod.predict_generator(generator=test_generator, verbose=1, use_multiprocessing=False, workers=1)
            if type(att_op) != list:
                att_op = [att_op]

        if save_model:    
            print("saving model o/p")
            os.makedirs(save_folder_name + fname_part, exist_ok=True)
            with open(fname_mod_op, 'wb') as f:
                pickle.dump(mod_op, f)
            if run_ind == 0:
                if save_trained_mod:
                    fname_mod = ("%s%s/mod~%d.h5" % (save_folder_name, fname_part, m_ind))
                    model.save(fname_mod)
                if m_ind == 0:
                    with open("%s%s/mod_sum.txt" % (save_folder_name, fname_part),'w') as fh:
                        model.summary(print_fn=lambda x: fh.write(x + '\n'))                                                                
            if gen_att:
                fname_att_op = ("%s%s/att_op~%d~%d.pickle" % (save_folder_name, fname_part, m_ind, run_ind))
                with open(fname_att_op, 'wb') as f:
                    pickle.dump(att_op, f)
        K.clear_session()
    return mod_op, att_op

def class_imb_loss_nonlin(trainY_noncat_list, class_imb_flag, num_classes_list, prob_trans_type, test_mode, save_fold_path, use_saved_data_stuff, save_data_stuff, classi_probs_label_str):
    filename = "%sclass_imb~%s~%s~%s~%s.pickle" % (save_fold_path, class_imb_flag, prob_trans_type, classi_probs_label_str, test_mode)
    if use_saved_data_stuff and os.path.isfile(filename):
        print("loading class imb for %s and %s; classi_probs_label_info = %s, test mode = %s" % (class_imb_flag, prob_trans_type, classi_probs_label_str, test_mode))
        with open(filename, 'rb') as f:
            nonlin, out_vec_size_list, cw_list = pickle.load(f)
    else:
        if prob_trans_type == "lp":    
            nonlin = 'softmax'
            out_vec_size_list = num_classes_list
            cw_list = [None]
        elif prob_trans_type == "di":    
            nonlin = 'sigmoid'
            out_vec_size_list = num_classes_list
            cw_list = [None]
        elif prob_trans_type == "dc":    
            nonlin = 'softmax'
            out_vec_size_list = num_classes_list
            cw_list = [None]
        elif prob_trans_type == "br":    
            nonlin = 'sigmoid'
            out_vec_size_list = [1]*len(trainY_noncat_list)
            cw_list = [None]*len(trainY_noncat_list)

        if class_imb_flag:
            cw_list = []
            if prob_trans_type == "di":
                for num_classes_var, trainY_noncat in zip(num_classes_list, trainY_noncat_list):
                    cw_arr = np.empty([num_classes_var, 2])
                    for i in range(num_classes_var):
                        cw_arr[i] = class_weight.compute_class_weight('balanced', [0,1], trainY_noncat[:, i])
                    cw_list.append(cw_arr)
            elif prob_trans_type == "dc":
                for trainY_noncat in trainY_noncat_list:
                    cw_list.append(weights_cat(trainY_noncat))
            elif prob_trans_type == "lp" or prob_trans_type == "br": 
                for num_classes_var, trainY_noncat in zip(num_classes_list, trainY_noncat_list):
                    tr_uniq = np.arange(num_classes_var)
                    cw_arr = class_weight.compute_class_weight('balanced', tr_uniq, trainY_noncat)
                    cw_list.append(cw_arr)

        if save_data_stuff:        
            print("saving class imb for %s and %s; classi_probs_label_info = %s, test mode = %s" % (class_imb_flag, prob_trans_type, classi_probs_label_str, test_mode))
            with open(filename, 'wb') as f:
                pickle.dump([nonlin, out_vec_size_list, cw_list], f)

    if class_imb_flag:
        loss_func_list = []
        for cw_arr in cw_list:
            if prob_trans_type == "lp":
                loss_func_list.append(lp_categ_loss(cw_arr))
            elif prob_trans_type == "di":
                loss_func_list.append(multi_binary_loss(cw_arr))
            elif prob_trans_type == "dc":
                loss_func_list.append(multi_cat_w_loss(cw_arr))
            elif prob_trans_type == "br":    
                loss_func_list.append(br_binary_loss(cw_arr))
    else:
        if prob_trans_type == "lp":
            loss_func_list = ['categorical_crossentropy']
        elif prob_trans_type == "di":
            loss_func_list = ['binary_crossentropy']*len(trainY_noncat_list)
        elif prob_trans_type == "dc":
            loss_func_list = [multi_cat_loss()]*len(trainY_noncat_list)
        elif prob_trans_type == "br":    
            loss_func_list = ['binary_crossentropy']*len(trainY_noncat_list)

    return loss_func_list, nonlin, out_vec_size_list, cw_list

def transform_labels(data_trainY, prob_trans_type, test_mode, save_fold_path, NUM_CLASSES, data_prob_type, classi_probs_label_info, classi_probs_label_str, use_saved_data_stuff, save_data_stuff):
    filename = "%slabel_info~%s~%s~%s.pickle" % (save_fold_path, prob_trans_type, classi_probs_label_str, test_mode)
    if use_saved_data_stuff and os.path.isfile(filename):
        print("loading label info for %s; classi_probs_label_info = %s, test mode = %s" % (prob_trans_type, classi_probs_label_str, test_mode))
        with open(filename, 'rb') as f:
            trainY_list, trainY_noncat_list, num_classes_list, bac_map = pickle.load(f)
    else:
        if prob_trans_type == "lp":        
            lp_trainY, num_classes_var, bac_map, for_map = fit_trans_labels_powerset(data_trainY, NUM_CLASSES)
            print("num of LP classes: ", num_classes_var)
            trainY_noncat_list = [lp_trainY]
            trainY_list = [to_categorical(lp_trainY, num_classes=num_classes_var)]
            num_classes_list = [num_classes_var]
        elif prob_trans_type == "di" or prob_trans_type == "dc":
            num_classes_list, trainY_list = trans_labels_multi_hot_list(data_trainY, classi_probs_label_info)
            bac_map = None
            trainY_noncat_list = list(trainY_list)
        elif prob_trans_type == "br":
            if data_prob_type == 'multi-label':
                trainY_list = trans_labels_BR(data_trainY, NUM_CLASSES)
                num_classes_list = [2]*NUM_CLASSES
            elif data_prob_type == 'binary':
                trainY_list = trans_labels_bin_classi(data_trainY)
                num_classes_list = [2]
            bac_map = None
            trainY_noncat_list = list(trainY_list)
        if save_data_stuff:        
            print("saving label info for %s; classi_probs_label_info = %s, test mode = %s" % (prob_trans_type, classi_probs_label_str, test_mode))
            with open(filename, 'wb') as f:
                pickle.dump([trainY_list, trainY_noncat_list, num_classes_list, bac_map], f)
    return trainY_list, trainY_noncat_list, num_classes_list, bac_map
