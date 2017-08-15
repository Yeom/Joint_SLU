import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics
import os
import time
import json
def calculate_score(slot_pred, slot_y, intent_pred, intent_y, slot_label_indices, intent_label_indices, sequnce_length,data_type):
    results = {}
    results['total'] = {}
    results['total'][data_type] = {}
    results['intent'] = {}
    results['intent'][data_type] = {}
    results['slot'] = {}
    results['slot'][data_type] = {}
    results['total'][data_type]['f1_score'] = {}
    results['total'][data_type]['f1_score']['micro'] = {}
    results['total'][data_type]['accuracy_score'] = {}
    results['slot'][data_type]['f1_score'] = {}
    results['slot'][data_type]['f1_score']['micro'] = {}
    results['slot'][data_type]['accuracy_score'] = {}
    results['intent'][data_type]['f1_score'] = {}
    results['intent'][data_type]['f1_score']['micro'] = {}
    results['intent'][data_type]['accuracy_score'] = {}
    results['intent'][data_type]['accuracy'] = {}
    slot_y = np.argmax(np.asarray(slot_y), axis = 2)
    intent_y = np.argmax(np.asarray(intent_y), axis = 2)
    intent_pred = intent_pred.tolist()
    intent_match = 0.0
    slot_match = 0.0
    total_match = 0.0
    utter = 0.0
    for s_pred, s_y, i_pred, i_y, seq_len in zip(slot_pred, slot_y, intent_pred, intent_y, sequnce_length):
        intent_flag = False
        slot_flag = False
        utter += 1
        if i_y[0] != i_pred:
            intent_flag = True
        for idx in range(seq_len):
            if s_pred[idx] != s_y[idx]:
                slot_flag = True

        if slot_flag == False and intent_flag == False:
            intent_match += 1
            slot_match += 1
            total_match += 1
        elif slot_flag == False and intent_flag == True:
            slot_match += 1
        elif slot_flag == True and intent_flag == False:
            intent_match += 1
        else:
            pass

    intent_acc = intent_match / utter
    slot_acc = slot_match / utter
    total_acc = total_match / utter
    results['total'][data_type]['f1_score']['micro'] =  total_acc
    results['total'][data_type]['accuracy_score'] = total_acc
    results['slot'][data_type]['f1_score']['micro'] =  slot_acc
    results['slot'][data_type]['accuracy_score'] = slot_acc
    results['intent'][data_type]['f1_score']['micro'] =  intent_acc
    results['intent'][data_type]['accuracy_score'] = intent_acc
    results['intent'][data_type]['accuracy'] = sklearn.metrics.accuracy_score(intent_y, intent_pred)

    return results
def plot_score_vs_epoch(results, stats_graph_folder, score, parameters,task,current_step):
    step_idxs = sorted(results['epoch'].keys())

    f1_dict_all = {}
    for data_type in ['train','valid','test']:
        f1_dict_all[data_type] = {}
        f1_dict_all[data_type][task] = []
    for step in step_idxs:
        result_epoch = results['epoch'][step][task]

        for data_type in ['train', 'valid', 'test']:
            f1_dict_all[data_type][task].append(result_epoch[data_type][score])
            #f1_dict_all[data_type][{'f1_score' : }, {'accuracy_score' : }]
    #Plot
    plt.figure()
    plot_handles = []
    f1_all = {}
    f1_all[task] = {}
    for data_type in ['train', 'valid', 'test']:
        f1_all[task][data_type] = {}
        if data_type not in results:
            results[data_type] = {}
        if task not in results[data_type]:
            results[data_type][task] = {}
        if score == 'f1_score':
            f1 = [f1_dict['micro'] for f1_dict in f1_dict_all[data_type][task]] #Accumulate f1_micro step by step
        else:
            f1 = [score_value for score_value in f1_dict_all[data_type][task]] # f1 = accuracy

        results[data_type][task]['best_{0}'.format(score)] = max(f1)
        results[data_type][task]['step_for_best_{0}'.format(score)] = 1000 * (int(np.asarray(f1).argmax())+ 1)#Validation set best score epoch
        f1_all[task][data_type] = f1
        plot_handles.extend(plt.plot(step_idxs, f1, '-', label=data_type + ' (max : {0:.4f})'.format(results[data_type][task]['best_{0}'.format(score)])))


    #RECORD the best value
    best_step = results['valid'][task]['step_for_best_{0}'.format(score)]
    if task == 'intent':
        c = 'k'
    else:
        c = 'm'
    plt.axvline(x=best_step, color=c, linestyle=':') #Add a vertical line at best epoch for valid
    for dataset_type in ['train', 'valid', 'test']:
        best_score_based_on_valid = f1_all[task][data_type][int(best_step/1000) - 1]
        results[dataset_type][task]['best_{0}_based_on_valid'.format(score)] = best_score_based_on_valid
        if dataset_type == 'test':
            plot_handles.append(plt.axhline(y=best_score_based_on_valid, label=dataset_type + ' (best: {0:.4f})'.format(best_score_based_on_valid),color=c, linestyle=':'))

        else:
            plt.axhline(y=best_score_based_on_valid, label='{0:.4f}'.format(best_score_based_on_valid), color=c, linestyle=':')
    title = '{0} vs step number\n'.format(score)
    xlabel = 'step number'
    ylabel = score
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(handles=plot_handles, loc=0)
    plt.savefig(os.path.join(stats_graph_folder, '{0}_vs_step.{1}'.format(score, "png")))
    plt.close()

def save_results(results, stats_graph_folder):
    json.dump(results, open(os.path.join(stats_graph_folder, 'results.json'), 'w'), indent = 4, sort_keys = True)
def evaluate_model(results, dataset,slot_pred, slot_y, intent_pred, intent_y,sequnce_length,stats_graph_folder, current_step, parameters, task='total'):
    if current_step not in results['epoch']:results['epoch'][current_step] = {}
    if task not in results['epoch'][current_step] : results['epoch'][current_step][task] = {}
    if 'slot' not in results['epoch'][current_step] : results['epoch'][current_step]['slot'] = {}
    if 'intent' not in results['epoch'][current_step] : results['epoch'][current_step]['intent'] = {}
    result_update = {}
    intent_label_indices = list(dataset.index_to_intent_label)
    slot_label_indices = list(dataset.index_to_slot_label)
    for data_type in ['train','valid','test']:
        result_update = {}
        slot_y_pred_original = slot_pred[data_type]
        slot_y_true_original = slot_y[data_type]
        intent_y_pred_original = intent_pred[data_type]
        intent_y_true_original = intent_y[data_type]
        sequnce_length_original = sequnce_length[data_type]

        result_update[data_type] = calculate_score(slot_y_pred_original, slot_y_true_original, intent_y_pred_original, intent_y_true_original, slot_label_indices, intent_label_indices, sequnce_length_original,data_type)
        results['epoch'][current_step]['total'].update(result_update[data_type]['total'])
        results['epoch'][current_step]['slot'].update(result_update[data_type]['slot'])
        results['epoch'][current_step]['intent'].update(result_update[data_type]['intent'])
    plot_score_vs_epoch(results, stats_graph_folder, 'f1_score', parameters,task,current_step)
    plot_score_vs_epoch(results, stats_graph_folder, 'accuracy_score', parameters,task,current_step)
    plot_score_vs_epoch(results, stats_graph_folder, 'f1_score', parameters,'slot',current_step)
    plot_score_vs_epoch(results, stats_graph_folder, 'accuracy_score', parameters,'slot',current_step)
    plot_score_vs_epoch(results, stats_graph_folder, 'f1_score', parameters,'intent',current_step)
    plot_score_vs_epoch(results, stats_graph_folder, 'accuracy_score', parameters,'intent',current_step)
 
    save_results(results, stats_graph_folder)

