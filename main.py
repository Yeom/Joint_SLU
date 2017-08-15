import tensorflow as tf
import numpy as np
import train
import random
import os
import time
import datetime
import data_helpers
import configparser
import evaluate
from cnn_lstm_crf import Text_CNN_LSTM_CRF
from tensorflow.contrib import learn
import pickle

def load_parameters(parameter_path = os.path.join('.','parameters.ini')):
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameter_path)
    nested_parameters = data_helpers.convert_configarser_to_dict(conf_parameters)
    parameters = {}

    for s,k in nested_parameters.items():
        parameters.update(k)
    for k,v in parameters.items():
        if k in ['filter_size','word_filter_size']:
            parameters[k] = v.split(',')
        if k in ['dropout_rate','learning_rate','l2_reg_lambda']:
            parameters[k] = float(v)
        if k in ['jaso_embedding_dim','char_embedding_dim','word_embedding_dim','lstm_hidden_state_dim','word_num_filters','num_filters','patience','maximum_number_of_epochs','number_of_cpu_threads','batch_size','evaluation_every','checkpoint_every']:
            parameters[k] = int(v)
    return parameters,conf_parameters
            
def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    #Get parameters from parameters.ini file
    parameters, conf_parameters = load_parameters()
    #Get Dataset
    dataset = data_helpers.Dataset()
    dataset.load_train_and_valid_dataset(parameters)
    #Create Graph and session
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
                inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
                device_count={'CPU' : 1, 'GPU' : 1},
                allow_soft_placement=True,
                log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        results = {}
        results['epoch'] = {}
        start_time = time.time()
        model_name = '{0}_model'.format(data_helpers.get_current_time())

        output_folder         = os.path.join('./','output')
        create_folder(output_folder)
        stats_graph_folder     = os.path.join(output_folder,model_name)
        create_folder(stats_graph_folder)
        model_folder        = os.path.join(stats_graph_folder,'model')
        create_folder(model_folder)
        ckpt_folder            = os.path.join(model_folder,'ckpt')
        create_folder(ckpt_folder)
        with open(os.path.join(model_folder,'parameters.ini'), 'w') as parameters_file:
            conf_parameters.write(parameters_file)
        pickle.dump(dataset, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))


        cnn_lstm_crf = Text_CNN_LSTM_CRF(dataset, parameters)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn_lstm_crf.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        model_saver = tf.train.Saver(max_to_keep = parameters['maximum_number_of_epochs'])
        sess.run(tf.global_variables_initializer())
        
        if parameters['pretrained_word_embedding'] != 'False' and parameters['word_type'] != 'pos':
            print("Get Non-pos Tagged W.E")
            word_embedding_path = parameters['word_embedding_path']
            cnn_lstm_crf.load_pretrained_token_embeddings(sess, dataset, parameters, word_embedding_path)

        bad_counter = 0
        previous_best_total_score = 0
        epoch = 0
        Early_Flag = False
        while True:
            epoch += 1
            print('\nStarting epoch {0}'.format(epoch))

            if epoch != 0:
                #Training
                sequence_numbers = list(range(len(dataset.token_indices_padded['train'])))
                batches = data_helpers.batch_iter(sequence_numbers, parameters['batch_size'], parameters['maximum_number_of_epochs'])
                for batch in batches:
                    current_step = tf.train.global_step(sess, global_step)
                    x_batch, x_char_batch,intent_y_batch, slot_y_batch, sequence_legnth_batch = data_helpers.get_batch(dataset, batch, 'train')
                    transition_params = train.train_step(sess, x_batch, x_char_batch,intent_y_batch,slot_y_batch, sequence_legnth_batch,cnn_lstm_crf, parameters,train_op,global_step)
                #Prediction
                    if current_step % parameters['evaluation_every'] == 0 and current_step != 0:

                        all_intent_pred, all_intent_y,all_slot_pred, all_slot_y, sequence_lengths = train.predict_labels(sess, dataset,cnn_lstm_crf, parameters,train_op, global_step,epoch,current_step,stats_graph_folder,transition_params)

                        evaluate.evaluate_model(results,dataset,all_slot_pred,all_slot_y, all_intent_pred, all_intent_y,sequence_lengths,stats_graph_folder, current_step, parameters)

                        model_saver.save(sess,os.path.join(model_folder,'ckpt'),global_step=global_step)

                        total_valid_f1_score = results['epoch'][current_step]['total']['valid']['f1_score']['micro']

                        if total_valid_f1_score > previous_best_total_score:
                            bad_counter = 0
                            previous_best_total_score = total_valid_f1_score
                        else:
                            bad_counter += 1
                            print(bad_counter)

                        if bad_counter > parameters['patience']:
                            print('EarlyStop\n')
                            Early_Flag = True
                            break
            if Early_Flag:
                break
            if epoch > parameters['maximum_number_of_epochs']:
                break

        end_time = time.time()
        print("Training Time {0}".format(end_time-start_time))
        evaluate.save_results(results,stats_graph_folder)
    sess.close()
if __name__ == "__main__":
    main()
