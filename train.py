import os
import tensorflow as tf
import numpy as np
import sklearn.metrics
import codecs
import data_helpers
def train_step(sess, x_batch, x_char_batch,intent_y_batch,slot_y_batch, sequence_length,model, parameters,train_op, global_step):

    #print(np.asarray(slot_y_batch).shape)
    feed_dict = {
            model.input_word_x : x_batch,
            model.input_char_x : x_char_batch,
            model.input_slot_y : slot_y_batch,
            model.input_intent_y : intent_y_batch,
            model.input_word_lengths : sequence_length,
            model.dropout_keep_prob : parameters["dropout_rate"]
            }
    _, step, intent_loss,slot_loss, total_loss, transition_params= sess.run(
            [train_op, global_step, model.intent_loss, model.slot_loss,model.loss,model.slot_transition_params],
            feed_dict)
    if step % 100 == 0:
        print("{0} step : intet loss = {1} slot loss = {2} total loss = {3}".format(step, intent_loss,  slot_loss, total_loss))
    return transition_params
     
def prediction_step(sess, dataset,x_batch, x_char_batch,intent_y_batch,slot_y_batch,sequence_length, data_type, model, parameters, epoch,current_step,output_folder,transition_params):
    all_intent_predictions = []
    all_slot_predictions = []
    all_intent_y_true = []
    all_slot_y_true = []
    output_filepath = os.path.join(output_folder,'{1}_{2}_{0}.txt'.format(data_type, epoch,current_step))
    output_file = codecs.open(output_filepath,'w','UTF-8')

    feed_dict = {
            model.input_word_x : x_batch,
            model.input_char_x : x_char_batch,
            model.input_slot_y : slot_y_batch,
            model.input_intent_y : intent_y_batch,
            model.input_word_lengths : sequence_length,
            model.dropout_keep_prob : 1.0
            }
    
    intent_scores, intent_predictions,slot_scores = sess.run([model.intent_scores, model.intent_predictions, model.unary_scores], feed_dict)


    _1 = intent_predictions
    _2 = intent_y_batch
    _3 = []
    _4 = slot_y_batch
    _5 = sequence_length
    intent_predictions = intent_predictions.tolist()

    intent_sequence = np.argmax(np.asarray(intent_y_batch), axis = 2)
    slot_sequence    = np.argmax(np.asarray(slot_y_batch), axis = 2)
    max_len = dataset.max_len
    for sentence, intent, intent_pred, slot_seq, slot_score, seq_len in zip(x_batch, intent_sequence, intent_predictions, slot_sequence, slot_scores,sequence_length):
        sentence = sentence[:seq_len]
        intent = intent[0]
        intent_pred = intent_pred
        slot_seq = slot_seq[:seq_len]
        slot_score = slot_score[:seq_len]

        viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
                slot_score, transition_params)
        pad_viterbi_sequence = viterbi_sequence + [0] * (max_len - len(viterbi_sequence))
        _3.append(pad_viterbi_sequence)
        output_string = ''
        for idx in range(seq_len):
            if idx == 0:
                token = "ACT"
                label = dataset.index_to_intent_label[intent]
                pred  = dataset.index_to_intent_label[intent_pred]
                all_intent_predictions.append(intent_pred)
                all_intent_y_true.append(intent)
                output_string += token + '\t'+ label + '\t' +pred+'\n'
            token = dataset.index_to_token[sentence[idx]]
            label = dataset.index_to_slot_label[slot_seq[idx]]
            pred  = dataset.index_to_slot_label[viterbi_sequence[idx]]
            all_slot_predictions.append(viterbi_sequence[idx])
            all_slot_y_true.append(slot_seq[idx])
            output_string += token+'\t'+label+'\t'+pred+'\n'
        output_file.write(output_string)

    output_file.close()
    return _1, _2, _3, _4, _5

def predict_labels(sess, dataset,  model, parameters,train_op, global_step,epoch,current_step, output_folder,transition_params):
    intent_y_pred = {}
    intent_y_true = {}
    slot_y_pred = {}
    slot_y_true = {}
    sequence_lengths = {}
    for data_type in ['train','valid','test']:
        total_batch = list(range(len(dataset.token_indices_padded[data_type])))
        x_batch, x_char_batch, intent_y_batch, slot_y_batch, sequence_length = data_helpers.get_batch(dataset, total_batch, data_type)

        intent_y_pred[data_type], intent_y_true[data_type],slot_y_pred[data_type], slot_y_true[data_type], sequence_lengths[data_type] = prediction_step(sess, dataset,x_batch,x_char_batch, intent_y_batch,slot_y_batch,sequence_length, data_type, model, parameters,epoch,current_step,output_folder,transition_params)
    
    return intent_y_pred, intent_y_true, slot_y_pred, slot_y_true, sequence_lengths
