#Question : yeomjohn@gmail.com
import tensorflow as tf
import numpy as np
import codecs


def intent_CNN(input,embedding_dim,filter_size,num_filters,initializer,sequence_length):
    """
    Args :
    input : input sentence, shape = [Batch, max_length, embedding_dim, 1]
    embedding_dim : word embedding dimension
    num_filters : number of filters. number of feature maps
    initializer : initializer method
    sequence_length : Max length of sentence

    Return : 
    h_pool_flat : max pooled vector
    """
    with tf.variable_scope("Intent_CNN_Layer"):
        pooled_outputs = []
        for i, _filter_ in enumerate(filter_size):
            with tf.variable_scope("conv-maxpool-{0}".format(_filter_)):
                filter_shape = [int(_filter_), embedding_dim, 1, num_filters]
    
                W = tf.get_variable("W", shape = filter_shape, initializer=initializer)
                b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name="b")

                conv = tf.nn.conv2d(
                    input,
                    W,
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv,b), name = "relu")

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1,sequence_length - int(_filter_) + 1, 1, 1],
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pooled)
            
        num_filters_total = num_filters * len(filter_size)
        h_pool = tf.concat(pooled_outputs,len(filter_size))
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    
    return h_pool_flat,num_filters_total
def bi_LSTM(input, lstm_hidden_state_dim, sequence_length,initializer):
    """
    Args :
    input : input sentence, shape = [Batch, max_length, embedding_dim, 1]
    lstm_hidden_state_dim : lstm cell dimension (hidden state & cell state)
    sequence_length : True length of sentence

    Return : 
    concat_output = concat of h_1 + ... + h_n
    last_output = h_n
    """
    with tf.variable_scope("word_Bi_LSTM_Layer"):
        batch_size = tf.shape(input)[0]
        lstm_cell = {}
        initial_state = {}
        for direction in ['forward','backward']:
            with tf.variable_scope(direction):
                lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
                        lstm_hidden_state_dim,
                        forget_bias = 1.0,
                        initializer = initializer,
                        state_is_tuple = True)
                
                initial_cell_state = tf.get_variable("initial_cell_state", shape = [1,lstm_hidden_state_dim], dtype = tf.float32, initializer = initializer)
                initial_output_state = tf.get_variable("initial_output_state", shape = [1,lstm_hidden_state_dim], dtype = tf.float32, initializer = initializer)

                c_states = tf.tile(initial_cell_state, tf.stack([batch_size,1]))
                h_states = tf.tile(initial_output_state, tf.stack([batch_size,1]))
                initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states,h_states)

        outputs , final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell['forward'],
                lstm_cell['backward'],
                input,
                sequence_length = sequence_length,
                initial_state_fw = initial_state['forward'],
                initial_state_bw = initial_state['backward'],
                dtype = tf.float32)

        output_fw, output_bw = outputs
        final_state_fw, final_state_bw = final_states
        concat_output = tf.concat([output_fw, output_bw], axis = 2)#(batch_size, max_len, lstm_hidden_state_dim*2)
        last_output = concat_output[:,-1,:] # (batch_size, lstm_hidden_state_dim*2)

        return concat_output, last_output   
def char_CNN(input,embedding_dim,filter_size, num_filters,initializer,sequence_length,max_len,dropout_keep_prob):
    """
    Args :
    input : input sentence, shape = [Batch, max_length, embedding_dim, 1]
    embedding_dim : char embedding dimension
    num_filters : number of filters. number of feature maps
    initializer : initializer method
    sequence_length : Max length of word 
    max_len : Max length of sentence

    Return : 
    h_pool_flat : max pooled vector
    """
    with tf.variable_scope("Char_CNN_Layer"):

        pooled_outputs = []
        for i, _filter_ in enumerate(filter_size):
            with tf.variable_scope("char-conv-maxpool-{0}".format(_filter_)):
                filter_shape = [1,int(_filter_), embedding_dim, 1, num_filters]
    
                W = tf.get_variable("W", shape = filter_shape, initializer=initializer)
                b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name="b")

                conv = tf.nn.conv3d(
                    input,
                    W,
                    strides=[1,1,1,1,1],
                    padding="VALID",
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")

            with tf.variable_scope("dropout_layer_after_char_CNN"):
                dropout_conv = tf.nn.dropout(h, dropout_keep_prob)

            with tf.variable_scope("char_max_pooling_layer"):
                pooled = tf.nn.max_pool3d(
                    dropout_conv,
                    ksize=[1,1,sequence_length - int(_filter_) + 1, 1, 1],
                    strides=[1,1,1,1,1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_size)
        h_pool = tf.concat(pooled_outputs,len(filter_size))
        h_pool_flat = tf.reshape(h_pool, [-1,max_len,num_filters_total])
    
    return h_pool_flat

class Text_CNN_LSTM_CRF(object):
    """
    input_word_x : input sentence(word) , shape = [Batch, max sentence length]
    input_char_x : input word (char), shape = [Batch, max sentence length, max word length]
    input_word_lengths : True sentence lengths
    input_intent_y : speech act labels
    input_slot_y : slot BIO tag labels
    dropout_keep_prob : drop out probability
    """
    def __init__(self, dataset, parameters):
        initializer = tf.contrib.layers.xavier_initializer()
        self.input_word_x = tf.placeholder(tf.int32, shape=[None,dataset.max_len], name="input_word_x")#[?,]
        self.input_char_x = tf.placeholder(tf.int32, shape=[None,dataset.max_len,dataset.max_word_len], name="input_char_x")
        self.input_word_lengths = tf.placeholder(tf.int32, shape = [None], name="input_word_lengths")
        self.input_intent_y = tf.placeholder(tf.float32, shape=[None,1,dataset.intent_number_of_classes], name="input_intent_y")#[15,]
        self.input_slot_y = tf.placeholder(tf.float32, shape=[None,dataset.max_len,dataset.slot_number_of_classes], name="input_slot_y")#[15,]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")#constant

        l2_loss = tf.constant(0.0)

        with tf.variable_scope("char_embedding_layer"):
            self.char_embedding_weights = tf.get_variable("char_embedding_weights",
                    shape = [dataset.char_size, parameters['char_embedding_dim']],
                    initializer = initializer)
            embedded_char = tf.nn.embedding_lookup(self.char_embedding_weights, self.input_char_x, name="embedded_chars") #shape = [Batch,max sentence length, max word length, char embedding_dim]
            embedded_char_expanded = tf.expand_dims(embedded_char, -1)

        if parameters['feature_extract_mode'] == 'cnn':
            with tf.variable_scope("char_convolution_layer"):
                char_feature = char_CNN(
						#shape = [Batch,max sentence length, num of filters] 
                    embedded_char_expanded,
                    parameters["char_embedding_dim"],
                    parameters["filter_size"],
                    parameters["num_filters"],
                    initializer,
                    dataset.max_word_len,
                    dataset.max_len,
                    self.dropout_keep_prob)

"""
Character Feature Extractor by using Bi-lstm
Extra codes are needed
*1) actual input Character lengths info
######################################################################################################################################
        elif parameters['feature_extract_mode'] == 'lstm':
            input_char_len = tf.reshape(self.input_char_lengths, [-1])
            embedded_char_ = tf.reshape(embedded_char, [-1, dataset.max_word_len, parameters['char_embedding_dim']])

            with tf.variable_scope('char_lstm_layer'):
                char_lstm_outputs, char_lstm_last_output = bi_LSTM(
                        embedded_char_,
                        parameters['char_lstm_hidden_state_dim'],
                        input_char_len,
                        initializer)
                char_feature = tf.reshape(char_lstm_last_output, [-1, dataset.max_len, 2*parameters['char_lstm_hidden_state_dim']])
######################################################################################################################################
"""     
        with tf.variable_scope("word_embedding_layer"):
            self.embedding_weights = tf.get_variable("embedding_weights",
                    shape = [dataset.vocabulary_size,parameters["word_embedding_dim"]],
                    initializer = initializer)
            
            embedded_tokens = tf.nn.embedding_lookup(self.embedding_weights, self.input_word_x, name="embedded_tokens") #shape = [Batch, max sentence length, word embedding_dim]
        
        with tf.variable_scope("concatentate_char_word_embedding"):
            token_lstm_input = tf.concat([char_feature,embedded_tokens], axis=2, name="token_fc_input") #shape = [Batch, max sentence length, word embedding_dim + char embedding_dim]

        with tf.variable_scope("token_dropout"):
            token_lstm_input_drop = tf.nn.dropout(token_lstm_input, self.dropout_keep_prob, name="token_lstm_input_drop")

        with tf.variable_scope("token_lstm_layer"):
            lstm_outputs,lstm_last_output = bi_LSTM(
                    token_lstm_input_drop,
                    parameters['lstm_hidden_state_dim'],
                    self.input_word_lengths,
                    initializer)
            #lstm_outputs : shape = [Batch,max sentence length, embedding_dim]
            #lstm_last_output : shape = [Batch, embedding_dim] 

        with tf.variable_scope("Intent_Convolution_Layer"):
            lstm_cnn_input = tf.expand_dims(lstm_outputs,-1)
            cnn_feature, num_filters_total = intent_CNN(
                    lstm_cnn_input,
                    lstm_cnn_input.get_shape().as_list()[2],#max sentence length
                    parameters["word_filter_size"],
                    parameters["word_num_filters"],
                    initializer,
                    dataset.max_len)
       	#cnn_feature : max pooled feature vector

        with tf.variable_scope("cnn_drop_out"):
            h_drop = tf.nn.dropout(cnn_feature, self.dropout_keep_prob)

        with tf.variable_scope("Intent_Classifier"):
            W = tf.get_variable("W",
                    shape = [num_filters_total, dataset.intent_number_of_classes],
                    initializer = initializer)
            b = tf.Variable(tf.constant(0.1, shape = [dataset.intent_number_of_classes]), name = "b")
            
            self.intent_scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            self.intent_predictions = tf.argmax(tf.nn.softmax(self.intent_scores),1,name = "predictions")

        with tf.variable_scope("Slot_Filling_Bi_LSTM_1"):
            lstm_output = tf.reshape(lstm_outputs,[-1,2*parameters['lstm_hidden_state_dim']]) #batch*leng , dim
            W = tf.get_variable("W",
                    shape = [2 * parameters['lstm_hidden_state_dim'], parameters['lstm_hidden_state_dim']],
                    initializer = initializer)
            b = tf.Variable(tf.constant(0.0 , shape = [parameters['lstm_hidden_state_dim']]), name= "b")
            before_slot_labeling = tf.nn.xw_plus_b(lstm_output, W,b,name="before_slot_tagging")

        with tf.variable_scope("Slot_Filling_Bi_LSTM_2"):
            W = tf.get_variable("W",
                    shape = [parameters['lstm_hidden_state_dim'], dataset.slot_number_of_classes],
                    initializer = initializer)
            b = tf.Variable(tf.constant(0.0 , shape = [dataset.slot_number_of_classes]), name= "b")

            self.slot_scores = tf.nn.xw_plus_b(before_slot_labeling, W, b, name="slot_scores")
            self.unary_scores = tf.reshape(self.slot_scores, [-1, dataset.max_len, dataset.slot_number_of_classes], name="unary_scores")
            
        with tf.variable_scope("intent_loss"):
            self.intent_losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.intent_scores, labels = self.input_intent_y)
            self.intent_loss = tf.reduce_mean(self.intent_losses)
        with tf.variable_scope("intent_accuracy"):
            self.input_intent_y_squeeze = tf.squeeze(self.input_intent_y,[1])
            correct_predictions = tf.equal(self.intent_predictions, tf.argmax(self.input_intent_y_squeeze,1))
            self.intent_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.variable_scope("CRF_Layer"):
            crf_input = tf.reshape(self.slot_scores, [-1, dataset.max_len, dataset.slot_number_of_classes], name="crf_input")
            crf_input_length = self.input_word_lengths
            crf_tag_input = tf.argmax(self.input_slot_y,2)
            crf_tag_input = tf.cast(crf_tag_input, tf.int32)
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                    crf_input, crf_tag_input, crf_input_length)
            self.slot_log_likelihood = log_likelihood
            self.slot_transition_params = transition_params
            
        with tf.variable_scope("slot_loss"):
            self.slot_loss = tf.reduce_mean(-log_likelihood)

        with tf.variable_scope("total_loss"):
            self.loss =  self.intent_loss +  self.slot_loss
    def load_pretrained_token_embeddings(self, sess, dataset, parameters, path):
        print("Load Pretrained Word Embeddings")
        token_to_vector = {}
        f = codecs.open(path, 'r', 'UTF-8')
        for line in f:
            line = line.strip()
            word = line.split('\t')[0]
            embed = line.split('\t')[-1].split(' ')
            token_to_vector[word] = embed
                                      
        init_weight = sess.run(self.embedding_weights.read_value())
        for token in dataset.token_to_index.keys():
            if token in token_to_vector.keys():
                init_weight[dataset.token_to_index[token]] = token_to_vector[token]
                                                              
        sess.run(self.embedding_weights.assign(init_weight))
