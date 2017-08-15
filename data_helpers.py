import operator
import os
import collections
import codecs
import time
import sklearn.preprocessing
import numpy as np
from hangul_utils import split_syllables, join_jamos
def get_current_time():
    return(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))
def convert_configarser_to_dict(config):
    config_parser_dict = {s : dict(config.items(s)) for s in config.sections()}
    return config_parser_dict
def order_dictionary(dictionary, mode, reverse=False):
    '''
    Order a dictionary by 'key' or 'value'.
    mode should be either 'key' or 'value'
    http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
    '''

    if mode =='key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(0),
                                              reverse=reverse))
    elif mode =='value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=operator.itemgetter(1),
                                              reverse=reverse))
    elif mode =='key_value':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              reverse=reverse))
    elif mode =='value_key':
        return collections.OrderedDict(sorted(dictionary.items(),
                                              key=lambda x: (x[1], x[0]),
                                              reverse=reverse))
    else:
        raise ValueError("Unknown mode. Should be 'key' or 'value'")

def reverse_dictionary(dictionary):
    '''
    http://stackoverflow.com/questions/483666/python-reverse-inverse-a-mapping
    http://stackoverflow.com/questions/25480089/right-way-to-initialize-an-ordereddict-using-its-constructor-such-that-it-retain
    '''
    if type(dictionary) is collections.OrderedDict:
        return collections.OrderedDict([(v, k) for k, v in dictionary.items()])
    else:
        return {v: k for k, v in dictionary.items()}

def pad_list(old_list, padding_size, padding_value):
    if not padding_size >= len(old_list):
        return old_list[:padding_size]
    else:
        return old_list + [padding_value] * (padding_size-len(old_list))

def get_batch(dataset, batch,data_type):
    batch_sequence = list(batch)
    x_batch = []
    intent_y_batch = []
    slot_y_batch = []
    x_char_batch = []
    sequence_length_batch = []
    for batch_idx in batch_sequence:
        x_batch.append(dataset.token_indices_padded[data_type][batch_idx])
        x_char_batch.append(dataset.character_indices_padded[data_type][batch_idx])
        intent_y_batch.append(dataset.intent_label_vector_indices[data_type][batch_idx])#[[class number]]
        slot_y_batch.append(dataset.slot_label_vector_indices[data_type][batch_idx])#[[slot1],[slot2],..,]
        sequence_length_batch.append(dataset.sequence_lengths[data_type][batch_idx][0])
    return x_batch, x_char_batch,intent_y_batch, slot_y_batch,sequence_length_batch
def batch_iter(sequence_numbers, batch_size, num_epochs, shuffle = True):
    sequence = np.array(sequence_numbers)
    sequence_size = len(sequence)
    num_batches_per_epoch = int((sequence_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(sequence_size))
            shuffled_sequence = sequence[shuffle_indices]
        else:
            shuffled_sequence = sequence

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index     = min((batch_num+1) * batch_size, sequence_size)
            yield shuffled_sequence[start_index:end_index]
def pad_jaso(temp_jaso_indices,max_char_len, PADDING_JASO_INDEX):
    _ = []
    for temp_jaso in temp_jaso_indices:
        _.append(pad_list(temp_jaso,max_char_len,PADDING_JASO_INDEX))
    return _
def get_jaso(word):
    jaso_list = []
    for char in word:
        for jaso in list(split_syllables(char)):
            jaso_list.append(jaso)
    return jaso_list
def get_jaso_index(word, jaso_to_index):
    jaso_list = []
    for char in word:
        for jaso in list(split_syllables(char)):
            jaso_list.append(jaso_to_index[jaso])
    return jaso_list

class Dataset(object):
    def __init__(self):
        pass

    def parse_dataset(self, filepath):
        token_count = collections.defaultdict(lambda:0)        
        character_count = collections.defaultdict(lambda:0)        
        intent_label_count = collections.defaultdict(lambda:0)        
        slot_label_count = collections.defaultdict(lambda:0)        
        jaso_count =  collections.defaultdict(lambda:0)
        max_len = 0
        max_word_len = 0
        max_char_len = 0
        line_count = -1
        tokens = []
        intent_labels = []
        slot_labels = []
        token_lengths = []
        new_token_sequence = []
        new_intent_label_sequence = []
        new_slot_label_sequence = []
        f = codecs.open(filepath,'r','UTF-8')
        for line in f:
            line_count += 1
            line = line.strip().split(' ')
            if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
                if len(new_token_sequence) > 0:
                    if len(new_token_sequence) > max_len:
                        max_len = len(new_token_sequence)
                    for token in new_token_sequence:
                        if len(token) > max_word_len:
                            max_word_len = len(token)
                        for char in token:
                            if len(split_syllables(char)) > max_char_len:
                                max_char_len = len(split_syllables(char))
                    tokens.append(new_token_sequence)
                    intent_labels.append(new_intent_label_sequence)
                    slot_labels.append(new_slot_label_sequence)
                    new_token_sequence = []
                    new_intent_label_sequence = []
                    new_slot_label_sequence = []
                continue
            if 'ACT' == line[0]:
                intent_label = str(line[-1])
                intent_label_count[intent_label] += 1
                new_intent_label_sequence.append(intent_label)
            else:
                slot_label = str(line[-1])
                slot_label_count[slot_label] += 1
                new_slot_label_sequence.append(slot_label)
            token = str(line[0])
            token_count[token] += 1
            new_token_sequence.append(token)

            for character in token:
                character_count[character] += 1
                for jaso in list(split_syllables(character)):
                    jaso_count[jaso] += 1

        if len(new_token_sequence) > 0:
            if len(new_token_sequence) > max_len:
                max_len = len(new_token_sequence)
            for token in new_token_sequence:
                if len(token) > max_word_len:
                    max_word_len = len(token)
                for char in token:
                    if len(split_syllables(char)) > max_char_len:
                        max_char_len = len(split_syllables(char))
            intent_labels.append(new_intent_label_sequence)
            slot_labels.append(new_slot_label_sequence)
            tokens.append(new_token_sequence)
        return intent_labels,slot_labels, tokens, token_count, intent_label_count, slot_label_count,character_count, jaso_count,max_len, max_word_len, max_char_len

    def load_train_and_valid_dataset(self,parameters):
        start_time = time.time()
        print("---Load Train and Valid Dataset---")
        test_path =os.path.join(parameters['dataset_text_folder'],'test.txt' )
        train_path=os.path.join(parameters['dataset_text_folder'],'train.txt')
        valid_path=os.path.join(parameters['dataset_text_folder'],'valid.txt')
        self.PADDING_JASO_INDEX = 0
        self.UNK_TOKEN_INDEX = 0
        self.PADDING_CHARACTER_INDEX = 0
        self.UNK = "UNK"
        self.intent_unique_labels = []
        self.slot_unique_labels = []
        intent_labels = {}
        slot_labels = {}
        tokens = {}
        characters = {}
        jasos = {}
        token_lengths = {}
        intent_label_count = {}
        slot_label_count = {}
        token_count = {}
        character_count = {}
        jaso_count = {}
        intent_labels['test'], slot_labels['test'], tokens['test'] , token_count['test' ], intent_label_count['test'], slot_label_count['test'], character_count['test' ],jaso_count['test'],test_max_len, test_word_max_len ,test_char_max_len= self.parse_dataset(test_path)
        intent_labels['train'], slot_labels['train'], tokens['train'], token_count['train'], intent_label_count['train'], slot_label_count['train'],character_count['train'],jaso_count['train'],train_max_len, train_word_max_len ,train_char_max_len= self.parse_dataset(train_path)
        intent_labels['valid'], slot_labels['valid'], tokens['valid'], token_count['valid'], intent_label_count['valid'], slot_label_count['valid'],character_count['valid'],jaso_count['valid'],valid_max_len , valid_word_max_len , valid_char_max_len= self.parse_dataset(valid_path)

        if train_max_len > valid_max_len:
            max_len = train_max_len
        else:
            max_len = valid_max_len

        if train_word_max_len > valid_word_max_len:
            max_word_len = train_word_max_len
        else:
            max_word_len = valid_word_max_len
       
        if train_char_max_len > valid_char_max_len:
            max_char_len = train_char_max_len * max_word_len
        else:
            max_char_len = valid_char_max_len * max_word_len

        token_count['all'] = {}

        for token in list(token_count['train'].keys()) + list(token_count['valid'].keys()):
            token_count['all'][token] = token_count['train'][token] + token_count['valid'][token]
       
        character_count['all'] = {}

        for character in list(character_count['train'].keys()) + list(character_count['valid'].keys()):
            character_count['all'][character] = character_count['train'][character] + character_count['valid'][character]
        
        for token_sequence in tokens['test']:
            for idx in range(len(token_sequence)):
                if token_sequence[idx] not in token_count['all'].keys():
                    token_sequence[idx] = self.UNK

        jaso_count['all'] = {} 
        for jaso in list(jaso_count['train'].keys()) + list(jaso_count['valid'].keys()):
            jaso_count['all'][jaso] = jaso_count['train'][jaso] + jaso_count['valid'][jaso]
        
        intent_label_count['all'] = {}
        for label in list(intent_label_count['train'].keys()) + list(intent_label_count['valid'].keys()):
            intent_label_count['all'][label] = intent_label_count['train'][label] + intent_label_count['valid'][label]

        slot_label_count['all'] = {}
        for label in list(slot_label_count['train'].keys()) + list(slot_label_count['valid'].keys()):
            slot_label_count['all'][label] = slot_label_count['train'][label] + slot_label_count['valid'][label]

        token_count['all'] = order_dictionary(token_count['all'], 'value_key', reverse = True)
        intent_label_count['all'] = order_dictionary(intent_label_count['all'], 'key', reverse = False)
        slot_label_count['all'] = order_dictionary(slot_label_count['all'], 'key', reverse = False)
        character_count['all'] = order_dictionary(character_count['all'], 'value', reverse = True)
        jaso_count['all'] = order_dictionary(jaso_count['all'], 'value', reverse = True)

        token_to_index = {}
        token_to_index[self.UNK] = self.UNK_TOKEN_INDEX
        iteration_number = 0
        for token, count in token_count['all'].items():
            if iteration_number == self.UNK_TOKEN_INDEX :
                iteration_number += 1
            token_to_index[token] = iteration_number
            iteration_number += 1
        
        intent_label_to_index = {}
        iteration_number = 0
        
        for label, count in intent_label_count['all'].items():
            intent_label_to_index[label] = iteration_number
            iteration_number += 1
            self.intent_unique_labels.append(label)
        slot_label_to_index = {}
        iteration_number = 0
        
        for label, count in slot_label_count['all'].items():
            slot_label_to_index[label] = iteration_number
            iteration_number += 1
            self.slot_unique_labels.append(label)


        character_to_index = {}
        iteration_number = 0

        for character, count in character_count['all'].items():
            if iteration_number == self.PADDING_CHARACTER_INDEX:
                character_to_index['U'] = self.PADDING_CHARACTER_INDEX
                character_to_index['N'] = self.PADDING_CHARACTER_INDEX
                character_to_index['K'] = self.PADDING_CHARACTER_INDEX
                iteration_number += 1
            character_to_index[character] = iteration_number
            iteration_number += 1

        jaso_to_index = {}
        iteration_number = 0
        for jaso, count in jaso_count['all'].items():
            if iteration_number == self.PADDING_JASO_INDEX:
                jaso_to_index['U'] = self.PADDING_JASO_INDEX
                jaso_to_index['N'] = self.PADDING_JASO_INDEX
                jaso_to_index['K'] = self.PADDING_JASO_INDEX
                iteration_number += 1
            jaso_to_index[jaso] = iteration_number
            iteration_number += 1
#        print(character_to_index)
#        print(jaso_to_index)
        token_to_index = order_dictionary(token_to_index, 'value', reverse = False)
        index_to_token = reverse_dictionary(token_to_index)

        intent_label_to_index = order_dictionary(intent_label_to_index, 'value', reverse = False)
        index_to_intent_label = reverse_dictionary(intent_label_to_index)
        slot_label_to_index = order_dictionary(slot_label_to_index, 'value', reverse = False)
        index_to_slot_label = reverse_dictionary(slot_label_to_index)
       
        character_to_index = order_dictionary(character_to_index, 'value', reverse = False)
        index_to_character = reverse_dictionary(character_to_index)

        jaso_to_index = order_dictionary(jaso_to_index, 'value', reverse = False)
        index_to_jaso = reverse_dictionary(jaso_to_index)
        
        self.JASO_PAD = [0]*max_char_len
        self.CHAR_PAD = [0]*max_word_len
        token_indices = {}
        intent_label_indices = {}
        slot_label_indices = {}
        character_indices = {}
        jaso_indices = {}

        token_indices_padded = {}
        character_indices_padded = {}
        jaso_indices_padded = {}

        sequence_lengths = {}
        dataset = ['valid','train','test']
        for data_type in dataset:
            token_indices[data_type] = []
            characters[data_type] = []
            character_indices[data_type] = []
            jasos[data_type] = []
            jaso_indices[data_type] = []
            jaso_indices_padded[data_type] = []

            token_lengths[data_type] = []
            token_indices_padded[data_type] = []
            character_indices_padded[data_type] = []
            sequence_lengths[data_type] = []

            for token_sequence in tokens[data_type]:
                #For remove ACT token index
                _ = [token_to_index[token] for token in token_sequence]
                del _[0]
                token_indices[data_type].append(_)
                characters[data_type].append([list(token) for token in token_sequence])
#                print(characters[data_type])
                jasos[data_type].append([get_jaso(token) for token in token_sequence])
#                print(jasos[data_type])
                character_indices[data_type].append([[character_to_index[character] for character in token] for token in token_sequence])
#                print(character_indices[data_type])
                jaso_indices[data_type].append([get_jaso_index(token, jaso_to_index)  for token in token_sequence])  
#                print(jaso_indices[data_type])
                token_lengths[data_type].append([len(token) for token in token_sequence])
                #sequence_lengths[data_type].append([len(token_sequence)])
                sequence_lengths[data_type].append([len(token_sequence)-1]) #Except for "ACT"
                longest_token_length_in_sequence = max(token_lengths[data_type][-1])
                
                token_indices_padded[data_type].append(pad_list(token_indices[data_type][-1], max_len, self.UNK_TOKEN_INDEX))
                character_indices_padded[data_type].append(pad_list([pad_list(temp_token_indices, max_word_len, self.PADDING_CHARACTER_INDEX) for temp_token_indices in character_indices[data_type][-1]],max_len,self.CHAR_PAD))
                jaso_indices_padded[data_type].append(pad_list([pad_list(temp_token_indices, max_char_len, self.PADDING_JASO_INDEX) for temp_token_indices in jaso_indices[data_type][-1]], max_len, self.JASO_PAD))
                #[[w1, w2 , w3 ,w4],[]]
                #[[[j11,j12,j13],[j21,j22,j23], [j31,j32,j33]], []]
            PAD_ = len(slot_label_to_index.keys())+1
            intent_label_indices[data_type] = []
            slot_label_indices[data_type] = []
            for label_sequence in intent_labels[data_type]:
                intent_label_indices[data_type].append([intent_label_to_index[label] for label in label_sequence])
            for label_sequence in slot_labels[data_type]:
                #slot_label_indices[data_type].append([slot_label_to_index[label] for label in label_sequence])
                slot_label_indices[data_type].append(pad_list([slot_label_to_index[label] for label in label_sequence],max_len, PAD_))


        intent_label_binarizer = sklearn.preprocessing.LabelBinarizer()
        intent_label_binarizer.fit(range(max(index_to_intent_label.keys()) + 1))
        intent_label_vector_indices = {}


        slot_label_binarizer = sklearn.preprocessing.LabelBinarizer()
        slot_label_binarizer.fit(range(max(index_to_slot_label.keys()) + 1))#For Except 0 class
        slot_label_vector_indices = {}

        for data_type in dataset:
            intent_label_vector_indices[data_type] = []
            for label_indices_sequence in intent_label_indices[data_type]:
                intent_label_vector_indices[data_type].append(intent_label_binarizer.transform(label_indices_sequence))
        for data_type in dataset:
            slot_label_vector_indices[data_type] = []
            for label_indices_sequence in slot_label_indices[data_type]:
                slot_label_vector_indices[data_type].append(slot_label_binarizer.transform(label_indices_sequence))

        #-------
        self.tokens = tokens
        self.intent_labels = intent_labels
        self.slot_labels = slot_labels
        self.characters = characters
        self.jasos = jasos
        self.token_lengths = token_lengths
        self.sequence_lengths = sequence_lengths

        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.intent_label_to_index = intent_label_to_index
        self.index_to_intent_label = index_to_intent_label
        self.slot_label_to_index = slot_label_to_index
        self.index_to_slot_label = index_to_slot_label

        self.character_to_index = character_to_index
        self.index_to_character = index_to_character
        self.jaso_to_index = jaso_to_index
        self.index_to_jaso = index_to_jaso

        self.token_indices = token_indices
        self.intent_label_indices = intent_label_indices
        self.slot_label_indices = slot_label_indices
        self.character_indices = character_indices
        self.jaso_indices = jaso_indices
        self.token_indices_padded = token_indices_padded
        self.character_indices_padded = character_indices_padded
        self.jaso_indices_padded = jaso_indices_padded

        self.intent_label_vector_indices = intent_label_vector_indices
        self.slot_label_vector_indices = slot_label_vector_indices

        self.intent_number_of_classes = max(self.index_to_intent_label.keys()) + 1
        self.slot_number_of_classes = max(self.index_to_slot_label.keys()) + 1
        self.vocabulary_size = max(self.index_to_token.keys()) + 1
        self.char_size = max(self.index_to_character.keys()) + 1
        self.jaso_size = max(self.index_to_jaso.keys()) + 1
        self.max_len = max_len
        self.max_word_len = max_word_len
        self.max_char_len = max_char_len
#For F1-score computing???
#        self.unique_labels_of_interest = list(self.unique_labels)
#        self.unique_label_indices_of_interest = []
#        for lab in self.unique_labels_of_interest:
#            self.unique_label_indices_of_interest.append(label_to_index[lab])
        end_time = time.time()
        print(index_to_slot_label)
        print(index_to_intent_label)
        print('LOADED for {0:.2f} seconds'.format(end_time - start_time))
        #END#
