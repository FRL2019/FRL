#coding:utf-8
import ipdb
import pickle
import numpy as np
from copy import deepcopy
from utils import ten_fold_split_ind, index2data, save_pkl, load_pkl

#create table tag_actions5(text_num int, sent_num int, af_sent varchar(400), tag_sent varchar(400));
class Environment:
    def __init__(self, args):
        print('Initializing the Environment...')  
        self.domain = args.domain
        self.tag_dim = args.tag_dim
        self.dis_dim = args.dis_dim
        self.word_dim = args.word_dim
        self.num_words = args.num_words
        self.action_rate = args.action_rate
        self.use_act_rate = args.use_act_rate
        self.use_act_att = args.use_act_att
        self.reward_base = args.reward_base
        self.ra = args.reward_assign

        self.args = args
        self.word2vec = args.word2vec
        self.k_fold = args.k_fold
        self.k_fold_indices = args.k_fold_indices
        self.agent_mode = args.agent_mode

        self.terminal_flag = False
        self.train_text_idx = -1
        self.valid_text_idx = -1
        self.train_epoch_end_flag = False
        self.valid_epoch_end_flag = False
        self.max_data_sent_len = 0
        self.build_dict()
        self.get_fold_data()

        args.num_pos = len(self.pos_dict) + 1
        temp_size = self.train_steps * args.epochs + self.valid_steps
        if temp_size < args.replay_size:
            args.replay_size = temp_size
        if args.train_episodes == 0:
            args.train_episodes = self.num_train
        args.valid_episodes = self.num_valid
        args.train_steps = self.train_steps
        args.valid_steps = self.valid_steps


    def build_dict(self):
        self.pos_dict = {'PAD': 0}
        for domain in ['cooking', 'win2k', 'wikihow']:
            sent_data = load_pkl('data/%s_dependency.pkl' % domain)
            for sents in sent_data:
                for sent in sents:
                    for word, pos in sent:
                        if pos not in self.pos_dict:
                            self.pos_dict[pos] = len(self.pos_dict)

        print('len(pos_dict): %d' % len(self.pos_dict))


    def get_fold_data(self):
        if self.agent_mode == 'act':
            data = self.read_act_texts()
        else:
            data = self.read_arg_sents()
        print('\n\nGet new fold data')
        self.train_data = data['train'][self.args.fold_id]
        self.valid_data = data['valid'][self.args.fold_id]
        self.train_steps = len(self.train_data) * self.num_words
        self.valid_steps = len(self.valid_data) * self.num_words

        self.num_train = len(self.train_data)
        self.num_valid = len(self.valid_data)
        print('training texts: %d\tvalidation texts: %d' % (len(self.train_data), len(self.valid_data)))
        print('max_data_sent_len: %d' % self.max_data_sent_len)
        print('self.train_steps: %d\tself.valid_steps: %d\n\n' % (self.train_steps, self.valid_steps))



    def read_act_texts(self):
        text_data = load_pkl('data/%s_labeled_text_data.pkl' % self.domain)
        pos_data = load_pkl('data/%s_dependency.pkl' % self.domain)
        act_texts = []
        #ipdb.set_trace()
        for i in range(len(text_data)):
            act_text = {}
            act_text['tokens'] = text_data[i]['words']
            act_text['sents'] = text_data[i]['sents']
            act_text['acts'] = text_data[i]['acts']
            act_text['sent_acts'] = text_data[i]['sent_acts']
            act_text['word2sent'] = text_data[i]['word2sent']
            act_text['tags'] = np.ones(len(text_data[i]['words']), dtype=np.int32)
            act_text['act2related'] = {}
            for acts in text_data[i]['acts']:
                act_text['act2related'][acts['act_idx']] = acts['related_acts']
                act_text['tags'][acts['act_idx']] = acts['act_type'] + 1 # 2, 3, 4
            act_text['pos'] = []
            for sent in pos_data[i]:
                for word, pos in sent:
                    act_text['pos'].append(self.pos_dict[pos])

            self.create_matrix(act_text)
            act_texts.append(act_text)
        act_indices = ten_fold_split_ind(len(act_texts), self.k_fold_indices, self.k_fold)
        act_data = index2data(act_indices, act_texts)
        return act_data


    def read_arg_sents(self):
        indata = load_pkl('data/refined_%s_data.pkl' % self.domain)[-1]
        pos_data = load_pkl('data/%s_arg_pos.pkl' % self.domain)
        arg_sents = []
        # ipdb.set_trace()
        for i in range(len(indata)):
            for j in range(len(indata[i])):
                if len(indata[i][j]) == 0:
                    continue
                # -1 obj_ind refer to UNK
                words = indata[i][j]['last_sent'] + indata[i][j]['this_sent'] + ['UNK'] 
                pos = [self.pos_dict[p] for w, p in pos_data[i][j][0] + pos_data[i][j][1]] + [0]
                if len(words) != len(words):
                    ipdb.set_trace()
                    print('len(words) != len(words)')
                sent_len = len(words)
                act_inds = [a['act_idx'] for a in indata[i][j]['acts'] if a['act_idx'] < self.num_words]
                for k in range(len(indata[i][j]['acts'])):
                    act_ind = indata[i][j]['acts'][k]['act_idx']
                    obj_inds = indata[i][j]['acts'][k]['obj_idxs']
                    arg_sent = {}
                    arg_tags = np.ones(sent_len, dtype=np.int32)
                    if len(obj_inds[1]) == 0:
                        arg_tags[obj_inds[0]] = 2 # essential objects
                    else:
                        arg_tags[obj_inds[0]] = 4 # exclusive objects
                        arg_tags[obj_inds[1]] = 4 # exclusive objects
                    position = np.zeros(sent_len, dtype=np.int32)
                    position.fill(act_ind)
                    distance = np.abs(np.arange(sent_len) - position)
                    
                    arg_sent['tokens'] = words
                    arg_sent['tags'] = arg_tags
                    arg_sent['pos'] = deepcopy(pos)
                    arg_sent['act_ind'] = act_ind
                    arg_sent['distance'] = distance
                    arg_sent['act_inds'] = act_inds
                    arg_sent['obj_inds'] = obj_inds
                    self.create_matrix(arg_sent)
                    arg_sents.append(arg_sent)

        arg_indices = ten_fold_split_ind(len(arg_sents), self.k_fold_indices, self.k_fold)
        arg_data = index2data(arg_indices, arg_sents)
        return arg_data
        

    def create_matrix(self, sentence):
        #ipdb.set_trace()
        sent_vec = []
        for w in sentence['tokens']:
            if w in self.word2vec.vocab:
                sent_vec.append(self.word2vec[w])
            else:
                sent_vec.append(np.zeros(self.word_dim))

        sent_vec = np.array(sent_vec)
        pad_len = self.num_words - len(sent_vec)
        if self.agent_mode == 'act':
            if pad_len > 0:
                sent_vec = np.concatenate((sent_vec, np.zeros([pad_len, self.word_dim])))
                sentence['tags'] = np.concatenate((np.array(sentence['tags']), np.ones(pad_len, dtype=np.int32)))
                sentence['pos'].extend([0] * pad_len)
            else:
                sent_vec = sent_vec[: self.num_words]
                sentence['pos'] = sentence['pos'][: self.num_words]
                sentence['tokens'] = sentence['tokens'][: self.num_words]
                sentence['tags'] = np.array(sentence['tags'])[: self.num_words]

        else: # self.agent_mode == 'arg':
            distance = np.zeros([self.num_words, self.dis_dim])
            act_vec = sent_vec[sentence['act_ind']]  # word vector of the input action 
            attention = np.sum(sent_vec * act_vec, axis=1)  # attention between the input action and its context 
            attention = np.exp(attention)
            attention /= sum(attention)
            if pad_len > 0:
                sent_vec = np.concatenate((sent_vec, np.zeros([pad_len, self.word_dim])))
                sentence['tags'] = np.concatenate((np.array(sentence['tags']), np.ones(pad_len, dtype=np.int32)))
                sentence['pos'].extend([0] * pad_len)
                attention = np.concatenate((attention, np.zeros(pad_len)))
                for d in range(len(sentence['distance'])):
                    distance[d] = sentence['distance'][d]
            else:
                sent_vec = sent_vec[: self.num_words]
                sentence['tokens'] = sentence['tokens'][: self.num_words]
                sentence['tags'] = np.array(sentence['tags'])[: self.num_words]
                sentence['pos'] = sentence['pos'][: self.num_words]
                attention = attention[: self.num_words]
                for d in range(self.num_words):
                    distance[d] = sentence['distance'][d]
            #ipdb.set_trace()
            if self.use_act_att: # apply attention to word embedding
                sent_vec = attention.reshape(-1, 1) * sent_vec
            sent_vec = np.concatenate((sent_vec, distance), axis=1)

        sentence['sent_vec'] = sent_vec
        sentence['pos'] = np.array(sentence['pos'])[:, np.newaxis]
        sentence['tags'].shape = (self.num_words, 1)


    def restart(self, train_flag, init=False):
        if train_flag:
            if init:
                self.train_text_idx = -1
                self.train_epoch_end_flag = False
            self.train_text_idx += 1
            if self.train_text_idx >= len(self.train_data):
                self.train_epoch_end_flag = True
                print('\n\n-----train_epoch_end_flag = True-----\n\n')
                return
            self.current_text = self.train_data[self.train_text_idx%self.num_train]
            print('\ntrain_text_ind: %d of %d' % (self.train_text_idx, len(self.train_data)))
        else:
            if init:
                self.valid_text_idx = -1
                self.valid_epoch_end_flag = False
            self.valid_text_idx += 1
            if self.valid_text_idx >= len(self.valid_data):
                self.valid_epoch_end_flag = True
                print('\n\n-----valid_epoch_end_flag = True-----\n\n')
                return
            self.current_text = self.valid_data[self.valid_text_idx]
            print('\nvalid_text_ind: %d of %d' % (self.valid_text_idx, len(self.valid_data)))
        #ipdb.set_trace()
        self.text_vec = np.concatenate([self.current_text['sent_vec'], self.current_text['tags']], axis=1)
        self.state_beta = self.text_vec.copy() # NB!
        self.state_beta[:, -1] = 0
        self.state_alpha = np.concatenate([self.current_text['pos'], self.current_text['tags']], axis=1)
        self.state_alpha[:, -1] = 0
        self.terminal_flag = False

        
    def act(self, action, word_ind):
        '''
        TODO:   Perform an action and return its reward
        params: action: 0 for non-action words, 1 for action words
                word_ind: 0 ~ self.num_words - 1, indicating the current word
        return: reward: float, indicating the reward of the current action
        '''
        self.state_beta[word_ind, -1] = action + 1
        self.state_alpha[word_ind, -1] = action + 1
        
        t_a_count = sum(self.state_beta[: word_ind + 1, -1]) - (word_ind + 1)
        t_a_rate = float(t_a_count)/self.num_words

        label = self.text_vec[word_ind,-1] 
        if self.agent_mode == 'arg':
            if label == 2:
                if action == 1:
                    reward = self.ra[1] * self.reward_base
                else:
                    reward = -self.ra[1] * self.reward_base
            elif label == 4:
                right_flag = True
                if word_ind in self.current_text['obj_inds'][0]:
                    exc_objs = self.current_text['obj_inds'][1]
                else:
                    exc_objs = self.current_text['obj_inds'][0]
                for oi in exc_objs: # exclusive objs
                    if self.state_beta[oi, -1] == 2:
                        right_flag = False
                        break
                if action == 1 and right_flag:
                    reward = self.ra[2] * self.reward_base
                elif action == 2 and not right_flag:
                    reward = self.ra[2] * self.reward_base
                elif action == 2 and word_ind != self.current_text['obj_inds'][1][-1]:
                    reward = self.ra[2] * self.reward_base
                else:
                    reward = -self.ra[2] * self.reward_base
            else: #if label == 1: # non_action 
                if action == 0:
                    reward = self.ra[0] * self.reward_base
                else:
                    reward = -self.ra[0] * self.reward_base

        else: # self.agent_mode == 'act'
            if label == 2: # required action
                if action == 1: # extracted as action
                    reward = self.ra[1] * self.reward_base
                else: # filtered out
                    reward = -self.ra[1] * self.reward_base
            elif label == 3: # optional action
                if action == 1:
                    reward = self.ra[0] * self.reward_base
                else:
                    reward = 0.0
            elif label == 4: # exclusive action
                assert word_ind in self.current_text['act2related']
                exclusive_act_idxs = self.current_text['act2related'][word_ind]
                exclusive_flag = False
                not_biggest_flag = False
                for idx in exclusive_act_idxs:
                    if self.state_beta[idx, -1] == 2: # extracted as action
                        exclusive_flag = True
                    if idx > word_ind:
                        not_biggest_flag = True
                if action == 1 and not exclusive_flag:
                # extract current word and no former exclusive action was extracted
                    reward = self.ra[2] * self.reward_base
                elif action == 0 and exclusive_flag:
                # filtered out current word because one former exclusive action was extracted
                    reward = self.ra[2] * self.reward_base
                elif action == 0 and not_biggest_flag:
                # filtered out current word and at least one exclusive action left 
                    reward = self.ra[2] * self.reward_base
                else:
                    reward = -self.ra[2] * self.reward_base
            else: #if label == 1: # non_action 
                if action == 0:
                    reward = self.ra[0] * self.reward_base
                else:
                    reward = -self.ra[0] * self.reward_base
        
        if self.use_act_rate and reward != 0:
            if t_a_rate <= self.action_rate and reward > 0:
                reward += 5.0 * np.square(t_a_rate) * self.reward_base
            else:
                reward -= 5.0 * np.square(t_a_rate) * self.reward_base
        # all words of current text are tagged, break
        if word_ind + 1 >= len(self.current_text['tokens']):
            self.terminal_flag = True
        
        return reward


    def getState(self):
        '''
        TODO:   Gets current text state
        '''
        return self.state_alpha, self.state_beta


    def isTerminal(self):
        '''
        TODO:   Returns if tag_actions is done
        PS:     if all the words of a text have been tagged, then terminate
        '''
        return self.terminal_flag
