#coding:utf-8
import ipdb
import pickle
import numpy as np

class ReplayMemory:
    def __init__(self, args):
        print('Initializing ReplayMemory...')
        self.size = args.replay_size
        self.num_words = args.num_words
        self.word_dim = args.word_dim
        if args.agent_mode == 'arg':
            self.word_dim += args.dis_dim

        self.actions = np.zeros(self.size, dtype=np.uint8)
        self.rewards = np.zeros(self.size, dtype=np.float16)
        self.states_beta = np.zeros([self.size, self.num_words, self.word_dim+1], dtype=np.float16)
        self.states_alpha = np.zeros([self.size, self.num_words, 2], dtype=np.uint8)
        self.terminals = np.zeros(self.size, dtype=np.bool)
        self.priority = args.priority
        self.positive_rate = args.positive_rate
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0


    def reset(self):
        print('Reset the replay memory')
        self.actions *= 0
        self.rewards *= 0.0
        self.states_beta *= 0.0
        self.states_alpha *= 0
        self.terminals *= False
        self.count = 0
        self.current = 0

        
    def add(self, action, reward, state_alpha, state_beta, terminal):
        # NB! state is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.states_alpha[self.current] = state_alpha
        self.states_beta[self.current] = state_beta
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)  
        self.current = (self.current + 1) % self.size



    def getMinibatch(self):
        """
        Memory must include poststate, prestate and history
        Sample random indices or with priority
        """
        pre_states_beta = np.zeros([self.batch_size, self.num_words, self.word_dim+1])
        pre_states_alpha = np.zeros([self.batch_size, self.num_words, 2])
        if self.priority:
            pos_amount =  int(self.positive_rate*self.batch_size) 

        indices = []
        count_pos = 0
        count_neg = 0
        count = 0 
        max_circles = 1000 # max times for choosing positive samples or nagative samples
        while len(indices) < self.batch_size:
            # find random index 
            while True:
                # sample one index (ignore states wraping over) 
                index = np.random.randint(1, self.count - 1)
                # NB! prestate (last state) can be terminal state!
                if self.terminals[index - 1]:
                    continue
                # use prioritized replay trick
                if self.priority:
                    if count < max_circles:
                        # if num_pos is already enough but current idx is also pos sample, continue
                        if (count_pos >= pos_amount) and (self.rewards[index] > 0):
                            count += 1
                            continue
                        # elif num_nag is already enough but current idx is also nag sample, continue
                        elif (count_neg >= self.batch_size - pos_amount) and (self.rewards[index] < 0): 
                            count += 1
                            continue
                    if self.rewards[index] > 0:
                        count_pos += 1
                    else:
                        count_neg += 1
                break
            
            pre_states_beta[len(indices)] = self.states_beta[index - 1]
            pre_states_alpha[len(indices)] = self.states_alpha[index - 1]
            indices.append(index)

        # copy actions, rewards and terminals with direct slicing
        actions = self.actions[indices]  
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]
        post_states_beta = self.states_beta[indices]
        post_states_alpha = self.states_alpha[indices]
        return pre_states_alpha, pre_states_beta, actions, rewards, post_states_alpha, post_states_beta, terminals
