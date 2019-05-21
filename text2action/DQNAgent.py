#coding:utf-8
import ipdb
import random
import numpy as np


class Agent:
    def __init__(self, environment, replay_memory, deep_q_network, args):
        print('Initializing the Agent...')
        self.args = args
        self.env = environment
        self.mem = replay_memory
        self.net = deep_q_network
        self.agent_mode = args.agent_mode
        self.num_words = args.num_words
        self.batch_size = args.batch_size
        self.num_actions = args.num_actions

        self.exp_rate_start = args.exploration_rate_start
        self.exp_rate_end = args.exploration_rate_end
        self.exp_decay_steps = args.exploration_decay_steps
        self.exploration_rate_test = args.exploration_rate_test
        self.total_train_steps = args.start_epoch * args.train_steps

        self.train_frequency = args.train_frequency
        self.train_repeat = args.train_repeat
        self.target_steps = args.target_steps
        self.random_play = args.random_play
        self.filter_act_idx = args.filter_act_idx 
        self.steps = 0  

    
    def _restart(self, train_flag, init=False):
        self.steps = 0
        self.env.restart(train_flag, init)


    def _explorationRate(self):
        # calculate decaying exploration rate
        if self.total_train_steps < self.exp_decay_steps:
            return self.exp_rate_start - self.total_train_steps * \
            (self.exp_rate_start - self.exp_rate_end) / self.exp_decay_steps
        else:
            return self.exp_rate_end
 

    def step(self, exploration_rate, predict_net, is_test=False):
        # exploration rate determines the probability of random moves
        if random.random() < exploration_rate:
            action = np.random.randint(self.num_actions)
        else:
            # otherwise choose action with highest Q-value
            state_alpha, state_beta = self.env.getState()
            qvalue = self.net.predict(state_alpha, state_beta, predict_net)
            action = np.argmax(qvalue)
            
        # perform the action  
        reward = self.env.act(action, self.steps)
        state_alpha, state_beta = self.env.getState()
        terminal = self.env.isTerminal()
        
        results = []
        self.steps += 1
        if terminal:
            if is_test:
                results = self.compute_f1(self.args.display_epoch_result)
            self.steps = 0
            reward += 2#*self.args.reward_base   #give a bonus to the terminal actions

        return action, reward, state_alpha, state_beta, terminal, results

 
    def train(self, train_steps, train_episodes, restart_init, predict_net):
        '''
        Play given number of steps
        '''
        trained_texts = 0
        ep_loss, ep_rewards = [], []
        # ep_results = {'rec': [], 'pre': [], 'f1': [], 'loss': [], 'rw': []}
        if restart_init:
            self._restart(train_flag=True, init=True)
        #self._restart(train_flag=True, init=restart_init)
        #ipdb.set_trace()
        for i in range(train_steps):
            if self.random_play:
                action, reward, state_alpha, state_beta, terminal, results = self.step(1)
            else:
                action, reward, state_alpha, state_beta, terminal, results = self.step(self._explorationRate(), predict_net)
                self.mem.add(action, reward, state_alpha, state_beta, terminal)

                # Update target network every target_steps steps
                if self.target_steps and i % self.target_steps == 0:
                    self.net.update_target_network()

                # train after every train_frequency steps
                if self.mem.count > self.mem.batch_size and i % self.train_frequency == 0:
                    # train for train_repeat times
                    for j in range(self.train_repeat):
                        # sample minibatch
                        minibatch = self.mem.getMinibatch()
                        # train the network
                        delta, loss = self.net.train(minibatch)
                        ep_loss.append(loss)
            
            # increase number of training steps for epsilon decay
            ep_rewards.append(reward)
            self.total_train_steps += 1
            if terminal:
                trained_texts += 1
                if len(ep_loss) > 0:
                    avg_loss = sum(ep_loss) / len(ep_loss)
                    max_loss = max(ep_loss)
                    min_loss = min(ep_loss)
                    print('max_loss: {:>6.6f}\t min_loss: {:>6.6f}\t avg_loss: {:>6.6f}'.format(max_loss, min_loss, avg_loss))
                    # ep_results['loss'].append(avg_loss)
                    
                # ep_results['rec'].append(results[-3])
                # ep_results['pre'].append(results[-2])
                # ep_results['f1'].append(results[-1])
                # ep_results['rw'].append(sum(ep_rewards)/len(ep_rewards))
                ep_loss, ep_rewards = [], []
                self._restart(train_flag=True)
            
                if self.env.train_epoch_end_flag or trained_texts >= train_episodes:
                    break

        # return ep_results
    

    def test(self, test_steps, outfile, predict_net):
        '''
        Play given number of steps
        '''
        t_total_rqs = t_tagged_rqs = t_right_rqs = 0
        t_total_ops = t_tagged_ops = t_right_ops = 0
        t_total_ecs = t_tagged_ecs = t_right_ecs = 0
        t_right_tag = t_right_acts = t_tagged_acts = t_total_acts = t_words = 0
        t_acc = t_rec = t_pre = t_f1 = 0.0
        
        cumulative_reward = 0
        self._restart(train_flag=False, init=True)
        for test_step in range(test_steps):
            if self.random_play:
                a, r, s_a, s_b, t, rs = self.step(1)
            else:
                a, r, s_a, s_b, t, rs = self.step(self.exploration_rate_test, predict_net, True)
            cumulative_reward += r
            if t:
                t_words += rs[0]
                t_total_rqs += rs[1]
                t_right_rqs += rs[2]
                t_tagged_rqs += rs[3]
                t_total_ops += rs[4]
                t_right_ops += rs[5]
                t_tagged_ops += rs[6]
                t_total_ecs += rs[7]
                t_right_ecs += rs[8]
                t_tagged_ecs += rs[9]
                t_total_acts += rs[10] 
                t_right_acts += rs[11]
                t_tagged_acts += rs[12]
                t_right_tag += rs[13]   
                self._restart(train_flag=False) 
                
            if self.env.valid_epoch_end_flag:
                break   

        average_reward = cumulative_reward / (test_step + 1)
        t_acc = float(t_right_tag) / t_words
        results = {'rec': [], 'pre': [], 'f1': []}
        self.basic_f1(t_total_rqs, t_right_rqs, t_tagged_rqs, results)
        self.basic_f1(t_total_ops, t_right_ops, t_tagged_ops, results)
        self.basic_f1(t_total_ecs, t_right_ecs, t_tagged_ecs, results)
        self.basic_f1(t_total_acts, t_right_acts, t_tagged_acts, results)
        t_rec = results['rec'][-1]
        t_pre = results['pre'][-1]
        t_f1 = results['f1'][-1]

        outfile.write('\n\npredict_net={} summary:\n'.format(predict_net))
        outfile.write('total_rqs: %d\t right_rqs: %d\t tagged_rqs: %d\n' % (t_total_rqs, t_right_rqs, t_tagged_rqs))
        outfile.write('total_ops: %d\t right_ops: %d\t tagged_ops: %d\n' % (t_total_ops, t_right_ops, t_tagged_ops))
        outfile.write('total_ecs: %d\t right_ecs: %d\t tagged_ecs: %d\n' % (t_total_ecs, t_right_ecs, t_tagged_ecs))
        outfile.write('total_act: %d\t right_act: %d\t tagged_act: %d\n' % (t_total_acts, t_right_acts, t_tagged_acts))  
        outfile.write('acc: %f\t rec: %f\t pre: %f\t f1: %f\n' % (t_acc, t_rec, t_pre, t_f1))
        for k, v in results.iteritems():
            outfile.write('{}: {}\n'.format(k, v))
            print(k, v)
        outfile.write('\ncumulative reward: %f\t average reward: %f\n' % (cumulative_reward, average_reward))
        print('\n\npredict_net={} summary:\n'.format(predict_net))
        print('total_rqs: %d\t right_rqs: %d\t tagged_rqs: %d' % (t_total_rqs, t_right_rqs, t_tagged_rqs))
        print('total_ops: %d\t right_ops: %d\t tagged_ops: %d' % (t_total_ops, t_right_ops, t_tagged_ops))
        print('total_ecs: %d\t right_ecs: %d\t tagged_ecs: %d' % (t_total_ecs, t_right_ecs, t_tagged_ecs))
        print('total_act: %d\t right_act: %d\t tagged_act: %d' % (t_total_acts, t_right_acts, t_tagged_acts))  
        print('acc: %f\t rec: %f\t pre: %f\t f1: %f' % (t_acc, t_rec, t_pre, t_f1))
        print('\ncumulative reward: %f\t average reward: %f\n' % (cumulative_reward, average_reward))
        return t_rec, t_pre, t_f1, average_reward


    def compute_f1(self, display):
        """
        Compute f1 score for current text
        """
        text_vec_tags = self.env.text_vec[:,-1]
        state_tags = self.env.state_beta[:,-1]
        if self.agent_mode == 'arg' and self.filter_act_idx: # act_idxs are not obj_idxs
            state_tags[self.env.current_text['act_inds']] = 1
        
        total_words = self.num_words
        temp_words = len(self.env.current_text['tokens'])
        if temp_words > total_words:
            temp_words = total_words

        record_ecs_act_idxs = []
        right_tag = right_acts = tagged_acts = total_acts = 0
        total_rqs = right_rqs = tagged_rqs = 0
        total_ecs = right_ecs = tagged_ecs = 0
        total_ops = right_ops = tagged_ops = 0
        for s in range(temp_words):
            if state_tags[s] == 2:
                tagged_acts += 1
            if text_vec_tags[s] == 2: # required actions
                total_acts += 1
                total_rqs += 1
                if state_tags[s] == 2: # extract
                    tagged_rqs += 1
                    right_rqs += 1
                    right_acts += 1
                    right_tag += 1
            elif text_vec_tags[s] == 3: # optional actions
                #total_acts += 1
                #total_ops += 1
                if state_tags[s] == 2: # extract
                    total_acts += 1
                    total_ops += 1
                    tagged_ops += 1
                    right_ops += 1
                    right_acts += 1
                    right_tag += 1
            elif text_vec_tags[s] == 4: # exclusive actions
                if state_tags[s] == 2:
                    tagged_ecs += 1
                if s not in record_ecs_act_idxs:
                    total_acts += 1
                    total_ecs += 1
                    record_ecs_act_idxs.append(s)
                if self.agent_mode == 'arg':
                    right_flag = True
                    if s in self.env.current_text['obj_inds'][0]:
                        exc_objs = self.env.current_text['obj_inds'][1]
                    else:
                        exc_objs = self.env.current_text['obj_inds'][0]
                    record_ecs_act_idxs.extend(exc_objs)
                    for oi in exc_objs:
                        if state_tags[oi] == 2:
                            right_flag = False
                            break
                    if state_tags[s] == 2 and right_flag:
                        right_ecs += 1
                        right_acts += 1
                        right_tag += 1
                    elif state_tags[s] == 1 and not right_flag:
                        right_tag += 1
                else:
                    assert s in self.env.current_text['act2related']
                    exclusive_act_idxs = self.env.current_text['act2related'][s]
                    record_ecs_act_idxs.extend(exclusive_act_idxs)
                    exclusive_flag = False
                    for idx in exclusive_act_idxs:
                        if state_tags[idx] == 2: # extracted as action
                            exclusive_flag = True
                            break
                    if not exclusive_flag and state_tags[s] == 2: # extract
                        right_ecs += 1
                        right_acts += 1
                        right_tag += 1
                    elif exclusive_flag and state_tags[s] == 1: # filtered out
                        right_tag += 1
            elif text_vec_tags[s] == 1: # non_actions
                if state_tags[s] == 1:
                    right_tag += 1
                else:
                    tagged_rqs += 1

        acc = float(right_tag)/temp_words
        results = {'rec': [], 'pre': [], 'f1': []}
        self.basic_f1(total_rqs, right_rqs, tagged_rqs, results)
        self.basic_f1(total_ops, right_ops, tagged_ops, results)
        self.basic_f1(total_ecs, right_ecs, tagged_ecs, results)
        self.basic_f1(total_acts, right_acts, tagged_acts, results)
        rec = results['rec'][-1]
        pre = results['pre'][-1]
        f1 = results['f1'][-1]
        if display:
            print('rec: {:>13.6f}\t pre: {:>13.6f}\t f1: {:>14.6f}'.format(rec, pre, f1))

        return temp_words, total_rqs, right_rqs, tagged_rqs, total_ops, right_ops, tagged_ops, total_ecs, \
        right_ecs, tagged_ecs, total_acts, right_acts, tagged_acts, right_tag, acc, rec, pre, f1


    def basic_f1(self, total, right, tagged, results):
        rec = pre = f1 = 0.0
        if total > 0:
            rec = right / float(total)
        if tagged > 0:
            pre = right / float(tagged)
        if rec + pre > 0:
            f1 = 2 * pre * rec / (pre + rec)
        results['rec'].append(rec)
        results['pre'].append(pre)
        results['f1'].append(f1)