#coding:utf-8
import time
import ipdb
import argparse
import tensorflow as tf

from utils import get_time, plot_results, str2bool
from DQNAgent import Agent
from EADQN import FRLDQN
from Environment import Environment
from ReplayMemory import ReplayMemory
from gensim.models import KeyedVectors


def args_init():
    parser = argparse.ArgumentParser()

    envarg = parser.add_argument_group('Environment')
    envarg.add_argument("--domain",         type=str,       default='cooking',  help="")
    envarg.add_argument("--reward_assign",  type=list,      default=[1, 2, 3],  help='')
    envarg.add_argument("--model_dim",      type=int,       default=50,         help="")
    envarg.add_argument("--tag_dim",        type=int,       default=50,         help="")
    envarg.add_argument("--dis_dim",        type=int,       default=50,         help="")
    envarg.add_argument("--pos_dim",        type=int,       default=50,         help="")
    envarg.add_argument("--autodim",        type=str2bool,  default=True,       help="")
    envarg.add_argument("--num_words",      type=int,       default=500,        help="")
    envarg.add_argument("--reward_base",    type=float,     default=50.0,       help="")
    envarg.add_argument("--object_rate",    type=float,     default=0.04,       help='')
    envarg.add_argument("--action_rate",    type=float,     default=0.05,       help="")
    envarg.add_argument("--use_act_rate",   type=str2bool,  default=True,       help='')
    envarg.add_argument("--use_act_att",    type=str2bool,  default=False,      help='')
    
    memarg = parser.add_argument_group('Replay memory')
    memarg.add_argument("--positive_rate",  type=float,     default=0.9,    help="")
    memarg.add_argument("--priority",       type=str2bool,  default=True,   help="")
    memarg.add_argument("--replay_size",    type=int,       default=50000,  help="")

    netarg = parser.add_argument_group('Deep Q-learning network')
    netarg.add_argument("--multi_channels",     type=int,       default=0,      help="")
    netarg.add_argument("--batch_size",         type=int,       default=32,     help="")
    netarg.add_argument("--num_actions",        type=int,       default=2,      help="")
    netarg.add_argument("--num_filters",        type=int,       default=32,     help="")
    netarg.add_argument("--optimizer",          type=str,       default='adam', help="")
    netarg.add_argument("--learning_rate",      type=float,     default=0.001,  help="")
    netarg.add_argument("--momentum",           type=float,     default=0.8,    help="")
    netarg.add_argument("--epsilon",            type=float,     default=1e-6,   help="")
    netarg.add_argument("--decay_rate",         type=float,     default=0.88,   help="")
    netarg.add_argument("--gamma",              type=float,     default=0.9,    help="")
    netarg.add_argument("--lambda_",            type=float,     default=0.5,    help="")
    netarg.add_argument("--preset_lambda",      type=str2bool,  default=False,  help="")
    netarg.add_argument("--add_train_noise",    type=str2bool,  default=True,   help="")
    netarg.add_argument("--add_predict_noise",  type=str2bool,  default=True,   help="")
    netarg.add_argument("--noise_prob",         type=float,     default=0.5,    help="")
    netarg.add_argument("--stddev",             type=float,     default=1.0,    help="")

    antarg = parser.add_argument_group('Agent')
    antarg.add_argument("--exploration_rate_start",     type=float,     default=1,      help="")
    antarg.add_argument("--exploration_rate_end",       type=float,     default=0.1,    help="")
    antarg.add_argument("--exploration_rate_test",      type=float,     default=0.0,    help="")
    antarg.add_argument("--exploration_decay_steps",    type=int,       default=1000,   help="")
    antarg.add_argument("--train_frequency",            type=int,       default=1,      help="")
    antarg.add_argument("--train_repeat",               type=int,       default=1,      help="")
    antarg.add_argument("--target_steps",               type=int,       default=5,      help="")
    antarg.add_argument("--random_play",                type=str2bool,  default=False,  help="")
    antarg.add_argument("--display_epoch_result",       type=str2bool,  default=True,   help='')
    antarg.add_argument("--filter_act_idx",             type=str2bool,  default=True,   help='')

    mainarg = parser.add_argument_group('Main loop')
    mainarg.add_argument("--gpu_fraction",      type=float,     default=0.22,       help="")
    mainarg.add_argument("--epochs",            type=int,       default=20,         help="")
    mainarg.add_argument("--start_epoch",       type=int,       default=0,          help="")
    mainarg.add_argument("--stop_epoch_gap",    type=int,       default=5,          help="")
    mainarg.add_argument("--load_weights",      type=str2bool,  default=False,      help="")
    mainarg.add_argument("--save_weights",      type=str2bool,  default=False,      help="")
    mainarg.add_argument("--fold_id",           type=int,       default=0,          help="")
    mainarg.add_argument("--start_fold",        type=int,       default=0,          help='')
    mainarg.add_argument("--end_fold",          type=int,       default=5,          help='')
    mainarg.add_argument("--k_fold",            type=int,       default=5,          help="")
    mainarg.add_argument("--agent_mode",        type=str,       default='act',      help='')
    mainarg.add_argument("--predict_net",       type=str,       default='alpha',     help='')
    mainarg.add_argument("--train_mode",        type=str,       default='single_alpha',     help='')
    mainarg.add_argument("--result_dir",        type=str,       default="test",    help="") 
    mainarg.add_argument("--train_episodes",    type=int,       default=50,             help="") 
    
    args = parser.parse_args()
    args.word_dim = args.model_dim
    if args.autodim:
        args.tag_dim = args.pos_dim = args.model_dim
    args.word2vec = KeyedVectors.load_word2vec_format('data/mymodel-new-5-%d'%args.model_dim, binary=True)
    args.data_name = 'data/%s_labeled_text_data.pkl' % args.domain 
    args.k_fold_indices = 'data/indices/%s_eas_%d_fold_indices.pkl' % (args.domain, args.k_fold)
    if args.train_mode in ['alpha', 'beta']:
        args.train_mode = 'single_' + args.train_mode
    args.result_dir = 'results/%s_%s_%s_%s_%s' % (args.domain, args.agent_mode, args.train_mode, args.predict_net, args.result_dir)
    if args.load_weights:
        args.exploration_rate_start = args.exploration_rate_end
    if args.end_fold > args.k_fold:
        args.end_fold = args.k_fold
    if args.agent_mode == 'arg':
        args.num_words = 100
        args.train_episodes = 500
        args.display_epoch_result = False
    return args


def train_single_net(args):
    start = time.time()
    print('Current time is: %s' % get_time())
    print('Starting at train_single_net...')
    fold_result = {'rec': [], 'pre': [], 'f1': [], 'rw': []}

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    
    for fi in range(args.start_fold, args.end_fold):
        fold_start = time.time()
        args.fold_id = fi

        # Initial environment, replay memory, deep_q_net and agent
        if args.fold_id == args.start_fold:
            env_act = Environment(args)
            mem_act = ReplayMemory(args)
        else:
            env_act.get_fold_data()
            mem_act.reset()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            net_act = FRLDQN(sess, args)
            agent = Agent(env_act, mem_act, net_act, args)

            # loop over epochs
            epoch_result = {'rec': [0.0], 'pre': [0.0], 'f1': [0.0], 'rw': [0.0]}
            log_epoch = 0
            net_name = args.predict_net

            with open("%s_fold%d.txt" % (args.result_dir, args.fold_id), 'w') as outfile:
                print('\n Arguments:')
                outfile.write('\n Arguments:\n')
                for k, v in sorted(args.__dict__.items(), key=lambda x:x[0]):
                    print('{}: {}'.format(k, v))
                    outfile.write('{}: {}\n'.format(k, v))
                print('\n')
                outfile.write('\n')

                if args.load_weights:
                    filename = 'weights/%s_%s_k%d_fold%d.h5' % (args.domain, args.agent_mode, args.k_fold, args.fold_id) 
                    net_act.load_weights(filename)
                    print('\n Test the performance of pre-trained %s net ... \n' % net_name)
                    rec, pre, f1, rw = agent.test(args.valid_steps, outfile, predict_net=net_name)
                    epoch_result['f1'].append(f1)
                    epoch_result['rec'].append(rec)
                    epoch_result['pre'].append(pre)
                    epoch_result['rw'].append(rw)


                for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
                    num_test = -1
                    env_act.train_epoch_end_flag = False
                    while not env_act.train_epoch_end_flag:
                        print('\n Training [%s] predicting [%s] \n' % (args.train_mode, net_name))
                        num_test += 1
                        restart_init = False if num_test > 0 else True
                        agent.train(args.train_steps, args.train_episodes, restart_init, predict_net=net_name)
                        
                        print('\n Test the performance of %s net ... \n' % net_name)
                        rec, pre, f1, rw = agent.test(args.valid_steps, outfile, predict_net=net_name)
                        if f1 > max(epoch_result['f1']):
                            if args.save_weights:
                                filename = 'weights/%s_%s_k%d_fold%d.h5' % (args.domain, args.agent_mode, args.k_fold, args.fold_id) 
                                net_act.save_weights(filename, net_name)
                            log_epoch = epoch
                            outfile.write('\n\nBest f1 value: {}\t best epoch: {}\n'.format(f1, log_epoch))
                            print('\n\nBest f1 value: {}\t best epoch: {}\n'.format(f1, log_epoch))

                        epoch_result['f1'].append(f1)
                        epoch_result['rec'].append(rec)
                        epoch_result['pre'].append(pre)
                        epoch_result['rw'].append(rw)

                    # if no improvement after args.stop_epoch_gap, break
                    if epoch - log_epoch >= args.stop_epoch_gap:
                        best_f1 = max(epoch_result['f1'])
                        best_ep = epoch_result['f1'].index(best_f1)
                        outfile.write('\n\nBest f1 value: {}\t Best epoch: {}\n'.format(best_f1, best_ep))
                        print('\n\nBest f1 value: {}\t Best epoch: {}\n'.format(best_f1, best_ep))
                        print('-----Early stopping, no improvement after %d epochs-----\n' % args.stop_epoch_gap)
                        break

                # filename = 'figures/%s_%s_fold%d_all_epochs.pdf'%(args.result_dir, net_name, args.fold_id)
                # plot_results(epoch_result, args.domain, filename)

                outfile.write('\nNetName: {}\n'.format(net_name))
                print('\nNetName: {}'.format(net_name))
                best_ind = epoch_result['f1'].index(max(epoch_result['f1']))
                for k in epoch_result:
                    fold_result[k].append(epoch_result[k][best_ind])
                    outfile.write('{}: {}\n'.format(k, fold_result[k]))
                    print(('{}: {}\n'.format(k, fold_result[k])))
                avg_f1 = sum(fold_result['f1']) / len(fold_result['f1'])
                avg_rw = sum(fold_result['rw']) / len(fold_result['rw'])
                outfile.write('\nAvg f1: {}  Avg reward: {}\n'.format(avg_f1, avg_rw))
                print('\nAvg f1: {}  Avg reward: {}\n'.format(avg_f1, avg_rw))

                fold_end = time.time()
                print('Total time cost of fold %d is: %ds' % (args.fold_id, fold_end - fold_start))
                outfile.write('\nTotal time cost of fold %d is: %ds\n' % (args.fold_id, fold_end - fold_start))
        
        # reset default graph after each epoch, and close the session to release GPU resource
        tf.reset_default_graph()
    end = time.time()
    print('Current time is: %s' % get_time())
    print('Total time cost: %ds\n' % (end - start))



def train_multi_nets(args):
    start = time.time()
    print('Current time is: %s' % get_time())
    print('Starting at train_multi_nets...')
    fold_result = { 'alpha': {'rec': [], 'pre': [], 'f1': [], 'rw': []},
                    'beta':  {'rec': [], 'pre': [], 'f1': [], 'rw': []},
                    'both':  {'rec': [], 'pre': [], 'f1': [], 'rw': []},
                    }

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    
    for fi in range(args.start_fold, args.end_fold):
        fold_start = time.time()
        args.fold_id = fi

        # Initial environment, replay memory, deep_q_net and agent
        if args.fold_id == args.start_fold:
            env_act = Environment(args)
            mem_act = ReplayMemory(args)
        else:
            env_act.get_fold_data()
            mem_act.reset()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            net_act = FRLDQN(sess, args)
            agent = Agent(env_act, mem_act, net_act, args)

            # loop over epochs
            epoch_result = {k: {'rec': [0.0], 'pre': [0.0], 'f1': [0.0], 'rw': [0.0]} for k in fold_result}
            log_epoch = 0
            with open("%s_fold%d.txt" % (args.result_dir, args.fold_id), 'w') as outfile:
                print('\n Arguments:')
                outfile.write('\n Arguments:\n')
                for k, v in sorted(args.__dict__.iteritems(), key=lambda x:x[0]):
                    print('{}: {}'.format(k, v))
                    outfile.write('{}: {}\n'.format(k, v))
                print('\n')
                outfile.write('\n')

                if args.load_weights:
                    filename = 'weights/%s_%s_k%d_fold%d.h5' % (args.domain, args.agent_mode, args.k_fold, args.fold_id) 
                    net_act.load_weights(filename)
                    for net_name in fold_result:
                        print('\n Test the performance of pre-trained %s net ... \n' % net_name)
                        rec, pre, f1, rw = agent.test(args.valid_steps, outfile, predict_net=net_name)
                        epoch_result[net_name]['f1'].append(f1)
                        epoch_result[net_name]['rec'].append(rec)
                        epoch_result[net_name]['pre'].append(pre)
                        epoch_result[net_name]['rw'].append(rw)
                    

                for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
                    num_test = -1
                    env_act.train_epoch_end_flag = False
                    while not env_act.train_epoch_end_flag:
                        print('\n Training [%s] predicting [%s] \n' % (args.train_mode, 'both'))
                        num_test += 1
                        restart_init = False if num_test > 0 else True
                        agent.train(args.train_steps, args.train_episodes, restart_init, predict_net='both')

                        for net_name in ['both']:#, 'alpha', 'beta']:
                            print('\n Test the performance of %s net ... \n' % net_name)
                            rec, pre, f1, rw = agent.test(args.valid_steps, outfile, predict_net=net_name)
                            epoch_result[net_name]['f1'].append(f1)
                            epoch_result[net_name]['rec'].append(rec)
                            epoch_result[net_name]['pre'].append(pre)
                            epoch_result[net_name]['rw'].append(rw)

                            if net_name == 'both' and f1 > max(epoch_result[net_name]['f1']):
                                if args.save_weights:
                                    filename = 'weights/%s_%s_k%d_fold%d.h5' % (args.domain, args.agent_mode, args.k_fold, args.fold_id) 
                                    net_act.save_weights(filename, net_name)
                                log_epoch = epoch
                                outfile.write('\n\nNetName: {}\t Best f1 value: {}\t best epoch: {}\n'.format(net_name, f1, log_epoch))
                                print('\n\nNetName: {}\t Best f1 value: {}\t best epoch: {}\n'.format(net_name, f1, log_epoch))

                    # if no improvement after args.stop_epoch_gap, break
                    if epoch - log_epoch >= args.stop_epoch_gap:
                        for net_name in epoch_result:
                            best_f1 = max(epoch_result[net_name]['f1'])
                            best_ep = epoch_result[net_name]['f1'].index(best_f1)
                            outfile.write('\n\nNetName: {}\t Best f1 value: {}\t Best epoch: {}\n'.format(net_name, best_f1, best_ep))
                            print('\n\nNetName: {}\t Best f1 value: {}\t Best epoch: {}\n'.format(net_name, best_f1, best_ep))
                        print('-----Early stopping, no improvement after %d epochs-----\n' % args.stop_epoch_gap)
                        break

                # for net_name in epoch_result:
                #     filename = 'figures/%s_%s_fold%d_all_epochs.pdf'%(args.result_dir, net_name, args.fold_id)
                #     plot_results(epoch_result[net_name], args.domain, filename)

                for net_name in epoch_result:
                    outfile.write('\nNetName: {}\n'.format(net_name))
                    print('\nNetName: {}'.format(net_name))
                    for k in epoch_result[net_name]:
                        fold_result[net_name][k].append(max(epoch_result[net_name][k]))
                        outfile.write('{}: {}\n'.format(k, fold_result[net_name][k]))
                        print(('{}: {}\n'.format(k, fold_result[net_name][k])))
                    avg_f1 = sum(fold_result[net_name]['f1']) / len(fold_result[net_name]['f1'])
                    avg_rw = sum(fold_result[net_name]['rw']) / len(fold_result[net_name]['rw'])
                    outfile.write('\nAvg f1: {}  Avg reward: {}\n'.format(avg_f1, avg_rw))
                    print('\nAvg f1: {}  Avg reward: {}\n'.format(avg_f1, avg_rw))

                fold_end = time.time()
                print('Total time cost of fold %d is: %ds' % (args.fold_id, fold_end - fold_start))
                outfile.write('\nTotal time cost of fold %d is: %ds\n' % (args.fold_id, fold_end - fold_start))
        
        # reset default graph after each epoch, and close the session to release GPU resource
        tf.reset_default_graph()
    end = time.time()
    print('Current time is: %s' % get_time())
    print('Total time cost: %ds\n' % (end - start))




if __name__ == '__main__':
    args = args_init()
    if args.train_mode in ['full', 'single_alpha', 'single_beta']:
        train_single_net(args)
    else:
        train_multi_nets(args)
