import ipdb
import time
import numpy as np
import tensorflow as tf


class FRLDQN(object):
    """docstring for FRLNetwork"""
    def __init__(self, sess, args):
        self.sess = sess
        self.height = args.screen_height
        self.width = args.screen_width
        self.channels = args.history_length
        self.num_actions = args.num_actions
        self.gamma = 0.9
        self.build_dqn()


    def conv2d(self, x, output_dim, kernel_size, stride, initializer, activation_fn=tf.nn.relu, padding='VALID', name='conv2d'):
        with tf.variable_scope(name):
            # data_format = 'NHWC'
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]
            
            w = tf.get_variable('w', kernel_size, tf.float32, initializer=initializer)
            conv = tf.nn.conv2d(x, w, stride, padding)

            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(conv, b)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b


    def max_pooling(self, x, kernel_size, stride, padding='VALID', name='max_pool'):
        with tf.variable_scope(name):
            stride = [1, stride[0], stride[1], 1]
            kernel_size = [1, kernel_size[0], kernel_size[1], 1]
            return tf.nn.max_pool(x, kernel_size, stride, padding)


    def linear(self, x, output_dim, activation_fn=None, name='linear'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [x.get_shape()[1], output_dim], tf.float32, 
                initializer=tf.truncated_normal_initializer(0, 0.1))
            b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.1))
            out = tf.nn.bias_add(tf.matmul(x, w), b)

        if activation_fn != None:
            out = activation_fn(out)
        return out, w, b


    def build_dqn(self):
        self.beta_w = {}
        self.beta_t_w = {}
        self.alpha_w = {}
        self.alpha_t_w = {}
        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        #initializer = tf.truncated_normal_initializer(0, 0.02)

        height, width, channels, num_actions = self.height, self.width, self.channels, self.num_actions
        self.s = tf.placeholder(tf.float32, [None, height, width, channels])
        #ipdb.set_trace()

        def build_nn(name, weight, s):
            with tf.variable_scope(name):
                layers = []
                print('Initializing %s network ...' % name)
                conv1, weight['conv1_w'], weight['conv1_b'] = self.conv2d(s, 16, [8, 8], [4, 4], initializer, name='conv1')
                conv2, weight['conv2_w'], weight['conv2_b'] = self.conv2d(conv1, 32, [4, 4], [2, 2], initializer, name='conv2')
                total_dims = reduce(lambda x, y: x * y, conv2.get_shape().as_list()[1:])
                conv2_flat = tf.reshape(conv2, [-1, total_dims])
                dense, weight['dense_w'], weight['dense_b'] = self.linear(conv2_flat, 256, tf.nn.relu, name='dense')

                q_s_t, weight['q_s_t_w'], weight['q_s_t_b'] = self.linear(dense, num_actions, name='output')
                layers = [conv1, conv2, conv2_flat, dense, q_s_t]
                for layer in layers:
                    print(layer.get_shape())

                return q_s_t


        self.beta_q = build_nn('beta_q', self.beta_w, self.s)
        self.beta_t_q = build_nn('beta_t_q', self.beta_t_w, self.s)
        self.alpha_postq = build_nn('alpha_q_s_t_plus_1', self.alpha_w, self.s)
        self.alpha_t_q = build_nn('alpha_t_q', self.alpha_t_w, self.s)

        def update_q_network_op(t_w, name):
            with tf.variable_scope(name):
                print('Initializing %s ...' % name)
                t_w_input = {}
                t_w_assign_op = {}

                for name in t_w:
                    t_w_input[name] = tf.placeholder(tf.float32, t_w[name].get_shape().as_list(), name)
                    t_w_assign_op[name] = t_w[name].assign(t_w_input[name])

                return t_w_input, t_w_assign_op


        self.beta_t_w_input, self.beta_t_w_assign_op = update_q_network_op(self.beta_t_w, 'beta_update_q_network_op')
        self.alpha_t_w_input, self.alpha_t_w_assign_op = update_q_network_op(self.alpha_t_w, 'alpha_update_q_network_op')

            
        with tf.variable_scope('beta_optimizer'):
            print('Initializing beta optimizer ...')
            self.target_q = tf.placeholder(tf.float32, [None, num_actions])
            self.delta_beta = self.target_q - self.beta_q  # for updating Q_beta, target_q is a constant

            self.loss_beta = tf.reduce_sum(tf.square(self.delta_beta), name='loss_beta')
            self.train_beta = tf.train.AdamOptimizer(0.001).minimize(self.loss_beta)


        with tf.variable_scope("alpha_optimizer"):
            print('Initializing alpha optimizer ...')
            self.factor = tf.placeholder(tf.float32, [None]) # = not_terminal * gamma * 1/2
            self.r_t = tf.placeholder(tf.float32, [None])  # for updating Q_alpha, beta_preq and beta_postq are constants
            self.beta_preq = tf.placeholder(tf.float32, [None, num_actions])
            self.beta_postq = tf.placeholder(tf.float32, [None, num_actions])
            self.alpha_beta_q = self.alpha_postq + self.beta_postq
            
            self.y_t = self.r_t + self.factor * tf.reduce_max(self.alpha_beta_q, axis=1)
            self.max_indices = tf.cast(tf.arg_max(self.alpha_beta_q, 1), tf.int32)
            tmp_max_ind = tf.reshape(self.max_indices, [-1, 1])
            tmp_ind = tf.reshape(tf.range(tf.shape(self.max_indices)[0]), [-1, 1])
            self.indices = tf.concat([tmp_ind, tmp_max_ind], axis=1)
            self.delta_alpha = self.y_t - tf.gather_nd(self.beta_preq, self.indices)

            self.loss_alpha = tf.reduce_sum(tf.square(self.delta_alpha), name='loss_alpha')
            self.train_alpha = tf.train.AdamOptimizer(0.001).minimize(self.loss_alpha)

        tf.global_variables_initializer().run()


    def learn_beta(self, prestates, actions, rewards, poststates, terminals):
        #ipdb.set_trace()
        beta_postq = self.beta_t_q.eval({self.s: poststates})
        alpha_postq = self.alpha_t_q.eval({self.s: poststates})
        alpha_beta_q = 0.5 * (beta_postq + alpha_postq)
        max_postq = np.max(alpha_beta_q, axis=1)
        preq = self.beta_q.eval({self.s: prestates})
        targets = preq.copy()

        for i, action in enumerate(actions):
            if terminals[i]:
                targets[i, action] = rewards[i]
            else:
                targets[i, action] = rewards[i] + self.gamma * max_postq[i]

        _, beta_q, delta_beta, loss_beta = self.sess.run([  self.train_beta, 
                                                            self.beta_q, 
                                                            self.delta_beta, 
                                                            self.loss_beta
                                                         ],
                                                         {  self.s: prestates, 
                                                            self.target_q: targets
                                                         })
        return beta_postq, beta_q, delta_beta, loss_beta

 
    def learn_alpha(self, beta_postq, beta_preq, rewards, terminals, alpha_states):
        factors = (1 - terminals) * self.gamma * 0.5 # (1/2)(Q_a + Q_b)
        _, alpha_q, delta_alpha, loss_alpha, indices, alpha_beta_q = self.sess.run([
                                                                        self.train_alpha, 
                                                                        self.alpha_t_q, 
                                                                        self.delta_alpha, 
                                                                        self.loss_alpha, 
                                                                        self.indices, 
                                                                        self.alpha_beta_q
                                                                       ],
                                                                       {
                                                                        self.factor: factors, 
                                                                        self.r_t: rewards, 
                                                                        self.beta_preq: beta_preq,
                                                                        self.beta_postq: beta_postq,
                                                                        self.s: alpha_states, 
                                                                       })
        return alpha_q, delta_alpha, loss_alpha, indices, alpha_beta_q



    def update_target_network(self):
        for name in self.beta_w:
            self.beta_t_w_assign_op[name].eval({self.beta_t_w_input[name]: self.beta_w[name].eval()})

        for name in self.alpha_w:
            self.alpha_t_w_assign_op[name].eval({self.alpha_t_w_input[name]: self.alpha_w[name].eval()})



    def train(self, minibatch):
        prestates, actions, rewards, poststates, terminals = minibatch
        prestates = np.transpose(prestates, (0,2,3,1))
        poststates = np.transpose(poststates, (0,2,3,1))
        
        #ipdb.set_trace()
        beta_postq, beta_q, delta_beta, loss_beta = self.learn_beta(prestates, actions, rewards, poststates, terminals)
        alpha_q, delta_alpha, loss_alpha, indices, alpha_beta_q = self.learn_alpha(beta_postq, beta_q, rewards, terminals, prestates)

        return loss_beta, loss_alpha, beta_q.mean(), alpha_q.mean()


    def predict(self, states):
        states = np.reshape(states, [1, self.height, self.width, self.channels])
        qvalues = self.beta_q.eval({self.s: states})

        return qvalues



class FRLActor(object):
    """docstring for FRLActor"""
    def __init__(self, sess, args):
        self.sess = sess
        height = args.screen_height
        width = args.screen_width
        channels = args.history_length
        self.num_actions = args.num_actions
        entropy_beta = 0.01     # entropy regurarlization constant

        self.s = tf.placeholder(tf.float32, [None, height, width, channels], 'state')
        self.a = tf.placeholder(tf.float32, [None, self.num_actions], 'action') # one-hot
        self.td = tf.placeholder(tf.float32, [None], 'td_error')

        with tf.variable_scope('Actor'):
            conv1 = tf.layers.conv2d(
                    inputs=self.s,
                    filters=16,
                    kernel_size=(8, 8),
                    strides=(4, 4),
                    # padding='valid',
                    # data_format='channels_last',
                    # dilation_rate=(1, 1),
                    activation=tf.nn.relu,
                    # use_bias=True,
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    bias_initializer=tf.constant_initializer(0.1),
                    # kernel_regularizer=None,
                    # bias_regularizer=None,
                    # activity_regularizer=None,
                    # trainable=True,
                    name='conv1',
                    # reuse=None
            )

            conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=32,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='conv2',
            )

            total_dims = reduce(lambda x, y: x*y, conv2.get_shape().as_list()[1:])
            conv2_flat = tf.reshape(conv2, [-1, total_dims])

            dense = tf.layers.dense(
                    inputs=conv2_flat,
                    units=256,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='dense'
            )

            self.pi = tf.layers.dense(
                    inputs=dense,
                    units=self.num_actions,
                    activation=tf.nn.softmax,
                    kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='policy'
            )

        with tf.variable_scope('actor_optimize'):
            #ipdb.set_trace()
            # avoid NaN with clipping when value in pi becomes zero
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))

            # policy entropy
            self.entropy = -tf.reduce_sum(self.pi * log_pi, axis=1)
      
            # policy loss (output)  (Adding minus, because the original paper's objective function is for gradient ascent, but we use gradient descent optimizer.)
            self.policy_loss = - tf.reduce_sum( tf.reduce_sum( tf.multiply( log_pi, self.a ), axis=1 ) * self.td + self.entropy * entropy_beta )

            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.policy_loss)

            layers = [conv1, conv2, conv2_flat, dense, self.pi, log_pi]
            for layer in layers:
                print(layer.shape)


    def learn(self, states, actions, td_errors):
        _, entropy, policy_loss = self.sess.run([self.train_op, self.entropy, self.policy_loss],
                        feed_dict={self.s: states, self.a: actions, self.td: td_errors})
        return entropy, policy_loss


    def choose_action(self, state):
        state = state[np.newaxis, :]
        pi = self.sess.run(self.pi, {self.s: state})   # ravel: Return a flattened array.
        #ipdb.set_trace()
        return np.random.choice(np.arange(self.num_actions), p=pi.ravel()) # return an int


class FRLCritic(object):
    """docstring for FRLCritic"""
    def __init__(self, sess, args):
        self.sess = sess
        height = args.screen_height
        width = args.screen_width
        channels = args.history_length
        gamma = 0.9     # discount rate

        self.s = tf.placeholder(tf.float32, [None, height, width, channels], 'state')
        self.v_ = tf.placeholder(tf.float32, [None], 'v_next')
        self.r = tf.placeholder(tf.float32, [None], 'td_error')

        with tf.variable_scope('Critic'):
            conv1 = tf.layers.conv2d(
                    inputs=self.s,
                    filters=16,
                    kernel_size=(8, 8),
                    strides=(4, 4),
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='conv1',
            )

            conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=32,
                    kernel_size=(4, 4),
                    strides=(2, 2),
                    activation=tf.nn.relu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='conv2',
            )

            total_dims = reduce(lambda x, y: x*y, conv2.get_shape().as_list()[1:])
            conv2_flat = tf.reshape(conv2, [-1, total_dims])

            dense = tf.layers.dense(
                    inputs=conv2_flat,
                    units=256,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='dense'
            )

            self.v = tf.layers.dense(
                    inputs=dense,
                    units=1, # single linear output unit for each action representing the action-value
                    activation=tf.nn.softmax,
                    kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='value'
            )

        with tf.variable_scope('critic_optimize'):
            #ipdb.set_trace()
            self.v = tf.squeeze(self.v)
            self.td = self.r + gamma * self.v_ - self.v
            self.value_loss = tf.reduce_sum(tf.square(self.td))  # TD_error = (r+gamma*V_next) - V_eval

            self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.value_loss)

            layers = [conv1, conv2, conv2_flat, dense]
            for layer in layers:
                print(layer.shape)



    def learn(self, prestates, rewards, poststates):
        v_ = self.sess.run(self.v, {self.s: poststates})
        _, td, value_loss = self.sess.run([self.train_op, self.td, self.value_loss], 
                feed_dict={self.s: prestates, self.v_: v_, self.r: rewards})
        return td, value_loss



def FRLAC(sess, args):
    from Agent import ACAgent
    from History3D import ReplayMemory
    from ALEEnvironment import ALEEnvironment

    #ipdb.set_trace()
    scale = 10000
    env = ALEEnvironment(args)
    args.num_actions = env.numActions()
    mem = ReplayMemory(args)
    
    actor = FRLActor(sess, args)
    critic = FRLCritic(sess, args)
    sess.run(tf.global_variables_initializer())

    agent = ACAgent(env, mem, actor, critic, args)
    with open('results/AC_result_%s.txt' % get_time(), 'w') as logger:
        logger.write('\n Parameters: \n')
        for k,v in args.__dict__.items():
            logger.write('{}: {}\n'.format(k, v))
        logger.write('\n Training process: \n')
        agent.train(500*scale, logger)


def get_time():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result
    return timed


@timeit
def main(sess, args):
    from Agent import DQNAgent
    from History3D import ReplayMemory
    from ALEEnvironment import ALEEnvironment

    #ipdb.set_trace()
    scale = 10000
    env = ALEEnvironment(args)
    args.num_actions = env.numActions()
    mem = ReplayMemory(args)
    net = FRLDQN(sess, args)
    agent = DQNAgent(env, mem, net, args)

    with open('results/DQN_result_%s.txt' % get_time(), 'w') as logger:
        logger.write('\n Parameters: \n')
        for k,v in args.__dict__.items():
            logger.write('{}: {}\n'.format(k, v))
        logger.write('\n Training process: \n')
        agent.train(5000*scale, logger)





if __name__ == '__main__':
    import argparse
    from ALEEnvironment import str2bool
    roms = ['breakout.bin','pong.bin','seaquest.bin','space_invaders.bin','montezuma_revenge.bin',]
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_length",     type=int,       default=4,          help="How many frames to be recorded per timestep.")
    parser.add_argument("--mode",               type=str,       default='train',    help="Record game sound in this file.")
    parser.add_argument("--rom_id",             type=int,       default=0,         help="ROM bin file or env id such as Breakout-v0 if training with Open AI Gym.")
    parser.add_argument("--display_screen",     type=str2bool,  default=False,      help="Display game screen during training and testing.")
    parser.add_argument("--sound",              type=str2bool,  default=False,      help="Play (or record) sound.")
    parser.add_argument("--frame_skip",         type=int,       default=1,          help="How many times to repeat each chosen action.")
    parser.add_argument("--minimal_action_set", type=str2bool,  default=True,       help="Use minimal action set.")
    parser.add_argument("--color_averaging",    type=str2bool,  default=True,       help="Perform color averaging with previous frame.")
    parser.add_argument("--screen_width",       type=int,       default=84,         help="Screen width after resize.")
    parser.add_argument("--screen_height",      type=int,       default=84,         help="Screen height after resize.")
    parser.add_argument("--record_screen_path", type=str,       default='',         help="Record game screens under this path. Subfolder for each game is created.")
    parser.add_argument("--record_sound_file",  type=str,       default='',         help="Record game sound in this file.")
    parser.add_argument("--random_seed",        type=int,       default=12345,      help="Random seed for repeatable experiments.")
    parser.add_argument("--cnn_format",         type=str,       default="NHWC",     help="Tensorflow CNN datafromat")
    parser.add_argument("--replay_size",        type=int,       default=100000,     help="")
    parser.add_argument("--explore_rate_start", type=float,     default=1,          help="Exploration rate at the beginning of decay.")
    parser.add_argument("--explore_rate_end",   type=float,     default=0.1,        help="Exploration rate at the end of decay.")
    parser.add_argument("--explore_decay_steps", type=int,      default=1000,       help="How many steps to decay the exploration rate.")
    parser.add_argument("--explore_rate_test",  type=float,     default=0.05,       help="Exploration rate used during testing.")
    parser.add_argument("--train_repeat",       type=int,       default=1,          help="Number of times to sample minibatch during training.")
    parser.add_argument("--target_steps",       type=int,       default=512,          help="Copy main network to target network after this many game steps.")
    parser.add_argument("--batch_size",         type=int,       default=32,         help="Copy main network to target network after this many game steps.")
    parser.add_argument("--gpu_fraction",       type=float,     default=0.3,          help="Copy main network to target network after this many game steps.")
    parser.add_argument("--model_type",         type=str,       default='DQN',          help="Copy main network to target network after this many game steps.")

    args = parser.parse_args()
    args.rom_file = 'roms/%s' % roms[args.rom_id]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if args.model_type == 'DQN':
            main(sess, args)
        else:
            FRLAC(sess, args)

