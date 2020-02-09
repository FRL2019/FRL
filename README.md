# FRL

### Code for the paper 'Federated Reinforcement Learning'

## Running

### GridWrold

All arguments are preset in main.py, so you can start training the FRL model by:

```bash
$ python main.py
```

If you want a more robust result, try the following command, which will train and test the FRL model for $k$ times and save the average results.

```bash
$ bash both_train.sh
```



For training the baseline model, DQN-alpha, run the following command:

```bash
$ bash alpha_train.sh
```

For DQN-full, run:

```bash
$ bash full_train.sh
```



## Text2Action (EASDRL)

 There are three sub-domains in Text2Action: win2k, wikihow and cooking. For each sub-domain, you need to train both Action-Name Extractor and Action-Argument Extractor.

All parameters are preset in `main.py`, but you need to change some of them according to the sub-domain and the type of extractor.

**Train FRL model**

Take win2k sub-domain as an example. 

- To train and test the Action-Name Extractor, run:

```bash
$ python main.py --domain 'win2k' --agent_mode 'act' --predict_net 'both' --train_mode 'frl_separate' --result_dir 'test_frl_act'
```

- *where `domain` indicates the name of sub-domains, `agent_mode` indicates the type of extractor, `predict_net` and `train_mode` indicate the model name.*
- For Action-Argument Extractor, change the `agent_mode` to `arg` and run:

```bash
$ python main.py --domain 'win2k' --agent_mode 'arg' --predict_net 'both' --train_mode 'frl_separate' --result_dir 'test_frl_arg'
```

You can easily change the parameter `--domain` to `'cooking'` or `'wikihow'` and repeat the procedure to train models for the other two sub-domains.



**Train DQN-alpha model**

Take win2k sub-domain as an example. 

- To train and test the Action-Name Extractor, run:

```bash
$ python main.py --domain 'win2k' --agent_mode 'act' --predict_net 'alpha' --train_mode 'single_alpha' --result_dir 'test_dqn_alpha_act'
```

- For Action-Argument Extractor, change the `agent_mode` to `arg` and run:

```bash
$ python main.py --domain 'win2k' --agent_mode 'arg' --predict_net 'alpha' --train_mode 'single_alpha' --result_dir 'test_dqn_alpha_arg'
```

You can easily change the parameter `--domain` to `'cooking'` or `'wikihow'` and repeat the procedure to train models for the other two sub-domains.



**Train DQN-full model**

Take win2k sub-domain as an example. 

- To train and test the Action-Name Extractor, run:

```bash
$ python main.py --domain 'win2k' --agent_mode 'act' --predict_net 'full' --train_mode 'full' --result_dir 'test_dqn_full_act'
```

- For Action-Argument Extractor, change the `agent_mode` to `arg` and run:

```bash
$ python main.py --domain 'win2k' --agent_mode 'arg' --predict_net 'full' --train_mode 'full' --result_dir 'test_dqn_full_arg'
```

You can easily change the parameter `--domain` to `'cooking'` or `'wikihow'` and repeat the procedure to train models for the other two sub-domains.