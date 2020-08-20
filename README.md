# dueling-DQN-pytorch
very easy implementation of dueling DQN in pytorch

**(update implementation in tensorflow v1(tf1) & v2(tf2))**

all things are in one file, easy to follow~~

![Architecture](./dueling-DQN.png)

## requirements 

- tensorflow (for tensorboard logging)
- pytorch (>=1.0, 1.0.1 used in my experiment)
- gym
- [ViZDoom](https://github.com/mwydmuch/ViZDoom)


## CartPole-v0

for training dueling DQN in CartPole, just run 

```
python dueling_dqn.py
```

common, no description~

in CartPole-v0 the network will convergence to 200 episode reward very quickly~~
## Visual doom

for training dueling DQN in Visual doom, just run
```
python visual_doom.py
```

for testing dueling DQN in Visual doom, just run
```
python visual_doom_test.py
```

use the [basic](https://github.com/mwydmuch/ViZDoom/blob/master/examples/python/basic.py) environment of Visual doom. The agent could play very experienced after about 1000 times of games~~

also provide the dqn code for comparision.

## reference

[Dueling Network Architectures for Deep Reinforcement Learning (arxiv)](https://arxiv.org/abs/1511.06581)


