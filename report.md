# Reinforcement Learning : Rainbow DQN Framework From Scratch 

## Introduction 

This project follows the one I just completed on Lunar Lander, in which I implemented Rainbow DQN step by step, one concept after another, following several tutorials and other implementations.

The goal of this project is to challenge myself to implement the Rainbow architecture relying solely on the original paper, the documentation of the frameworks and libraries used (PyTorch, NumPy, etc.), and my own implementation ideas. No video or text tutorials, no AI.

## I - Architecture of the Project

In my previous project, I faced a limitation that I really wanted to correct in this one: I wasn't able to test my system in different configurations. 

Indeed, the Rainbow architecture consists of different modules that improve upon the standard DQN algorithm; these modules are independent, yet when combined, they deliver exceptional performance. However, I reached a point where I wanted to test different configurations and compare their performance (module 1 with module 3 vs. module 1 with module 2 with module 3, for example), but my previous architecture didn't allow me to do that.  

That's why, in this project, I wanted to treat every module as an independent part, allowing me to enable or disable each one and thus test any configuration I want. 

Therefore, I chose to structure the project as a framework that everyone could download and use on any compatible environment(gymnasium env with discrete action possibilities), with any configuration as well. Here's the project structure:

```bash
Rainbow Framework
│
├── Configuration
│      │
│      └── YAML
│
├── Factory / Composition
│      │
│      ├── builds configured components
│      └── preprocessor
│             ├── enabled -> yes/no
│             └── implementation -> preprocessing strategy
│
├── Agent
│      │
│      ├── Model
│      │    ├── Standard Network
│      │    ├── CNN Network
│      │    ├── Dueling Network
│      │    ├── Noisy Networks
│      │    └── Distributional / C51
│      │
│      ├── Replay Buffer
│      │    ├── Uniform Replay
│      │    └── Prioritized Replay (PER)
│      │
│      └── Learning
│           ├── Target Strategy
│           │    ├── DQN
│           │    └── DDQN
│           │
│           ├── Return Strategy
│           │    ├── 1-step
│           │    └── n-step
│           │
│           └── Distributional Strategy
│                ├── Expected Value
│                └── C51
│
├── Training
│      └── orchestrates the training process
│
└── Evaluation
       └── evaluates a trained Agent
``` 
```bash
Configuration
      ↓
Composition Validator
      ↓
Factory
      ↓
Environment ───┐
Preprocessor ──┼──→ Agent
Model ─────────┤
Replay Buffer ─┤
Learning ──────┘
      ↓
Training / Evaluation
```

## II - The Journey 

This section is mainly dedicated to documenting all the stages I went through during the project, as well as explaining the concepts learned and used during the implementation process.

### 1 - DQN Baseline

After roughly defining the framework architecture, the next step was to Implement the baseline DQN. To do so, I first try to implement the paper [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf), which Implements basic DQN methods to all the Atari games. I use Gymnasium Atari enviromnent, and start with only one game : Breakout.  

Here's how I structured the baseline training architecture : 
- Almost everything is objects.
- The factory owns all the objects and environment creation 
- The online model computes a q-value for each action regarding the observation of the environment
- The agent selects the action with the highest value 
- The replay buffer stores a tuple containing the observation, the action taken, the reward obtained for this action, the next observation of the environment and if the game is terminated, then we perform a learning step
- For the learning step, the learner object will use another model (target model) to compute the target value(by using the next observation) for the observation of the batch.
- In the learner, we then use the Huber loss(which is the pytorch loss that is the closer to the one of the original paper ) to compute the error
- Then we perform a backpropagation with loss.backward(), and finally a gradient descent step with optimizer.step(). We use RMS prop for the optimizer, with the config described below. 
- At the end of training, we save a pth file containing the weights state dict, the seed used, the number of training episodes, the total emulator frames, the last average reward over an interval of episodes, the training duration in seconds and finally the average evaluation reward(which is defined at 0 and then overwritten during evaluation) 

For the evaluation baseline, we just load the pth file, use the weights saved, then plays for a number of eval episodes in eval mod (torch.no_grad() and model.eval()).

Now let's dive in the creation of every parts of all this workflow.


