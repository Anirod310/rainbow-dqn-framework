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

#### A) Atari

*<u>Implementation and Hyperparameters</u>*

After roughly defining the framework architecture, the next step was to implement the baseline DQN. To do so, I first tried to implement the paper [Human-level control through deep reinforcement learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf), which implements the basic DQN method for all the Atari games. I use the Gymnasium Atari environment and start with only one game: Breakout.

Here's how I structured the baseline training architecture:
- Almost everything is an object, and I use a config.yaml file for as many parameters as I can, in order to control all the experiments from this file only.

- The factory owns all the objects and environment creation.

- The environment is preprocessed with the Gymnasium wrappers, following the paper: each action is repeated 4 times (frame skip) and the observation is the pixel-wise maximum over the last two frames, converted to grayscale and resized to 84x84. A random number of no-op actions (up to 30) is played at the start of each episode, and the 4 most recent frames are stacked to form the input of the network. Observations are stored as uint8 then converted in torch tensors and divided by 255 when they are given to the network. A custom wrapper also presses FIRE at the start of the game and after each life lost, since Breakout needs it to launch the ball.

- The online model computes a Q-value for each action given the observation of the environment.

- The agent follows an epsilon-greedy policy: with probability epsilon it selects a random action, otherwise it selects the action with the highest Q-value. Epsilon is linearly annealed from 1 to 0.1 over the first 1,000,000 emulator frames, and then stays fixed at 0.1.

- The replay buffer stores a tuple containing the observation, the action taken, the reward obtained for this action (clipped to -1, 0 or +1), the next observation of the environment and whether the transition is terminal. A transition is considered terminal when the game is over, but also each time a life is lost, as in the paper. The episode itself continues until the game is over, and the score is not affected by this.

- Learning starts once the replay buffer contains 50,000 transitions. After that, a learning step is performed once every 4 agent steps (i.e. every 16 emulator frames), on a minibatch sampled uniformly from the replay buffer.

- For the learning step, the learner object will use another model (target model) to compute the target value (by using the next observation) for the observations of the batch. The target is not bootstrapped on terminal transitions.

- In the learner, we then use the Huber loss (which is the PyTorch loss that is the closest to the one of the original paper) to compute the error.

- Then we perform backpropagation with loss.backward(), and finally a gradient descent step with optimizer.step(). We use RMSprop for the optimizer, with the configuration described below. The target model is updated with the weights of the online model every 10,000 learning steps.

- At the end of training, we save a .pth file containing the weights state dict, the seed used, the number of training episodes, the total emulator frames, the last average reward over an interval of episodes, the training duration in seconds and finally the average evaluation reward (which is initialized to 0 and then overwritten during evaluation).

For the evaluation baseline, we just load the .pth file, use the saved weights, then play for a number of evaluation episodes in eval mode (torch.no_grad() and model.eval()).

In order to be as faithful to the original paper as possible, I use the hyperparameters listed in the *Extended Data Table 1* of the paper :

| Hyperparameter | Paper (Mnih et al., 2015) | My implementation |
|---|---|---|
| minibatch size | 32 | 32 |
| replay memory size | 1,000,000 | 500,000 / 300,000 (depending on the implementation; further explanations of these choices below) |
| agent history length | 4 | 4 |
| target network update frequency | 10,000 | 10,000 |
| discount factor | 0.99 | 0.99 |
| action repeat | 4 | 4 |
| update frequency | 4 | 4 |
| learning rate | 0.00025 | 0.00025 |
| gradient momentum | 0.95 | 0.95 |
| squared gradient momentum | 0.95 | 0.95 |
| min squared gradient | 0.01 | 0.01 (slight difference from the paper: PyTorch RMSprop eps, added outside the square root) |
| initial exploration | 1 | 1 |
| final exploration | 0.1 | 0.1 |
| final exploration frame | 1,000,000 | 1,000,000 |
| replay start size | 50,000 | 50,000 (counted in transitions, i.e. agent steps) |
| no-op max | 30 | 30 (default of the AtariPreprocessing Gymnasium wrapper) |
|

**Other differences from the paper**

| Aspect | Paper | My implementation |
|---|---|---|
| Loss | Error term clipped to [-1, 1] | Huber loss (equivalent gradient), averaged over the minibatch |
| Optimizer | RMSProp | PyTorch RMSprop with centered=True |
| Reward | Clipped to -1, 0, +1 | Clipped to -1, 0, +1 with np.sign (stored in the replay buffer) |
| Life loss | Marks the end of an episode during training | Same, with a forced FIRE action after each life lost |
| Environment | Arcade Learning Environment, Breakout | ALE/Breakout-v5, repeat_action_probability=0 |


---

*<u>Results</u>*

In order to test the robustness of my implementation, I tried 3 games and compared my results with the paper's results, in conditions as close as possible to those of paper. We tested the implementation on **Breakout**, **Pong** and **Space Invaders**.

**Greedy evaluation of the saved checkpoints** (100 episodes each, mean ± standard deviation of the episode scores, no-op starts up to 30, evaluation seed 42). 


| Game | Run | Training seed | Replay buffer | Frames at checkpoint | Eval ε = 0 | Eval ε = 0.05 |
|---|---|---|---|---|---|---|
| Breakout | 10,000 episodes | 42 | 300,000 | 16.7 M | 263.29 ± 40.93 | 158.41 ± 95.74 |
| Breakout | 15,000 episodes | 42 | 500,000 | 18.1 M | 265.86 ± 34.69 | 197.68 ± 96.23 |
| Breakout | 30,000 episodes (final model) | 42 | 300,000 | 83.3 M | 2.35 ± 2.29 | 2.64 ± 2.77 |
| Pong | 10 M frames | 42 | 300,000 | 10.0 M | 12.36 ± 7.00 | 7.72 ± 7.81 |
| Pong | 15,000 episodes | 42 | 300,000 | 20.0 M | 20.49 ± 0.50 | 16.98 ± 3.28 |
| Space Invaders | 15,000 episodes | 41 | 300,000 | 20.0 M | 676.35 ± 189.16 | 635.25 ± 170.47 |
| Space Invaders | 50 M frames budget (best checkpoint) | 41 | 300,000 | 27.6 M | 481.55 ± 91.94 | 579.15 ± 157.48 |
|

**Reference scores** (Mnih et al., 2015, Extended Data Table 2; DQN after 50 M frames).
|
| Game | Random | Human | DQN (paper) |
|---|---|---|---|
| Breakout | 1.7 | 31.8 | 401.2 ± 26.9 |
| Pong | -20.7 | 9.3 | 18.9 ± 1.3 |
| Space Invaders | 148 | 1652 | 1976 ± 893 |
|

*<u>Result's Interpretations</u>*

**Breakout**  
Breakout was the first game I worked on, and I used it to test each component of the baseline architecture. Regarding the results, I observed that the agent needed more than 5 million frames before it started to learn effectively. Furthermore, after approximately 15–20 million frames, learning appeared to stagnate and, more importantly, performance began to deteriorate. For example, in the 80-million-frame run, the average evaluation reward eventually dropped to a level close to that obtained with random actions.

I have not yet identified the exact cause of this behaviour. My main hypothesis is that the replay buffer capacity may be a contributing factor. Due to the RAM limitations of my setup, I use a buffer that is half the size of the one used in the original paper, as I cannot afford to store one million transitions. This smaller capacity could reduce the diversity of the experiences available for training and cause the agent to forget previously learned behaviours as older transitions are replaced by newer ones.

This limitation could also partly explain the differences between the average rewards obtained in my experiments and those reported in the original paper. However, further investigation is needed to determine whether the replay buffer capacity is actually responsible for these differences.  

---

**Pong**  
The results obtained on Pong were quite different. The agent learned relatively quickly to play the game at a near-optimal level, consistently achieving scores close to the maximum.

By visually observing the agent's behaviour, I noticed that it had learned a very specific strategy: it repeatedly hit the ball with the corner of its paddle while moving the paddle downwards. This produced a trajectory that the opponent's paddle could not reach, allowing the agent to win almost every time.

This was particularly interesting because it showed that the agent could discover an effective strategy through reinforcement learning without being explicitly programmed to use it. It was also satisfying to see the framework learn such an effective policy, demonstrating that the baseline architecture could successfully solve at least some Atari environments.

---

**Space Invaders**  
Finally, Space Invaders highlighted the potential differences between my implementation and the one described in the original paper. Although the agent learned a policy, its performance remained significantly lower than the results reported in the paper, reaching approximately 30% of the reported performance.

Initially, I suspected that this could be due to the lack of sufficiently informative rewards during the early stages of training, combined with the greater complexity of the game compared with the two previous environments. However, when I extended the training duration, I observed a similar deterioration in performance to the one seen in Breakout.

This raised the possibility that some aspects of my implementation or training configuration might be limiting performance. Since such a significant gap could indicate differences in the experimental setup or implementation, I decided to investigate further rather than simply increasing the training duration.

To help identify potential issues and validate the modularity of the framework, I implemented the next development step: extending support to other Gymnasium environments with different configurations, observation spaces, and action spaces. LunarLander was selected to test the framework on a non-Atari environment and verify that the architecture could handle vector observations as well as image-based observations.

#### B) Gymnasium environment: LunarLander

*<u>Implementation and Hyperparameters</u>*

After testing the baseline on Atari games, I wanted to verify that the framework could handle environments with different observation and action spaces. I chose LunarLander-v3, a non-Atari environment where the agent must learn to land a spacecraft safely (I already learned and applied rainbow dqn on this environment, for more details check my [lunar-lander-rainbow-dqn](https://github.com/Anirod310/lunar-lander-rainbow-dqn ) project).

Unlike Atari, the environment provides a vector of 8 numerical values describing the lander's state instead of an image. Therefore, I implemented an MLP backbone to process these observations, while keeping the rest of the baseline architecture modular.

I also adapted the training configuration to this environment, notably the optimizer and exploration schedule. The goal was to establish a strong baseline before implementing the Rainbow components.

| Hyperparameter | Value |
|---|---:|
| Environment | LunarLander-v3 |
| Random seed | 41,42,43,44 |
| Device | CPU/CUDA (for such a small neural network, using cuda just slows the learning and doesn't add real value) |
| Replay buffer capacity | 1,000,000 |
| Mini-batch size | 128 |
| Discount factor (gamma) | 0.99 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Adam epsilon | 0.00015 |
| Target network update frequency | 1,000 steps |
| Learning update frequency | Every 1 agent action |
| Initial exploration rate (epsilon_start) | 1.0 |
| Final exploration rate (epsilon_end) | 0.1 |
| Epsilon annealing duration | 50,000 steps |
| Replay learning start | 1,000 steps |
| Number of training episodes | 5,000 |
| Total frame budget | 500,000 |
| Evaluation episodes | 100 |
| Rendered evaluation episodes | 10 |


*<u>Results</u>*

To test the robustness of the implementation, I evaluated the agent using three different training seeds: 41, 42 and 43. I also ran a second experiment with seed 42 to check the consistency of the results.

| Training seed | Frames at checkpoint | Average training reward | Average evaluation reward |
|---|---:|---:|---:|
| 43 | 500,262 | 154.66 | 185.15 |
| 41 | 279,898 | 223.32 | 169.88 |
| 42 | 309,627 | 236.85 | 246.94 |
| 42 (second run) | 438,809 | 222.24 | 247.43 |

*<u>Results' Interpretations</u>*

The results show that the agent successfully learned to land in LunarLander, with evaluation rewards above 200 in both runs using seed 42. However, the results for seeds 41 and 43 were lower, showing that performance still varies depending on the training run.

Overall, these experiments showed that the framework could successfully learn in a non-Atari environment with vector observations, achieving promising evaluation rewards across different training runs. This suggests that the core DQN implementation is functional, although it does not guarantee that every component is correct. 

The difference between these results and those obtained on Atari may therefore come from the image preprocessing pipeline, Atari-specific environment handling, or differences in training configurations. Further investigation of these components will be necessary to identify the main causes of the performance gap with the original DQN paper.