# Reinforcement Learning : Rainbow DQN Framework From Scratch 

## Introduction 

This project follows the one I just completed on Lunar Lander, in which I implemented Rainbow DQN step by step, one concept after another, following several tutorials and other implementations. The goal of this project is to challenge myself to implement the Rainbow architecture relying solely on the original paper, the documentation of the frameworks and libraries used (PyTorch, NumPy, etc.), and my own implementation ideas. No video or text tutorials, no AI.

## I - Architecture of the Project

In my previous project, I faced a limitation that I really wanted to correct in this one: I wasn't able to test my system in different configurations. Indeed, the Rainbow architecture consists of different modules that improve upon the standard DQN algorithm; these modules are independent, yet when combined, they deliver exceptional performance. However, I reached a point where I wanted to test different configurations and compare their performance (module 1 with module 3 vs. module 1 with module 2 with module 3, for example), but my previous architecture didn't allow me to do that. That's why, in this project, I wanted to treat every module as an independent part, allowing me to enable or disable each one and thus test any configuration I want. Therefore, I chose to structure the project as a framework that everyone could download and use on any compatible environment, with any configuration as well. Here's the project structure:


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