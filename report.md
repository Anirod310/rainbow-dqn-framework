```bash
Rainbow Framework
│
├── Configuration
│      │
│      └── YAML
│
├── Factory / Composition
│      │
│      └── builds configured components
│
├── Environment
│      │
│      └── Gymnasium / custom environments
│
├── Agent
│      │
│      ├── Model
│      │    ├── Standard Network
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
                         │
                         ▼
                      Factory
                         │
                         ▼
                       Agent
                    ┌────┼────┐
                    ▼    ▼    ▼
                  Model Replay Learning
                           │      │
                           │      ├── Target Strategy
                           │      ├── Return Strategy
                           │      └── Distributional Strategy
                           │
                           └────────── Model
                         
             ┌─────────────┴─────────────┐
             ▼                           ▼
         Training                    Evaluation
```