# How-and-When-Inferring-Passenger-Destination-Based-on-Dynamic-Prices-from-an-Attacker-s-Perspective
《How and When: Inferring Passenger Destination Based on Dynamic Prices from an Attacker’s Perspective》IEEE TRANSACTIONS ON MOBILE COMPUTING 2025
How and When: Inferring Passenger Destination Based on Dynamic Prices from an Attacker's Perspective

https://ieeexplore.ieee.org/document/XXXXXXX
https://www.python.org/
https://www.tensorflow.org/
LICENSE

This repository contains the official implementation for the paper "How and When: Inferring Passenger Destination Based on Dynamic Prices from an Attacker's Perspective" published in IEEE Transactions on Mobile Computing, 2025.

📖 Abstract

Ride-on-Demand (RoD) services like Uber and Didi have revolutionized urban transportation through dynamic pricing mechanisms. While improving efficiency, these pricing strategies introduce significant privacy concerns. This work investigates how attackers can leverage dynamic pricing information to infer passenger destinations more accurately and determines the optimal timing for such privacy attacks.

We propose a comprehensive framework addressing two key questions:
• How: How can dynamic prices help attackers infer passenger destinations more accurately?

• When: When is the optimal timing to conduct such attacks to maximize success while avoiding premature trajectory endings?



🏗️ Architecture Overview

Core Components

Our solution consists of two main components:

1. Conditional BiLSTM-Attention Model (CBAM) - The "How"

• BiLSTM Network: Captures forward and backward trajectory dependencies

• Conditional Recurrent Mechanism: Incorporates dynamic price information into initial states

• Attention Layer: Addresses long-range dependency problems

• Multi-modal Fusion: Integrates GPS trajectories, dynamic prices, and auxiliary features



2. Deep Reinforcement Learning Model - The "When"

• Double DQN Architecture: Determines optimal attack timing

• State Representation: Combines partial trajectories and prediction confidence

• Reward Design: Balances attack success and timeliness

• Real-time Decision Making: Dynamically adjusts attack strategy

📊 Key Features

• Dynamic Price Integration: First work to incorporate dynamic pricing in passenger destination inference

• Dual-Problem Solution: Addresses both inference accuracy (How) and timing optimization (When)

• Real-world Evaluation: Extensive experiments on real RoD service datasets

• Privacy Threat Analysis: Comprehensive study from attacker's perspective

• Modular Design: Flexible components for different attack scenarios

🗂️ Dataset Structure

Required Data Files


data/
├── training/
│   ├── train_80%_all_level8_new_clusters_1.csv
│   └── ...
├── validation/
│   └── ...
├── testing/
│   └── ...
└── processed_data/
    ├── clusters_5_ring/
    │   └── mean_shift clustering/
    │       └── 965_clusters_center_coords.csv
    └── ...


Data Fields

• coords_of_traj: GPS trajectory coordinates

• dest_cluster: Destination cluster ID (0-964)

• dp_cur, dp_30min_prev, dp_1h_prev: Dynamic price multipliers

• daytype, weather_*, hour_sin, hour_cos: Auxiliary features



🚀 Quick Start

Installation

# Clone the repository
git clone https://github.com/your-username/how-when-destination-inference.git
cd how-when-destination-inference

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


Basic Usage

1. Train the CBAM Model:
from models.cbam import CBAMModel

# Initialize model
model = CBAMModel(
    lstm_units=512,
    dense_units=1024,
    num_classes=965,
    use_dp_condition=True
)

# Train model
model.train(
    train_data_path="data/training/train_80%_all_level8_new_clusters_1.csv",
    val_data_path="data/validation/val_80%_all_level8_new_clusters_1.csv",
    epochs=200,
    batch_size=128
)


2. Destination Inference:
# Load trained model
model.load_weights("checkpoints/cbam_model.h5")

# Infer destination from partial trajectory
trajectory = [[116.35, 39.95], [116.36, 39.94], ...]  # Partial GPS coordinates
dynamic_prices = [1.42, 1.22, 1.15]  # Current, 30min ago, 1h ago
auxiliary_features = [0, 1, 0, 0, 0, 0, 0.8, 0.6]  # Day type, weather, time

predictions = model.infer_destination(
    trajectory=trajectory,
    dynamic_prices=dynamic_prices,
    auxiliary_features=auxiliary_features
)


3. RL-based Timing Decision:
from rl.ddqn_agent import DQNAgent

# Initialize RL agent
agent = DQNAgent(
    state_dim=3,  # [x, y, mean_distance]
    action_dim=2,  # [wait, attack]
    learning_rate=0.001
)

# Train timing decision model
agent.train(
    trajectories=data,
    num_episodes=10000,
    batch_size=64
)


📈 Experimental Results

Performance Metrics

Model Top-1 Distance Error (m) Top-5 Accuracy (%)

CBAM (Ours) 3,268 76.9

CBAM-noDP 3,318 75.9

T-CONV* 3,517 -

LSTM+* 3,395 -
Dynamic Price Impact Analysis
Scenario Improvement with Dynamic Prices

Early Trajectory (10%) +48.0% Top-5 Accuracy

Mid Trajectory (50%) Minimal improvement

Late Trajectory (90%) +1.3% Top-5 Accuracy



🔧 Advanced Usage

Ablation Studies

The code includes comprehensive ablation studies to analyze component contributions:
# Run ablation studies
python ablation_studies.py --config all

# Specific configurations
python ablation_studies.py --config no_dp_condition
python ablation_studies.py --config no_attention
python ablation_studies.py --config unidirectional


Transformer-based Variant

For advanced users, we provide a Transformer-based implementation:
from models.transformer_destination import TransformerDestinationPredictor

transformer_model = TransformerDestinationPredictor(
    d_model=256,
    num_heads=8,
    num_layers=3,
    dff=512
)


📋 Configuration Options

Model Parameters

config = {
    # Architecture
    "lstm_units": 512,
    "dense_units": 1024,
    "attention_units": 256,
    
    # Training
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 200,
    
    # Dynamic Prices
    "use_dp_condition": True,
    "dp_dim": 3,
    
    # Data
    "max_traj_len": 81,
    "grid_level": 8,
    "num_clusters": 965
}


📁 Repository Structure


how-when-destination-inference/
├── models/
│   ├── cbam.py                 # Main CBAM implementation
│   ├── transformer_model.py    # Transformer variant
│   └── ablation_models.py      # Ablation study configurations
├── rl/
│   ├── ddqn_agent.py          # Reinforcement learning agent
│   ├── environment.py         # RL environment
│   └── replay_buffer.py       # Experience replay
├── data/
│   ├── preprocess.py          # Data preprocessing
│   ├── loaders.py             # Data loading utilities
│   └── augmentation.py        # Data augmentation
├── utils/
│   ├── metrics.py             # Evaluation metrics
│   ├── visualization.py       # Plotting utilities
│   └── config.py              # Configuration management
├── experiments/
│   ├── ablation_studies.py    # Ablation experiments
│   ├── baseline_comparison.py # Baseline comparisons
│   └── case_studies.py        # Case studies
├── requirements.txt           # Python dependencies
├── setup.py                  # Package installation
└── README.md                 # This file


🎯 Citation

If you use this code in your research, please cite our paper:
@article{guo2025how,
  title={How and When: Inferring Passenger Destination Based on Dynamic Prices from an Attacker's Perspective},
  author={Guo, Suiming and Liu, Weilin and Sun, Yuxia and Chen, Chao and Liao, Chengwu and Liu, Yaxiao and Xu, Ke},
  journal={IEEE Transactions on Mobile Computing},
  volume={XX},
  number={X},
  pages={1--18},
  year={2025},
  publisher={IEEE}
}


👥 Contributors

• Suiming Guo (Jinan University) - [guosuiming@email.jnu.edu.cn]

• Weilin Liu (Jinan University)

• Yuxia Sun (Jinan - Corresponding Author

• Chao Chen (Chongqing University) - Corresponding Author

• Chengwu Liao (China Unicom)

• Yaxiao Liu (Tsinghua University)

• Ke Xu (Tsinghua University)

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing

We welcome contributions from the research community! Please feel free to submit issues, feature requests, or pull requests.

💡 Future Work

• Extension to other services with dynamic pricing (airline, hotel, smart grid)

• Improved trajectory representation learning

• Federated learning for privacy preservation

• Real-time attack detection and defense mechanisms

📞 Contact

For questions or discussions about this work, please contact:
• Yuxia Sun: tyxsun@email.jnu.edu.cn  

• Chao Chen: cschaochen@cqu.edu.cn



Disclaimer: This research is presented from an attacker's perspective to raise awareness about privacy implications of dynamic pricing. The authors do not endorse or encourage any malicious use of these techniques.
