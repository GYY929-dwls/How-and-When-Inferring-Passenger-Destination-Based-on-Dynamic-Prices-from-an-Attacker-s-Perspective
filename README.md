
This repository provides the reference implementation for the paper:

> **How and When: Inferring Passenger Destination Based on Dynamic Prices from an Attacker’s Perspective**

The paper is currently **under review** at *IEEE Transactions on Mobile Computing (TMC)*.


本代码仓库为论文  
**《How and When: Inferring Passenger Destination Based on Dynamic Prices from an Attacker’s Perspective》**  
的参考实现。

该论文目前**正在投稿至 IEEE Transactions on Mobile Computing（TMC）审稿中**。

---

## 📖 Abstract | 摘要

### English

Ride-on-Demand (RoD) services such as Uber and Didi have significantly improved urban transportation efficiency through dynamic pricing mechanisms. However, such pricing strategies also introduce serious privacy risks.

This work investigates how attackers can leverage dynamic pricing information to infer passenger destinations more accurately and further explores when such attacks should be launched to maximize inference success while avoiding premature termination due to incomplete trajectories.

We focus on two key research questions:

- **How**: How can dynamic prices assist attackers in improving destination inference accuracy?
- **When**: When is the optimal timing to conduct inference attacks during trajectory evolution?

---

### 中文

网约车（Ride-on-Demand，RoD）服务（如 Uber、滴滴）通过动态定价机制显著提升了城市交通运行效率，但与此同时也引入了潜在的隐私风险。

本文从**攻击者视角**出发，研究攻击者如何利用动态价格信息更准确地推断乘客目的地，并进一步探讨在轨迹演化过程中，攻击者应在何时发起攻击以在保证成功率的同时避免因轨迹过短而失败。

本文重点围绕以下两个问题展开：

- **How（如何）**：动态价格如何辅助提升目的地推断精度？
- **When（何时）**：在轨迹演化过程中，攻击的最优时机是什么？
<img width="829" height="349" alt="1" src="https://github.com/user-attachments/assets/fe8e2591-ffd3-448b-8037-b8dea7b52954" />

---


## 🏗️ Architecture Overview | 架构概览

### Core Components | 核心模块

Our framework consists of two main components:

本研究提出的整体框架由两个核心模块组成：

---

### 1. Conditional BiLSTM-Attention Model (CBAM) — *The “How”*  
**条件式 BiLSTM-注意力模型（CBAM）——“如何推断”**

- BiLSTM network capturing forward and backward trajectory dependencies  
- Conditional recurrent mechanism integrating dynamic price information  
- Attention layer for modeling long-range dependencies  
- Multi-modal fusion of GPS trajectories, dynamic prices, and auxiliary features  
<img width="881" height="580" alt="1" src="https://github.com/user-attachments/assets/09483dc1-57f2-4c64-9b55-b801e0098d6a" />


---

### 2. Deep Reinforcement Learning Model — *The “When”*  
**深度强化学习模型——“何时攻击”**

- Double DQN architecture for optimal attack timing decisions  
- State representation combining partial trajectories and prediction confidence  
- Reward design balancing accuracy and timeliness  
- Real-time decision-making during trajectory evolution  
<img width="735" height="507" alt="1" src="https://github.com/user-attachments/assets/b2f80eac-ba7b-4773-b975-416d6637f8cf" />

---



## 📊 Key Features | 关键特性

- Dynamic price integration for destination inference  
- Joint modeling of inference accuracy (**How**) and attack timing (**When**)  
- Extensive evaluation on real-world RoD datasets  
- Privacy threat analysis from an attacker’s perspective  
- Modular and extensible framework design  

---

## 🗂️ Example Data Architecture | 示例数据集结构

### Required Data Files

```text
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
````

### Data Fields | 数据字段说明

* **coords_of_traj**: GPS trajectory coordinates
* **dest_cluster**: destination cluster ID (0–964)
* **dp_cur, dp_30min_prev, dp_1h_prev**: dynamic price multipliers
* **daytype, weather_*, hour_sin, hour_cos**: auxiliary contextual features

---

## 🚀 Quick Start | 快速开始

> **Note**: This repository provides research-oriented reference implementations for experimental reproduction rather than production use.
> **说明**：本仓库代码用于科研复现实验，不作为工业级或生产级系统。

```bash
# Clone the repository
git clone https://github.com/your-username/how-when-destination-inference.git
cd how-when-destination-inference

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Train CBAM model (The "How")
python models/CBAM.py

# Train Transformer-based variant
python models/Transformer.py

# Train Double DQN attacker (The "When")
python rl/Double_DQN.py

# Run ablation studies
python experiments/simple_ablation.py
```

---

## 📈 Comparing with Baselines Experimental Results | 基线实验对照实验结果

| Model       | Top-1 Avg.Distance Error (m) | 
| ----------- | ------------------------ | 
| CBAM-0.5h   | 3268                     | 
| CBAM-noDP   | 3318                     |
| T-CONV*     | 3517                     | 
| LSTM+*      | 3395                     | 
| Seq2Seq*	  | 4,005	                 |
| MLP*	      | 4,387	                 |(pytest = 80%)


### Dynamic Price Impact Analysis

| Scenario               | Improvement           |
| ---------------------- | --------------------- |
| Early trajectory (10%) | +48.0% Top-5 Accuracy |
| Mid trajectory (50%)   | Minimal               |
| Late trajectory (90%)  |  Slight Refinement    |

---

## 📁 Repository Structure | 仓库结构

```text
how-when-destination-inference/
├── models/
├── rl/
├── data/
├── experiments/
├── utils/
├── requirements.txt
└── README.md
```


---

## 👥 Contributors | 作者与贡献者

* **Suiming Guo** (Jinan University)
* **Weilin Liu** (Jinan University)
* **Yuxia Sun** (Jinan University, Corresponding Author)
* **Chao Chen** (Chongqing University, Corresponding Author)
* **Chengwu Liao** (China Unicom)
* **Yaxiao Liu** (Tsinghua University)
* **Ke Xu** (Tsinghua University)

---


## 🤝 Contributing | 贡献说明

We welcome contributions from the research community.
欢迎学术界同行提出问题、建议或贡献代码。

---

## 💡 Future Work | 未来工作

* Extension to other dynamic pricing services (airline, hotel, smart grid)
* Improved trajectory representation learning
* Federated learning for privacy preservation
* Real-time attack detection and defense

---


### Disclaimer | 免责声明

This research is presented from an attacker’s perspective to highlight privacy risks.
The authors do not encourage or endorse any malicious use.

本研究从攻击者视角出发，旨在揭示潜在隐私风险，作者不支持或鼓励任何恶意使用行为。



##  Note | 注意

This repository is provided to support the peer-review process.
The code structure and documentation may be further refined after paper acceptance.


本仓库用于支持论文审稿与学术交流。论文录用后，代码结构与文档可能会进一步完善。

