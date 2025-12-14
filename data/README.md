# Dataset Description | 数据集说明

This directory describes the data organization used in the experiments of the paper:

本目录用于说明论文实验中所使用的数据组织形式，对应论文：

> **How and When: Inferring Passenger Destination Based on Dynamic Prices from an Attacker’s Perspective**

---

## Data Availability | 数据可用性说明

Due to **privacy, confidentiality, and licensing constraints**, the raw Ride-on-Demand (RoD) datasets used in this study **cannot be publicly released**.

由于涉及**用户隐私、数据保密协议及授权限制**，本文所使用的网约车（Ride-on-Demand, RoD）原始数据 **无法公开发布**。

---

## Data Organization | 数据组织结构

The expected directory structure is as follows:

实验代码所期望的数据目录结构如下：

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
```
## Data Fields | 数据字段说明

Each trajectory record includes the following key fields:

每条轨迹数据主要包含以下字段：

- **coords_of_traj**  
  GPS trajectory coordinates of a trip  
  轨迹对应的 GPS 坐标序列  

- **dest_cluster**  
  Destination cluster identifier  
  目的地区域簇编号  

- **dp_cur**  
  Current dynamic price multiplier  
  当前时刻的动态价格倍率  

- **dp_30min_prev**  
  Dynamic price multiplier 30 minutes before the current time  
  前 30 分钟的动态价格倍率  

- **dp_1h_prev**  
  Dynamic price multiplier 1 hour before the current time  
  前 1 小时的动态价格倍率  

- **daytype**  
  Indicator of weekday or weekend  
  工作日 / 周末标识  

- **weather_\***  
  Weather-related contextual features  
  与天气相关的上下文特征  

- **hour_sin, hour_cos**  
  Time-of-day features encoded using sine and cosine functions  
  使用正余弦编码的时间特征  

---

## Notes | 说明

- The provided code assumes that the dataset has been preprocessed into the above format.  
  代码默认数据已按照上述格式完成预处理。

- The clustering centers are generated using a mean-shift clustering algorithm.  
  目的地区域中心通过 Mean-Shift 聚类算法生成。

- Researchers interested in reproducing the experiments may adapt the code to alternative datasets with similar structures.  
  有兴趣复现实验的研究者可将代码适配至具有相似结构的其他数据集。
---

## Disclaimer | 免责声明

The dataset is used solely for academic research purposes.  
The authors do not distribute or provide access to the original data.

本数据仅用于学术研究目的，作者不提供原始数据的分发或访问权限。

