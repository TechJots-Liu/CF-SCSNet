# Accurate segmentation of pine wilt disease in UAV remote sensing imagery: From dataset construction to vision-language segmentation framework with coarse-fine labels
![overview](static/surveyRegion.png)
## :evergreen_tree: Overview

- **Research Background**：Pine wilt disease (PWD) is a major forest pest in Jilin Province, posing a severe threat to the pine forest ecosystems.  Traditional ground surveys are inefficient and limited in coverage, urgently requiring remote sensing technologies for large-scale dynamic monitoring.
- **Technical route**：To the best of our knowledge, this study marks the first effort to integrate semantic, textural, and structural information from UAV-based visible remote sensing imagery with tailored textual semantic descriptions customized to the pathological characteristics of pine wilt disease—for monitoring pine wilt disease.  Through cross-modal collaborative semantic modeling, the approach enables precise monitoring of pine wilt disease in small-scale, high-interference scenarios for the first time.
- **Core innovation**：
  - Task-specific dataset with aligned cross-modal annotations
  - Coarse-fine dual-granularity labeling framework for imbalanced small targets
  - Text-driven semantically coupled segmentation network (CF-SCSNet)
![overview](static/datasets.png)
## :bar_chart: 模型性能对比
| **数据集**         | **方法**               | **准确率(%)↑** | **召回率(%)↑** | **F1-score↑** | 模型下载 |
| :------------------ | :--------------------- | :------------- | :------------- | :------------ | :------- |
| **吉林省东部林区** | 改进U-Net++（本文）    | **94.2**       | **91.8**       | **92.3**      | [链接]   |
|                     | 传统SVM                | 78.5           | 76.3           | 77.4          | -        |
|                     | 原始U-Net              | 89.7           | 86.5           | 88.1          | -        |
| **公开数据集（Xiong et al. 2023）** | 改进U-Net++（本文） | 90.5           | 88.7           | 89.6          | [链接]   |

## :map: 监测结果可视化
<details open>
  <summary>吉林省松材线虫病空间分布</summary>
  <div align="center">
    <img src="./assets/jilin_distribution.jpg" width="80%" alt="吉林省病害分布">
    <p>左：2023年实测分布 | 右：本模型预测分布（红框为高风险区）</p>
  </div>
</details>

<details>
  <summary>病害特征对比</summary>
  <div align="center">
    <img src="./assets/symptom_comparison.jpg" width="80%" alt="健康与患病松树特征">
    <p>上：哨兵-2影像光谱曲线 | 下：无人机激光雷达冠层结构差异</p>
  </div>
</details>

<details>
  <summary>风险等级区划</summary>
  <div align="center">
    <img src="./assets/risk_zones.jpg" width="80%" alt="吉林省风险区划">
    <p>基于海拔、温度、林分密度的综合风险评估（红→黄→绿：高→中→低风险）</p>
  </div>
</details>

## :computer: 环境配置
<details open>
  <summary>依赖安装步骤</summary>
  
  1. **克隆仓库并创建环境**
     ```bash
     git clone https://github.com/你的用户名/你的仓库名.git
     cd 你的仓库名
     
     conda create -n pwdm python=3.10
     conda activate pwdm
