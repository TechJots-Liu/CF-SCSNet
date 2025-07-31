# Accurate segmentation of pine wilt disease in UAV remote sensing imagery: From dataset construction to vision-language segmentation framework with coarse-fine labels
![overview](static/surveyRegion.png)
https://github.com/TechJots-Liu/CF-SCSNet/blob/main/static/datasetes.png
## :newspaper:R Research progress
- **[2025/04/21]** 完成吉林省2024年松材线虫病遥感监测数据集构建，包含10万+样点数据 :chart_with_upwards_trend:
- **[2025/XX/XX]** 提出基于多源遥感数据（哨兵-2、无人机LiDAR）的融合检测模型，F1-score达92.3% :star:
- **[2024/XX/XX]** 项目启动，聚焦吉林省东部林区松材线虫病早期预警研究 :seedling:

## :evergreen_tree: 研究概述

- **研究背景**：松材线虫病是吉林省林业重大病虫害，对长白山松树林生态系统构成严重威胁。传统地面调查效率低、覆盖范围有限，亟需遥感技术实现大范围动态监测。
- **技术路线**：首次融合光学遥感（哨兵-2）、无人机高光谱与LiDAR数据，通过改进的U-Net++模型提取病木光谱-结构特征，并结合气象因子构建时空预测模型。
- **核心创新**：
  - 提出"病害胁迫指数（DSI）"量化松树健康状态
  - 开发适用于寒区森林的抗干扰检测模块（针对吉林省冬季积雪、云雾干扰）
  - 构建吉林省松材线虫病风险等级区划模型

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
