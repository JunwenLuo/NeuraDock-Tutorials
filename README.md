Neuradock EEG 开发者工具包 & 算法仓库
<img width="1280" height="800" alt="2631763012845_ pic" src="https://github.com/user-attachments/assets/b0706b75-89a2-48ee-8c37-38a0b8216b75" />


欢迎来到 Neuradock EEG 系统的官方算法仓库！🧠

本仓库旨在为开发者、研究人员和学生提供一个从零到一、全面、且易于上手的实战指南。无论您是脑机接口领域的新手，还是希望利用我们强大的硬件进行前沿探索的资深专家，这里都将为您提供所需的全部代码、教程和支持。

我们的目标是：赋能每一位开发者，将原始的脑电波转化为有价值的洞察和创新的应用。

✨ 核心特性

实时数据流： 通过简洁的 Python API，在毫秒间获取稳定、高质量的实时脑电数据。

标准化预处理： 提供即插即用的代码，快速完成滤波、去噪和伪影去除，确保信号质量。

经典范式复现： 包含 Alpha 波、SSVEP 和 P300 等核心脑机接口范式的完整实现，从范式设计、数据采集到离线分析一应俱全。

前沿应用评估框架： 提供针对非侵入式脑刺激、VR数字疗法、大模型认知负荷等热门领域的“开箱即用”评估方案。

模块化与可扩展： 所有代码均采用模块化设计，方便您轻松地将其集成到自己的项目中，或在其之上构建全新的应用。

配套视频讲解： 每一节教程都配有详细的视频讲解，确保您能看懂、学会、用好。

⚠️ 硬件要求

本仓库中的所有代码和教程均是为 Neuradock EEG 硬件 量身定制。API 和数据采集脚本无法与其他硬件兼容。

🚀 快速开始

准备硬件： 确保您的 Neuradock 设备已充电并准备就绪。

克隆仓库：

code
Bash
download
content_copy
expand_less
git clone https://github.com/your-username/neuradock-eeg-repo.git
cd neuradock-eeg-repo

配置环境： 我们强烈建议使用虚拟环境。

code
Bash
download
content_copy
expand_less
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

安装依赖：

code
Bash
download
content_copy
expand_less
pip install -r requirements.txt
```5.  **开始学习：** 从 `Tutorial_01_Hardware_Setup` 文件夹开始您的探索之旅！
📚 教程大纲

我们为您设计了10个由浅入深的实战教程。每一个都包含独立的 README.md 文件、源代码和视频讲解链接。

教程	主题	描述	快速跳转
Tutorial 01	硬件与环境配置	了解 Neuradock 硬件，学习正确的佩戴方法，并完成您本地 Python 开发环境的全部配置。	Go to Tutorial
Tutorial 02	API 入门与数据流获取	学习使用 Neuradock 的 Python API，实时获取、打印并可视化您的脑电数据流。	Go to Tutorial
Tutorial 03	数据预处理	掌握滤波、陷波、基线校正等关键预处理技术，从原始数据中提取出干净、可靠的脑电信号。	Go to Tutorial
Tutorial 04	Alpha 波：从监测到分析	以经典的 Alpha 波为例，学习如何进行频域分析（FFT），并实现一个简单的“闭眼/睁眼”状态分类器。	Go to Tutorial
Tutorial 05	解码视觉：SSVEP 脑机接口	构建您的第一个主动式 BCI 系统！学习设计 SSVEP 视觉刺激范式，并编写算法解码用户正在注视的目标。	Go to Tutorial
Tutorial 06	探索决策：P300 脑机接口	复现经典的 P300“oddball”范式，学习如何从脑电信号中检测出与“决策”相关的关键 ERP 成分。	Go to Tutorial
Tutorial 07	量化干预：评估非侵入式脑刺激	学习如何设计实验，利用脑电客观、量化地评估 tDCS/tACS 等非侵入式脑刺激方案的即时效果。	Go to Tutorial
Tutorial 08	沉浸式评估：VR 数字疗法	结合 VR 技术，学习如何同步记录用户在虚拟环境中的脑电活动，客观评估不同 VR 治疗方案的神经效应。	Go to Tutorial
Tutorial 09	对话未来：大模型认知负荷评估	探索前沿！学习如何利用脑电指标，客观评估用户在阅读和理解不同复杂度的大模型（如 GPT）输出内容时的认知负荷。	Go to Tutorial
Tutorial 10	框架赋能：开发您的专属应用	总结所有核心模块，提供一个灵活的开发框架，指导您如何快速整合已有代码，去实现您自己感兴趣的全新应用。	Go to Tutorial
🤝 如何贡献

我们欢迎任何形式的贡献！

如果您发现了 bug，请在 Issues 中提交。

如果您有好的想法或代码改进，欢迎提交 Pull Request。

📄 许可证

本项目采用 MIT License 授权。

💬 联系我们

技术支持： 如果您在使用中遇到任何问题，请优先在 Issues 区域提问。

商务合作： contact@neuradock.com <!-- 请替换为您的联系邮箱 -->

让我们一起开启探索大脑奥秘的旅程！ 🚀
