# EchoWorld World Model Lab v2

这是基于现有 EchoWorld 规则状态机的可复现实验项目，不是 Genie 4 模型复现。

## 交付
- `EchoWorld_WorldModel_Lab.html`: 完整离线评测网页，包含 10/50 步曲线、边界测试和**真实视觉 CNN 预测图像**。
- `src/experiments.py`: 使用 v1 PyTorch 状态模型的 10/50 步自回归滚动和 2,136 个边界用例。
- `src/visual.py`: 根据 EchoWorld 状态机生成 32x32 RGB 图像，训练动作条件 CNN 预测下一帧，并保存权重和测试样本。
- `artifacts/`: 模型权重、JSON 指标、PNG 预测对比。

## 运行
```bash
pip install torch numpy pillow
python src/experiments.py --episodes 100
python src/visual.py --epochs 4 --train-size 650 --test-size 120
python src/build_lab.py
```

为了 CPU 运行，本次视觉实验只采用 650 个训练样本、4 轮，评测效果尚未超过简单的复制前帧基线。此结果必须如实保留。

## 评测边界
- 10/50 步输出是逐步将模型预测反馈到自身的**闭环状态预测**，各有 100 个独立随机动作序列；单步输入来自模型预测，Ground Truth 独立由规则引擎产生。
- 50 步误差可能小于 10 步误差，因为很多状态达到固定值、两组轨迹分布不同。不可将其解释为模型越滚越准。
- 边界测试包含不可达状态（人为构造的矛盾布尔组合），属于 OOD 鲁棒性压力测试，不是游戏正常路径性能。
- 视觉模型从程序化渲染的 RGB 帧、动作 one-hot 直接预测下一 RGB 帧，**不是基于状态的纯手绘输出冒充模型**；Ground Truth 是规则引擎生成帧，预测图是推理输出。
- 简单复制上一帧的 MSE 明显优于当前 CNN；需优先增加事件变化样本、采用残差预测和变化区加权损失。
- 原训练模型结构为两层 64 宽隐藏层 MLP，视觉网络为 3 层 32 通道卷积与 1x1 输出层，尚无视频 latent diffusion、无限场景、真实物理一致性能力。
