# FedRefiner算法实现

## 算法概述

FedRefiner是一个两阶段的联邦学习算法，专门用于处理标签噪声问题：

### 第一阶段：预热阶段
- 使用Symmetric Cross Entropy (SCE) 损失进行预热训练
- 帮助模型学习基本的特征表示
- 为第二阶段的噪声检测和标签修正做准备

### 第二阶段：精炼阶段
- 使用Gaussian Mixture Model (GMM) 识别噪声样本
- 对高置信度的噪声样本进行标签修正
- 使用基于TriTAN的对比学习损失进行特征学习

## 核心特性

1. **噪声样本识别**：使用GMM对交叉熵损失进行聚类，自动识别噪声样本
2. **智能标签修正**：对置信度高于阈值`tao`的噪声样本，使用模型预测作为新标签
3. **对比学习**：基于TriTAN的loss_2实现对比损失，提升特征表示质量
4. **两阶段训练**：预热阶段稳定训练，精炼阶段优化性能

## 参数配置

在`args.py`中已添加以下参数：

```python
# FedRefiner参数
parser.add_argument('--tao', type=float, default=0.8, help='confidence threshold for label correction in FedRefiner')
parser.add_argument('--fedrefiner_warmup', type=int, default=5, help='warmup rounds for FedRefiner')
parser.add_argument('--contrastive_weight', type=float, default=0.1, help='weight for contrastive loss in FedRefiner')
```

### 参数说明

- `tao`: 置信度阈值，用于决定是否对噪声样本进行标签修正（默认0.8）
- `fedrefiner_warmup`: 预热轮数（默认5轮）
- `contrastive_weight`: 对比损失权重（默认0.1）

## 使用方法

### 1. 运行FedRefiner算法

```bash
python FL_train.py --alg fedrefiner --dataset RS-15 --model resnet18 --n_parties 10 --comm_round 100 --epochs 5 --tao 0.8 --fedrefiner_warmup 5 --contrastive_weight 0.1
```

### 2. 参数调优建议

- **tao (置信度阈值)**：
  - 较低值（0.6-0.7）：更激进的标签修正，可能引入更多错误
  - 较高值（0.8-0.9）：更保守的标签修正，更安全但效果可能有限
  - 建议从0.8开始调优

- **contrastive_weight (对比损失权重)**：
  - 较低值（0.05-0.1）：主要依赖SCE损失
  - 较高值（0.2-0.3）：更强调对比学习
  - 建议从0.1开始调优

- **fedrefiner_warmup (预热轮数)**：
  - 较少轮数（3-5）：快速进入精炼阶段
  - 较多轮数（8-10）：更稳定的预热
  - 建议根据数据集大小调整

## 算法流程

### 第一阶段（预热）
```
1. 加载全局模型到本地客户端
2. 使用SCE损失进行本地训练
3. 聚合本地模型更新
4. 重复直到完成预热轮数
```

### 第二阶段（精炼）
```
1. 加载全局模型到本地客户端
2. 对每个batch：
   a. 计算交叉熵损失
   b. 使用GMM识别噪声样本
   c. 计算模型预测置信度
   d. 对高置信度噪声样本进行标签修正
   e. 计算SCE损失（使用修正后的标签）
   f. 计算对比损失
   g. 更新模型参数
3. 聚合本地模型更新
4. 重复直到完成所有通信轮数
```

## 文件结构

```
algorithms/
├── fedrefiner.py          # FedRefiner算法主实现
├── symmetricCE.py         # SCE损失函数
└── client.py              # 客户端训练函数（已更新）

FL_train.py                # 主训练脚本（已更新）
args.py                    # 参数配置（已更新）
```

## 核心函数

### `fedrefiner_alg()`
- 主算法函数，协调两阶段训练
- 处理模型聚合和评估

### `train_net_fedrefiner_stage2()`
- 第二阶段的核心训练逻辑
- 实现GMM噪声检测、标签修正和对比损失

### `compute_contrastive_loss()`
- 基于TriTAN的对比损失计算
- 使用硬负样本挖掘

## 优化器配置

FedRefiner算法统一使用SGD优化器：
- **优化器类型**：SGD
- **动量参数**：0.9
- **权重衰减**：args.reg
- **学习率**：args.lr

这种配置确保了训练的稳定性和一致性。

## 性能优势

1. **鲁棒性**：通过GMM自动识别噪声样本，减少噪声影响
2. **自适应性**：根据模型置信度动态调整标签修正策略
3. **特征学习**：对比损失提升特征表示质量
4. **稳定性**：两阶段设计确保训练稳定性

## 注意事项

1. **计算开销**：GMM聚类和对比损失计算会增加训练时间
2. **内存使用**：对比损失需要计算相似度矩阵，注意batch size设置
3. **参数调优**：需要根据具体数据集调整`tao`和`contrastive_weight`
4. **依赖包**：需要安装`sklearn`用于GMM聚类
5. **优化器**：算法只使用SGD优化器，momentum=0.9，weight_decay=args.reg

## 测试验证

运行测试脚本验证算法组件：

```bash
python simple_test_fedrefiner.py
```

该脚本会测试：
- GMM噪声检测功能
- 标签修正逻辑
- 对比损失计算
- SCE损失函数
- 完整的第二阶段逻辑

## 扩展性

FedRefiner算法具有良好的扩展性：

1. **可替换噪声检测方法**：可以替换GMM为其他聚类方法
2. **可调整对比损失**：可以修改对比损失的计算方式
3. **可集成其他技术**：可以结合其他联邦学习技术
4. **可适配不同数据集**：通过参数调整适配不同场景
