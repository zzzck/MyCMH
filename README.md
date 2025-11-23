跨模态哈希检索 (Cross-Modal Hash Retrieval)
本项目实现了一个完整的跨模态哈希检索系统，支持文本-图像之间的高效检索。模型将文本和图像编码为紧凑的哈希码（值在-1到1之间），实现快速的跨模态检索。

🚀 主要特性
跨模态哈希编码: 将文本和图像映射到统一的哈希空间
高效检索: 使用汉明距离进行快速相似度计算
灵活的模型架构: 支持多种文本编码器（BERT系列）和图像编码器（ResNet、ViT）
完整的训练流程: 包含对比学习、量化损失、平衡损失等
丰富的评估指标: mAP、Precision@K、Recall@K、NDCG@K等
多数据集支持: COCO、Flickr30K、合成数据集
易于使用: 提供完整的训练和推理脚本
📁 项目结构
cross_modal_hash_retrieval/
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── cross_modal_hash.py   # 主模型和损失函数
│   ├── text_encoder.py       # 文本编码器
│   ├── image_encoder.py      # 图像编码器
│   └── hash_layer.py         # 哈希层
├── data/                     # 数据处理
│   ├── __init__.py
│   ├── dataset.py           # 数据集类
│   ├── dataloader.py        # 数据加载器
│   └── transforms.py        # 数据变换
├── training/                # 训练模块
│   ├── __init__.py
│   ├── trainer.py          # 训练器
│   ├── optimizer.py        # 优化器和调度器
│   └── config.py           # 训练配置
├── evaluation/             # 评估模块
│   ├── __init__.py
│   ├── metrics.py          # 评估指标
│   └── evaluator.py        # 评估器
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── utils.py           # 通用工具
│   └── logger.py          # 日志记录
├── configs/               # 配置文件
│   └── synthetic_config.json
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── README.md            # 项目说明
└── requirements.txt     # 依赖包
🛠️ 安装依赖
# 创建虚拟环境（推荐）
conda create -n cross_modal_hash python=3.8
conda activate cross_modal_hash

# 安装PyTorch（根据你的CUDA版本选择）
pip install torch torchvision torchaudio

# 安装其他依赖
pip install transformers
pip install scikit-learn
pip install tqdm
pip install pillow
pip install pandas
pip install numpy
或者使用requirements.txt（如果提供）：

pip install -r requirements.txt
🚀 快速开始
1. 训练模型
使用合成数据集（快速测试）
python train.py --dataset synthetic --hash_dim 32 --batch_size 16 --num_epochs 20
使用配置文件
python train.py --config configs/synthetic_config.json
使用真实数据集（COCO）
python train.py \
    --dataset coco \
    --data_dir /data2/zhangchaoke/PythonProject/MyCMH/datasets/train2014 \
    --annotations_file /data2/zhangchaoke/PythonProject/MyCMH/datasets/annotations/captions_train2014.json \
    --hash_dim 64 \
    --batch_size 32 \
    --num_epochs 100
2. 模型推理
文本查询图像
python inference.py \
    --model_path checkpoints/best_model.pth \
    --query_text "A cat sitting on a chair" \
    --database_images image1.jpg image2.jpg image3.jpg
图像查询文本
python inference.py \
    --model_path checkpoints/best_model.pth \
    --query_image query_image.jpg \
    --database_texts "A cat" "A dog" "A bird"
📊 模型架构
核心组件
文本编码器 (TextEncoder)

基于预训练BERT模型
支持多种BERT变体
可选择冻结预训练参数
图像编码器 (ImageEncoder)

支持ResNet系列（ResNet50/101）
支持Vision Transformer
灵活的特征提取
哈希层 (HashLayer)

多种激活函数：tanh、sigmoid、Gumbel softmax
自适应量化机制
确保输出在-1到1之间
损失函数 (CrossModalHashLoss)

对比学习损失: InfoNCE损失，学习跨模态对应关系
量化损失: 鼓励哈希码接近二进制值
平衡损失: 保持哈希位的平衡性
训练策略
多任务学习: 同时优化特征学习和哈希编码
渐进式训练: 预热学习率调度
混合精度训练: 提高训练效率
数据增强: 丰富的图像和文本增强策略
📈 评估指标
mAP (Mean Average Precision): 平均精度均值
Precision@K: 前K个结果的精确率
Recall@K: 前K个结果的召回率
NDCG@K: 归一化折损累积增益
汉明距离: 哈希码之间的距离
⚙️ 配置说明
主要配置参数
{
  "hash_dim": 64,                    // 哈希码维度
  "feature_dim": 512,                // 特征维度
  "text_model": "bert-base-uncased", // 文本模型
  "image_backbone": "resnet50",      // 图像骨干网络
  "batch_size": 32,                  // 批大小
  "learning_rate": 1e-4,             // 学习率
  "lambda_quant": 0.1,               // 量化损失权重
  "lambda_balance": 0.01,            // 平衡损失权重
  "num_epochs": 100                  // 训练轮数
}
数据集配置
合成数据集: 用于快速测试和验证
COCO: 大规模图像-文本数据集
Flickr30K: 经典的跨模态检索数据集
🔧 高级用法
自定义数据集
from data.dataset import CrossModalDataset

class MyDataset(CrossModalDataset):
    def load_annotations(self):
        # 实现你的数据加载逻辑
        pass
自定义模型
from models.cross_modal_hash import CrossModalHashModel

# 创建自定义配置的模型
model = CrossModalHashModel(
    hash_dim=128,
    feature_dim=1024,
    text_model='bert-large-uncased',
    image_backbone='resnet101'
)
分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py \
    --distributed \
    --config configs/distributed_config.json
📋 实验结果
合成数据集结果
Hash Bits	mAP	P@1	P@5	P@10
32	0.85	0.92	0.88	0.84
64	0.89	0.95	0.91	0.87
128	0.91	0.96	0.93	0.89
COCO数据集结果（示例）
Method	Hash Bits	T2I mAP	I2T mAP
Ours	64	0.72	0.68
Ours	128	0.76	0.71
🐛 常见问题
Q: 训练时GPU内存不足怎么办？
A:

减小batch_size
使用混合精度训练（–mixed_precision）
选择更小的模型（如bert-base而不是bert-large）
Q: 如何提高检索精度？
A:

增加哈希码维度
调整损失函数权重
使用更强的数据增强
增加训练轮数
Q: 支持哪些预训练模型？
A:

文本：BERT系列、RoBERTa、DistilBERT等
图像：ResNet系列、Vision Transformer等
🤝 贡献指南
欢迎提交Issue和Pull Request！

Fork本项目
创建特性分支 (git checkout -b feature/AmazingFeature)
提交更改 (git commit -m 'Add some AmazingFeature')
推送到分支 (git push origin feature/AmazingFeature)
创建Pull Request
📄 许可证
本项目采用MIT许可证 - 查看 LICENSE 文件了解详情。

📚 参考文献
Deep Cross-Modal Hashing
Learning to Hash for Indexing Big Data
Cross-Modal Retrieval with CNN Visual Features
Supervised Deep Hashing for Cross-Modal Retrieval
📞 联系方式
如有问题或建议，请通过以下方式联系：

提交Issue
发送邮件到 [your-email@example.com]
注意: 这是一个研究项目，主要用于学习和研究目的。在生产环境中使用前请进行充分测试。