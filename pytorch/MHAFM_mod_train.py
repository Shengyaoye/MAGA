import itertools
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import OrderedDict, namedtuple, defaultdict

def get_auc(loader, model):
    pred, target = [], []                                                # 存储预测值和真实标签
    model.eval()                                                         # 切换模型为评估模式（关闭Dropout等）
    with torch.no_grad():                                                # 禁用梯度计算
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device).float()            # 数据移至指定设备
            y_hat = model(x)                                             # 前向传播获取预测
            pred += list(y_hat.cpu().numpy())                            # 预测值转CPU并存入列表
            target += list(y.cpu().numpy())
    auc = roc_auc_score(target, pred)                                    # 计算AUC
    return auc

class MultiHeadAFMLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, attention_factor, drop_rate=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.attention_factor = attention_factor
        print([embedding_size,self.num_heads])
        assert embedding_size % num_heads == 0, "embedding_size必须能被num_heads整除"
        self.head_dim = embedding_size // num_heads

        # 多头注意力参数
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*self.head_dim, attention_factor),  # 特征对拼接
                nn.ReLU(),
                nn.Linear(attention_factor, 1, bias=False)
            ) for _ in range(num_heads)
        ])
        
        # 最终投影参数
        self.projection = nn.Linear(num_heads * self.head_dim, 1)
        self.dropout = nn.Dropout(drop_rate)
        
        # 参数初始化
        for attn in self.attention_weights:
            nn.init.xavier_normal_(attn[0].weight)
            nn.init.xavier_normal_(attn[2].weight)
        nn.init.xavier_normal_(self.projection.weight)

    def _split_heads(self, x):
        """将嵌入向量分割为多头"""
        batch_size, num_pairs, _ = x.shape
        return x.view(batch_size, num_pairs, self.num_heads, self.head_dim)
    
    def _cross_pairs(self, embeds):
        """生成多头特征交叉对"""
        # 输入embeds: [B, F, E]
        batch_size, num_fields, _ = embeds.shape
        head_embeds = embeds.view(batch_size, num_fields, self.num_heads, self.head_dim)
        
        # 生成交叉对索引
        row, col = [], []
        for r, c in itertools.combinations(range(num_fields), 2):
            row.append(r)
            col.append(c)
        pairs = (head_embeds[:,row], head_embeds[:,col])  # 每对形状[B, P, H, D]
        
        # 拼接特征对
        return torch.cat(pairs, dim=-1)  # [B, P, H, 2D]

    def forward(self, inputs):
        # inputs: 多个嵌入向量组成的列表 -> [B, F, E]
        embeds = torch.cat(inputs, dim=1)  # [B, F, E]
        batch_size, num_fields, _ = embeds.shape
        
        # 生成多头交叉对
        cross_pairs = self._cross_pairs(embeds)  # [B, P, H, 2D]
        num_pairs = cross_pairs.shape[1]
        
        # 多头注意力计算
        head_outputs = []
        for h in range(self.num_heads):
            # 当前头的交叉对 [B, P, 2D]
            pair_h = cross_pairs[:, :, h, :]
            
            # 注意力得分 [B, P, 1]
            attn_logits = self.attention_weights[h](pair_h)  # [B, P, 1]
            attn_weights = torch.softmax(attn_logits, dim=1)
            
            # 加权交叉项 [B, P, D]
            weighted_cross = attn_weights * cross_pairs[:, :, h, :self.head_dim]
            head_outputs.append(weighted_cross.sum(dim=1))  # [B, D]
        
        # 多头聚合
        multi_head_out = torch.cat(head_outputs, dim=-1)  # [B, H*D]
        output = self.projection(multi_head_out)  # [B, E]
        return self.dropout(output)  # 保持原AFM输出形状[B, 1]
    
class AFM(nn.Module):
    def __init__(self, feat_size, embedding_size, dnn_feature_columns,
                 use_attention=True, attention_factor=8, l2_reg=0.00001, drop_rate=0.1):#use_attention,TrueAFM;FalseFM
        super(AFM, self).__init__()

        # 提取稀疏特征列（如类别型特征）
        self.sparse_feature_columns = list(filter(lambda x: x[1]=='sparse', dnn_feature_columns))
        # 嵌入层字典：为每个稀疏特征创建嵌入表
        self.embedding_dic = nn.ModuleDict({
            feat[0]:nn.Embedding(feat_size[feat[0]], embedding_size, sparse=False) for feat in self.sparse_feature_columns
        })
        # 特征索引映射：记录每个特征在输入中的位置
        self.feature_index = defaultdict(int)
        start = 0
        for feat in feat_size:
            self.feature_index[feat] = start
            start += 1

         # 选择使用AFM层或普通FM,True:AFM,False:FM
        self.use_attention = use_attention
        if self.use_attention:
            self.fm = MultiHeadAFMLayer(embedding_size, num_heads=4, attention_factor=16, drop_rate=0.1) 
        else:
            self.fm = FM()
        #self.fm返回的是二阶交叉部分

        # 线性部分（全连接层）
        dnn_hidden_units = [len(feat_size), 1]                      # 输入维度为特征数，输出1维
        self.linear = nn.ModuleList([
            nn.Linear(dnn_hidden_units[i], dnn_hidden_units[i + 1]) for i in range(len(dnn_hidden_units) - 1)
        ])
        # 初始化线性层权重
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.00001)
        self.out = nn.Sigmoid()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, X):
        # 稀疏特征嵌入：将每个稀疏特征转换为嵌入向量
        sparse_embedding = [self.embedding_dic[feat[0]](X[:, self.feature_index[feat[0]]].long()).reshape(X.shape[0], 1, -1)
                            for feat in self.sparse_feature_columns]    # 列表中的每个元素形状：(batch_size, 1, embedding_size)
        # 线性部分处理（全连接层）
        logit = X
        for i in range(len(self.linear)):
            fc = self.linear[i](logit)
            fc = self.act(fc)
            fc = self.dropout(fc)
            logit = fc

        # 合并FM/AFM交叉项
        if self.use_attention:
            logit += self.fm(sparse_embedding)
        else:
            logit += self.fm(torch.cat(sparse_embedding, dim=1))

        # 输出概率
        y_pred = torch.sigmoid(logit) # Sigmoid将logit转为概率 # 这里踩了个坑 最开始写成立 nn.softmax(dim=1) 结果训练集 验证集loss都不降

        return y_pred
    


import joblib

def preprocess_in_chunks(file_path, chunk_size=1e6):
    """分块预处理数据并保存
    关联模块：
    为ChunkedDataset提供预处理后的分块数据。
    保存的label_encoders.pkl和minmax_scaler.pkl用于后续推理时的特征转换
    """
    # 初始化预处理器
    sparse_features = [f'C{i}' for i in range(1, 27)]
    dense_features = [f'I{i}' for i in range(1, 14)]
    
    # 首次遍历拟合预处理参数,# 第一次遍历：拟合LabelEncoder
    label_encoders = {}
    for feat in sparse_features:
        lbe = LabelEncoder()
        # 分块拟合稀疏特征编码器
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=[feat]):
            lbe.fit(chunk[feat].fillna('-1').astype(str))
        label_encoders[feat] = lbe
    joblib.dump(label_encoders, 'label_encoders.pkl') # 保存编码器
    
    # 拟合密集特征归一化
    nms = MinMaxScaler()
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=dense_features):
        nms.partial_fit(chunk.fillna(0))    # 支持分块更新统计量
    joblib.dump(nms, 'minmax_scaler.pkl')
    
    # 第二次遍历转换数据并保存分块
    chunk_id = 0
    for raw_chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # 稀疏特征编码
        for feat in sparse_features:
            raw_chunk[feat] = label_encoders[feat].transform(
                raw_chunk[feat].fillna('-1').astype(str)
            )
        # 密集特征归一化
        raw_chunk[dense_features] = nms.transform(raw_chunk[dense_features].fillna(0))
        # 保存预处理后的分块
        raw_chunk.to_parquet(f'train_chunk_{chunk_id}.parquet')
        chunk_id += 1
        
        # 计算特征维度参数
    feat_sizes = {}
    for feat in sparse_features:
        feat_sizes[feat] = len(label_encoders[feat].classes_)
    for feat in dense_features:
        feat_sizes[feat] = 1
    
    # 定义特征列类型
    dnn_feature_columns = [(feat, 'sparse') for feat in sparse_features] + [(feat, 'dense') for feat in dense_features]
    
    # 保存所有预处理参数
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(nms, 'minmax_scaler.pkl')
    joblib.dump(feat_sizes, 'feat_sizes.pkl')
    joblib.dump(dnn_feature_columns, 'dnn_feature_columns.pkl')

class ChunkedDataset(Dataset):
    """​作用：实现高效的分块数据加载，支持全局随机打乱。
    ​关键功能：
    ​全局索引管理：生成所有样本的(chunk_id, row_id)索引，支持跨分块随机访问。
    ​惰性加载：仅当需要访问某分块时将其加载到内存，减少内存占用。
    ​数据格式转换：将NumPy数组转换为PyTorch张量
    """
    def __init__(self, chunk_files, shuffle=True):
        self.chunk_files = chunk_files  # Parquet文件列表
        self.shuffle = shuffle  # 是否打乱数据顺序
        self.global_indices = self._generate_global_indices()
        self.current_chunk = None
        self.current_chunk_id = -1

    def _generate_global_indices(self):
        """生成全局索引列表 [(chunk_id, row_id), ...]"""
        indices = []
        for chunk_id, file in enumerate(self.chunk_files):
            chunk = pd.read_parquet(file)
            indices += [(chunk_id, i) for i in range(len(chunk))]
        if self.shuffle:
            np.random.shuffle(indices)  # 全局打乱
        return indices

    def _load_chunk(self, chunk_id):
        """加载指定分块到内存"""
        if chunk_id != self.current_chunk_id:
            self.current_chunk = pd.read_parquet(self.chunk_files[chunk_id]).values
            self.current_chunk_id = chunk_id

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, idx):
        chunk_id, row_id = self.global_indices[idx]
        self._load_chunk(chunk_id)  # 加载对应分块
        features = self.current_chunk[row_id, 1:].astype(np.float32)
        label = self.current_chunk[row_id, 0].astype(np.float32)
        return torch.tensor(features), torch.tensor(label)

        
if __name__ == '__main__':
    # 初始化环境
    torch.manual_seed(2022)
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    
    # 1. 预处理（首次运行需执行）
    if not os.path.exists('label_encoders.pkl'):
        preprocess_in_chunks('/root/yesy/MAGA/pytorch/train.txt')
    
    import glob
    # 2. 加载预处理后的分块数据
    chunk_files = sorted(glob.glob('train_chunk_*.parquet'))
    full_dataset = ChunkedDataset(chunk_files, shuffle=True)
    
    # 动态分割训练集和测试集（80%训练，20%测试）
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(2022)  # 固定随机种子
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4096,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4096,
        num_workers=2,
        pin_memory=True
    )
    
    # 加载预处理参数
    label_encoders = joblib.load('label_encoders.pkl')
    nms = joblib.load('minmax_scaler.pkl')
    feat_sizes = joblib.load('feat_sizes.pkl')
    dnn_feature_columns = joblib.load('dnn_feature_columns.pkl')
    
    # 硬编码特征列名（需与预处理一致）
    sparse_features = [f'C{i}' for i in range(1, 27)]
    dense_features = [f'I{i}' for i in range(1, 14)]
    
    # 3. 初始化模型与优化器
    model = AFM(
        feat_sizes=feat_sizes,
        embedding_size=8,
        dnn_feature_columns=dnn_feature_columns,
        use_attention=True
    ).to(device)
  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    
    # 4. 训练循环（含早停）
    best_auc = 0.0
    patience = 5  # 允许连续不提升的epoch数
    no_improve = 0
    best_model_state = None  # 保存最佳模型状态
    
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            # 混合精度训练
            with torch.cuda.amp.autocast():
                y_hat = model(x.to(device))
                loss = F.binary_cross_entropy(y_hat, y.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
        
            # 验证阶段
        current_auc = get_auc(test_loader, model)
        
        # 更新最佳模型
        if current_auc > best_auc:
            best_auc = current_auc
            best_model_state = model.state_dict().copy()  # 深拷贝模型参数
            no_improve = 0
            print(f"🎯 发现新最佳AUC: {best_auc:.4f}")
        else:
            no_improve += 1
            print(f"⚠️ AUC未提升 ({no_improve}/{patience})")

        # 早停判断
        if no_improve >= patience:
            print(f"🛑 提前终止! 最佳AUC: {best_auc:.4f}")
            model.load_state_dict(best_model_state)  # 恢复最佳参数
            break

    # 训练结束后保存最终最佳模型
    torch.save(best_model_state, 'best_model.pth')
