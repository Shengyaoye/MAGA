import itertools

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
    auc = roc_auc_score(target, pred)                                   # 计算AUC
    return auc

#计算所有特征对的二阶交叉项之和
class FM(nn.Module):
    def __init__(self):
        super(FM, self).__init__()
    def forward(self, inputs):
        fm_input = inputs                                                              # 输入形状：(batch_size, num_features, embedding_size)
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)         # (sum x_i)^2
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)            # sum x_i^2
        cross_term = square_of_sum - sum_of_square                                      # 差集
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)                  # 沿嵌入维度求和
        return cross_term                                                               # 输出形状：(batch_size, 1)

class AFMLayer(nn.Module):
    def __init__(self, embedding_size, attention_factor=4, l2_reg=0.0, drop_rate=0.0):
        super(AFMLayer, self).__init__()

        # 参数定义
        self.embedding_size = embedding_size                                            # 嵌入维度
        self.attention_factor = attention_factor                                        # 注意力隐层维度
        self.l2_reg = l2_reg                                                            # L2正则化系数（未使用）
        self.drop_rate = drop_rate                                                      # Dropout比例

        # 注意力权重参数
        self.attention_W = nn.Parameter(torch.Tensor(self.embedding_size, self.attention_factor))
        self.attention_b = nn.Parameter(torch.Tensor(self.attention_factor))
        self.projection_h = nn.Parameter(torch.Tensor(self.attention_factor, 1))        # 隐层→注意力得分
        self.projection_p = nn.Parameter(torch.Tensor(self.embedding_size, 1))          # 最终投影

        # 参数初始化
        for tensor in [self.attention_W, self.projection_h, self.projection_p]:
            nn.init.xavier_normal_(tensor,)
        for tensor in [self.attention_b]:
            nn.init.zeros_(tensor,)                                                     # 偏置初始化为0

        # 定义层
        self.drop = nn.Dropout(self.drop_rate)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        # 生成所有特征交叉对 (r, c)
        embeds_vec_list = inputs                                                        # 输入为多个嵌入向量组成的列表
        row, col = [], []

        for r, c in itertools.combinations(embeds_vec_list, 2):                         # 两两组合
            row.append(r)                                                               # 添加第一个特征嵌入     
            col.append(c)                                                               # 添加第二个特征嵌入

        # 拼接交互对并计算内积
        p = torch.cat(row, dim=1)                                                       # 形状：(batch_size, num_pairs, embedding_size)
        q = torch.cat(col, dim=1)
        inner_product = p * q

        bi_interaction = inner_product

        #其中nn.ReLU作为一个层结构，必须添加到nn.Module容器中才能使用，
        # 而F.ReLU则作为一个函数调用，看上去作为一个函数调用更方便更简洁。具
        # 体使用哪种方式，取决于编程风格。在PyTorch中,nn.X都有对应的函数版本F.X，
        # 但是并不是所有的F.X均可以用于forward或其它代码段中，因为当网络模型训练完毕时，
        # 在存储model时，在forward中的F.X函数中的参数是无法保存的。
        # 也就是说，在forward中，使用的F.X函数一般均没有状态参数，比如F.ReLU，F.avg_pool2d等，
        # 均没有参数，它们可以用在任何代码片段中。

        # 注意力得分计算
        attention_temp = self.relu(torch.tensordot(
            bi_interaction, self.attention_W, dims=([-1], [0])) + self.attention_b)     # 形状：(batch_size, num_pairs, attention_factor)
        normalized_att_score = self.softmax(torch.tensordot(
            attention_temp, self.projection_h, dims=([-1], [0])))                       # 形状：(batch_size, num_pairs, 1)

        # 加权交叉项并投影输出
        attention_output = torch.sum(normalized_att_score * bi_interaction, dim=1)      # 形状：(batch_size, embedding_size)
        attention_output = self.drop(attention_output)                                  # 应用Dropout
        afm_out = torch.tensordot(attention_output, self.projection_p, dims=([-1], [0]))# 形状：(batch_size, 1)

        #print(afm_out)
        return  afm_out

class AFM(nn.Module):
    def __init__(self, feat_size, embedding_size, dnn_feature_columns,
                 use_attention=True, attention_factor=8, l2_reg=0.00001, drop_rate=0.9):#use_attention,TrueAFM;FalseFM
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
        self.use_attention = True
        if self.use_attention:
            self.fm = AFMLayer(embedding_size, attention_factor, l2_reg, drop_rate=0.9)
        else:
            self.fm = FM()

        # 线性部分（全连接层）
        dnn_hidden_units = [len(feat_size), 1]                      # 输入维度为特征数，输出1维
        self.linear = nn.ModuleList([
            nn.Linear(len(feat_size), 1)
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

if __name__ == '__main__':

    torch.cuda.empty_cache()  # 训练开始前执行
    batch_size = 2048
    lr = 1e-4
    wd = 0
    epoches = 100
    seed = 2022
    embedding_size = 8
    device = 'cuda:0' # device = 'cuda:0'

    # 定义特征列
    sparse_feature = ['C' + str(i) for i in range(1, 27)]               # 26个稀疏特征（类别型）
    dense_feature = ['I' + str(i) for i in range(1, 14)]                # 13个密集特征（数值型）
    col_names = ['label'] + dense_feature + sparse_feature
    data = pd.read_csv('/root/yesy/MAGA/pytorch/train_sam.txt', names=col_names, sep='\t')  #100w数据集来自https://www.kaggle.com/datasets/shengyaoye/maga-paperdatast/data

    data[sparse_feature] = data[sparse_feature].fillna('-1', )          # 稀疏特征填充-1
    data[dense_feature] = data[dense_feature].fillna('0',)              # 密集特征填充0
    target = ['label']

    feat_sizes = {}     
    feat_sizes_dense = {feat:1 for feat in dense_feature}                               # 密集特征：维度为1
    feat_sizes_sparse = {feat:len(data[feat].unique()) for feat in sparse_feature}      # 稀疏特征：记录类别数
    feat_sizes.update(feat_sizes_dense)                 # 形如：{'C1': 2, 'C2': 3}
    feat_sizes.update(feat_sizes_sparse)

    # 对稀疏特征进行标签编码
    for feat in sparse_feature:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 对密集特征归一化到[0,1]
    nms = MinMaxScaler(feature_range=(0, 1))
    data[dense_feature] = nms.fit_transform(data[dense_feature])

    fixlen_feature_columns = [(feat,'sparse') for feat in sparse_feature ]  + [(feat,'dense') for feat in dense_feature]
    dnn_feature_columns = fixlen_feature_columns                
    # linear_feature_columns = fixlen_feature_columns
    # [
    # ('C1', 'sparse'), ('C2', 'sparse'), ...,  # 所有稀疏特征
    # ('I1', 'dense'), ('I2', 'dense'), ...      # 所有密集特征
    # ]

    train, test = train_test_split(data, test_size=0.2, random_state=seed)

    device = 'cuda:0'   # device = 'cuda:0'
    model = AFM(feat_sizes, embedding_size, dnn_feature_columns, use_attention=True).to(device)

    train_label = pd.DataFrame(train['label'])
    train = train.drop(columns=['label'])
    train_tensor_data = TensorDataset(torch.from_numpy(np.array(train)), torch.from_numpy(np.array(train_label)))   #特征和标签转换为PyTorch张量并封装为Dataset
    train_loader = DataLoader(train_tensor_data, shuffle=True, batch_size=batch_size)   # 将数据集 train_tensor_data 转换为可迭代的DataLoader，支持按批次加载数据,shuffle打乱epoch

    test_label = pd.DataFrame(test['label'])
    test = test.drop(columns=['label'])
    test_tensor_data = TensorDataset(torch.from_numpy(np.array(test)), torch.from_numpy(np.array(test_label)))
    test_loader = DataLoader(test_tensor_data, batch_size=batch_size)

    loss_func = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epoches):
        total_loss_epoch = 0.0
        total_tmp = 0
        model.train()
        for index, (x, y) in enumerate(train_loader):
            x, y = x.to(device).float(), y.to(device).float()
            y_hat = model(x)

            optimizer.zero_grad()
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            total_tmp += 1
        auc = get_auc(test_loader, model)
        print('epoch/epoches: {}/{}, train loss: {:.3f}, test auc: {:.3f}'.format(epoch, epoches,
                                                                                  total_loss_epoch / total_tmp, auc))


