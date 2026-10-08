import optimizer
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据集
df_all = pd.read_csv("train.csv")
# 1.划分训练子集、测试子集
train_raw, test_raw = train_test_split(
    df_all,
    test_size=0.2,        # 测试集占全部数据20%，训练集80%
    random_state=42,      # 固定随机种子，每次运行划分结果不变，实验可复现
    stratify=df_all["Survived"]   # 分层采样！保持训练、测试集生还/死亡比例和原数据集一致
)
# 2.只在训练子集计算统计参数
Age_mean_train = train_raw["Age"].mean()
Embarked_mode_train = train_raw["Embarked"].mode()[0]

#数据预处理
def preprocess_data(df, Age_mean, Embarked_mode):
    df = df.copy()
    # 第一步：直接删除无用列
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    # 缺失值填充
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # Sex编码
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    # Embarked独热
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    # 构造新特征
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df

# 执行预处理
train_processed = preprocess_data(train_raw, Age_mean_train, Embarked_mode_train)
test_processed = preprocess_data(test_raw, Age_mean_train, Embarked_mode_train)
# 列对齐：保证训练集、测试集特征列完全一致，防止维度不匹配报错
test_processed = test_processed.reindex(columns=train_processed.columns, fill_value=0)   #缺少的列补齐之后用0填满

#数值特征标准化
num_feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]
scaler = StandardScaler()
# fit仅在训练集执行，学习均值方差
train_processed[num_feature_cols] = scaler.fit_transform(train_processed[num_feature_cols])
# 测试集只transform，不重新拟合参数
test_processed[num_feature_cols] = scaler.transform(test_processed[num_feature_cols])

#分离特征x和标签y，转化为pytorch张量
X_train = train_processed.drop("Survived", axis=1).astype(float).to_numpy()
y_train = train_processed["Survived"].astype(float).to_numpy()

X_test = test_processed.drop("Survived", axis=1).astype(float).to_numpy()
y_test = test_processed["Survived"].astype(float).to_numpy()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

#自定义pytorch dataset
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        #保存传入的特征值X，y
        self.X = X
        self.y = y
    #返回样本的总数量
    def __len__(self):
        return len(self.X)
    #根据索引取出单条样本
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
train_dataset = TitanicDataset(X_train_tensor, y_train_tensor)
test_dataset = TitanicDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#自主二分类模型
class TitanicModel(torch.nn.Module):
    def __init__(self, input_dim):    #input_dim：输入特征的数量
        super().__init__()
        self.net = torch.nn.Sequential(    #自主搭建神经网络
            torch.nn.Linear(input_dim, 16),    #第一个隐藏层，把一个样本的`input_dim`个特征，经过线性变换，变成 16 个新数值。
            torch.nn.ReLU(),        #激活函数。引入非线性。
            torch.nn.Dropout(0.2),       #随机失活，防止过拟合，训练的时候，随机把 20% 神经元输出置 0，不让模型过度记住训练集细节。
            torch.nn.Linear(16, 8),     #第二个隐藏层，Linear输入+RelU
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)  # 输出层，输出logits，不写Sigmoid！搭配BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1]      #拿到的就是神经网络第一层 Linear 需要的输入维度。
model = TitanicModel(input_dim)

#损失函数、优化器
criterion = torch.nn.BCEWithLogitsLoss()         #二分类交叉熵
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)    #加入L2正则化

#训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    if epoch < 20:
        lr = 0.001
    else:
        lr = 0.0005
        # 更新优化器的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()        ## 清空上一轮梯度
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()           #反向传播，自动求梯度
        optimizer.step()          # 根据梯度更新权重
        total_loss += loss.item()

    # 每10轮评估一次测试集
    if (epoch+1) % 10 == 0:
        model.eval()      #作用：关闭 Dropout、BN
        #创建空列表，用来保存所有测试集的预测结果、真实标签，等循环结束统一算指标
        all_preds = []
        all_labels = []
        with torch.no_grad():  #不计算梯度，不保存反向传播需要的信息，防止偷偷学习测试集
            for batch_X, batch_y in test_loader:
                logits = model(batch_X)
                # logits>0等价于sigmoid后>0.5
                pred = (logits > 0).float()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))


        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Accuracy: {acc:.4f}")