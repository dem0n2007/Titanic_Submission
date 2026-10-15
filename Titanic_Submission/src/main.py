# ========== 导入库（删掉错误的 import optimizer） ==========
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------- 读取数据集 ----------------------
df_all = pd.read_csv("train.csv")
# 1.划分训练子集、测试子集
train_raw, test_raw = train_test_split(
    df_all,
    test_size=0.2,        # 测试集占全部数据20%，训练集80%
    random_state=42,      # 固定随机种子，每次运行划分结果不变，实验可复现
    stratify=df_all["Survived"]   # 分层采样！保持训练、测试集生还/死亡比例和原数据集一致
)
# 2.只在训练子集计算统计参数（用于后续填充缺失值，避免数据泄露）
Age_mean_train = train_raw["Age"].mean()
Embarked_mode_train = train_raw["Embarked"].mode()[0]

# ---------------------- 数据预处理函数 ----------------------
def preprocess_data(df, Age_mean, Embarked_mode):
    df = df.copy()
    # 第一步：直接删除无用列
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    # 缺失值填充：注意！原代码这里有bug，应该用传入的Age_mean而不是df["Age"].mean()，已修正
    df["Age"] = df["Age"].fillna(Age_mean)
    df["Embarked"] = df["Embarked"].fillna(Embarked_mode)
    # Sex编码
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    # Embarked独热编码 drop_first=True，去掉一列防止多重共线性
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    # 构造新特征：家庭成员总数 = 兄弟姐妹+父母子女 +自己
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df

# 执行预处理
train_processed = preprocess_data(train_raw, Age_mean_train, Embarked_mode_train)
test_processed = preprocess_data(test_raw, Age_mean_train, Embarked_mode_train)
# 列对齐：保证训练集、测试集特征列完全一致，防止维度不匹配报错
test_processed = test_processed.reindex(columns=train_processed.columns, fill_value=0)

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
            torch.nn.Linear(input_dim, 16),    #第一个隐藏层，把一个样本的input_dim个特征，经过线性变换，变成 16 个新数值。
            torch.nn.ReLU(),        #激活函数。引入非线性。
            torch.nn.Dropout(0.2),       #随机失活，防止过拟合，训练的时候，随机把 20% 神经元输出置 0，不让模型过度记住训练集细节。
            torch.nn.Linear(16, 8),     #第二个隐藏层
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)  # 输出层输出logits，不写Sigmoid！搭配BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)

input_dim = X_train.shape[1]      #拿到神经网络第一层 Linear 需要的输入维度。
model = TitanicModel(input_dim)

#损失函数、优化器
criterion = torch.nn.BCEWithLogitsLoss()         #二分类交叉熵
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)    #加入L2正则化

#训练模型
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
epochs = 50
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    # 学习率衰减
    if epoch < 20:
        lr = 0.001
    else:
        lr = 0.0005
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    all_train_preds = []    #存放训练集全部样本的预测结果（预测标签）
    all_train_labels = []    #存放训练集全部样本真实标签（真实 y 值）
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()      # 清空上一轮梯度
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()         #反向传播，自动求梯度
        optimizer.step()       # 根据梯度更新权重
        total_train_loss += loss.item()
        # 记录训练集预测，计算训练集准确率
        pred = (logits > 0).float()
        all_train_preds.extend(pred.cpu().numpy())
        all_train_labels.extend(batch_y.cpu().numpy())
    avg_train_loss = total_train_loss / len(train_loader)
    train_acc = np.mean(np.array(all_train_preds) == np.array(all_train_labels))

    #测试集评估：每10轮打印一次
    if (epoch+1) % 10 == 0:
        model.eval()      #作用：关闭 Dropout
        total_val_loss = 0
        all_val_preds = []
        all_val_labels = []
        with torch.no_grad():        #不计算梯度、不保存计算图
            for batch_X, batch_y in test_loader:
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item()
                pred = (logits > 0).float()
                all_val_preds.extend(pred.cpu().numpy())
                all_val_labels.extend(batch_y.cpu().numpy())
        avg_val_loss = total_val_loss / len(test_loader)
        val_acc = np.mean(np.array(all_val_preds) == np.array(all_val_labels))

        # 保存到列表
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss:{avg_train_loss:.4f}, Val Loss:{avg_val_loss:.4f}")
        print(f"Train Accuracy:{train_acc:.4f}, Val Accuracy:{val_acc:.4f}")
        model.train()   #恢复训练模式，Dropout生效


# --------------------------【训练完成，保存模型权重】--------------------------
# 只保存模型参数（推荐方式，占用空间小）
torch.save(model.state_dict(), "titanic_model.pth")
print("\n✅ 模型权重已保存为 titanic_model.pth")

# 保存训练集特征列名，预测新样本时用来对齐特征顺序！非常关键
feature_columns = train_processed.drop("Survived", axis=1).columns.tolist()
np.save("feature_cols.npy", feature_columns)
print("✅ 特征列名已保存 feature_cols.npy")

# --------------------------【绘图】--------------------------
# 横坐标，5个评估点
x_epoch = [10,20,30,40,50]

plt.figure(figsize=(12,5))

# 损失曲线
plt.subplot(1,2,1)
plt.plot(x_epoch, train_loss_list, 'o-', label="Train Loss", color="#1f77b4")
plt.plot(x_epoch, val_loss_list, 'o-', label="Val Loss", color="#ff7f0e")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Trend: Train vs Validation")
plt.legend()
plt.grid(alpha=0.3)

# 准确率曲线
plt.subplot(1,2,2)
plt.plot(x_epoch, train_acc_list, 'o-', label="Train Acc", color="#2ca02c")
plt.plot(x_epoch, val_acc_list, 'o-', label="Val Acc", color="#d62728")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve: Train vs Validation")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --------------------------【加载模型 + 新乘客预测函数】--------------------------
def predict_new_passenger(new_passenger_dict):
    """
    new_passenger_dict: 字典，新乘客原始信息（和csv字段一致）
    返回：是否幸存（1幸存 /0遇难） + 幸存概率
    """
    # 1. 构造单行DataFrame
    df_new = pd.DataFrame([new_passenger_dict])
    # 2. 使用训练集统计参数做预处理，和训练流程完全保持一致
    df_new = preprocess_data(df_new, Age_mean_train, Embarked_mode_train)
    # 3. 对齐特征列顺序，缺少的列填充0
    df_new = df_new.reindex(columns=feature_columns, fill_value=0)
    # 4. 数值特征标准化！必须用之前训练好的scaler，不能重新fit
    df_new[num_feature_cols] = scaler.transform(df_new[num_feature_cols])
    # 5. 转为torch tensor
    X_new = torch.tensor(df_new.to_numpy(), dtype=torch.float32)

    # 6. 加载模型，评估模式预测
    model = TitanicModel(input_dim=len(feature_columns))
    model.load_state_dict(torch.load("titanic_model.pth"))
    model.eval()

    with torch.no_grad():
        logit = model(X_new)
        prob = torch.sigmoid(logit)   # logit转成0~1概率
        pred_label = (logit > 0).float()

    return int(pred_label.item()), round(prob.item(),4)

# --------------------------【测试：举例新乘客】--------------------------
# 示例1：一等舱女性，30岁，没有家人，南安普顿登船
passenger1 = {
    "Pclass": 1,
    "Sex": "female",
    "Age": 30,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 80,
    "Embarked": "S",
    "PassengerId": 999,
    "Name": "Test Lady",
    "Ticket": "A123",
    "Cabin": "C12"
}

# 示例2：三等舱男性，25岁，孤身
passenger2 = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 8,
    "Embarked": "S",
    "PassengerId": 1000,
    "Name": "Test Man",
    "Ticket": "B456",
    "Cabin": np.nan
}

# 执行预测
res1, prob1 = predict_new_passenger(passenger1)
res2, prob2 = predict_new_passenger(passenger2)

print("\n===== 新乘客预测结果 =====")
print(f"乘客1：{'幸存' if res1==1 else '遇难'}，幸存概率：{prob1}")
print(f"乘客2：{'幸存' if res2==1 else '遇难'}，幸存概率：{prob2}")
