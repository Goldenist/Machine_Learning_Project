## 配置环境说明

1. 我们小组的电脑配置的环境与 Rechorus 官方要求的环境有些许不一样，主要体现在 torch版本，官方版本为 1.12.1，我们小组使用此版本后发现会出现与电脑的 cuda 版本的兼容问题，故使用的 torch 版本为 2.2.1，其余环境与官方 requirements.txt一致。
2. 官方 BaseModel.py  文件在运行时会出现报错，将 146 行改为：if stack_val.dtype == object: 
3. 官方 utils.py 文件在运行时会出现报错，将 65 行改为：if type(m) is float or type(m) is np.float32 or type(m) is np.float64:

## 添加的代码文件

### 1. models/general 目录

- JMP\_GCF\_K\_1.py : 这个代码将 LightGCNBase 中的 build\_adjmat() 函数生成邻接矩阵 $\hat{A}_1$(由公式（1）给出），即粒度 k = 1。代码将调用 BaseRunner 和 BaseReader。
- JMP\_GCF\_K\_2.py : 同理，将 build\_adjmat() 函数生成 $\hat{A}_2$，即粒度 k = 2。代码将调用 BaseRunner 以及 BaseReader。
- JMP\_GCF\_012.py : build\_adjmat() 函数生成由三个不同粒度的邻接矩阵组成的列表，元素分别为 $\hat{A}_0$，$\hat{A}_1$，$\hat{A}_2$。将编码函数由 LightEncoder 修改为 JMP\_GCFEncoder，其中的 forward() 函数会对每一个粒度的邻接矩阵生成对应的嵌入矩阵。代码将调用 JMP_以及 JMP\_GCF\_kRunner。
- JMP\_GCF.py ：代码内容与 JMP\_GCF\_012.py 类似，但是调用的是 JMP_GCFRunner。

### 2. helps 目录

- JMP\_GCF\_kRunner.py : 这个代码由 BaseRunner.py 修改得到，主要修改部分为 fit() 函数。通过调用 JMP\_GCF\_012 模型的 forward() 函数，得到三种粒度对应的嵌入矩阵，存储在 out\_fit 列表中，对这三个嵌入矩阵进行遍历和反向传播。
- JMP\_GCFRunner.py : 这个代码在 JMP\_GCF\_KRunner 的基础上加入了多阶段堆叠训练。fit() 函数按照 epoch 划分为三个训练阶段，分别优化 $L_1$，$L_2$，$L_3$。通过调用 JMP\_GCF 模型的 forward() 函数，得到三种粒度的嵌入矩阵列表 out\_fit，根据当前阶段选择不同的嵌入矩阵进行训练和反向传播。

## 运行模型

1. 进入 Rechorus 目录

```sh
cd ReChorus
```

2. 加载数据集与运行模型

BPRMF 模型：

```sh
python .\src\main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset Grocery_and_Gourmet_Food --gpu "0" --num_workers 0
python .\src\main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MIND_Large\MINDTOPK --gpu "0" --num_workers 0
python .\src\main.py --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MovieLens_1M\ML_1MTOPK --gpu "0" --num_workers 0
```

LightGCN 模型：

```sh
python .\src\main.py --model_name LightGCN --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset Grocery_and_Gourmet_Food --gpu "0" --num_workers 0
python .\src\main.py --model_name LightGCN --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MIND_Large\MINDTOPK --gpu "0" --num_workers 0
python .\src\main.py --model_name LightGCN --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MovieLens_1M\ML_1MTOPK --gpu "0" --num_workers 0
```

JMP_GCF_k_1 模型：

```sh
python .\src\main.py --model_name JMP_GCF_k_1 --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset Grocery_and_Gourmet_Food --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF_k_1 --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MIND_Large\MINDTOPK --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF_k_1 --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MovieLens_1M\ML_1MTOPK --gpu "0" --num_workers 0
```

JMP_GCF_k_2 模型：

```sh
python .\src\main.py --model_name JMP_GCF_k_2 --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset Grocery_and_Gourmet_Food --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF_k_2 --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MIND_Large\MINDTOPK --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF_k_2 --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MovieLens_1M\ML_1MTOPK --gpu "0" --num_workers 0
```

JMP_GCF_k_012 模型：

```sh
python .\src\main.py --model_name JMP_GCF_k_012 --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset Grocery_and_Gourmet_Food --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF_k_012 --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MIND_Large\MINDTOPK --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF_k_012 --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MovieLens_1M\ML_1MTOPK --gpu "0" --num_workers 0
```

JMP_GCF 模型：

```sh
python .\src\main.py --model_name JMP_GCF --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset Grocery_and_Gourmet_Food --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MIND_Large\MINDTOPK --gpu "0" --num_workers 0
python .\src\main.py --model_name JMP_GCF --emb_size 64 --lr 1e-3 --l2 1e-06  --dataset MovieLens_1M\ML_1MTOPK --gpu "0" --num_workers 0
```
