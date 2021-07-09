# GNN-Task-7
## 1. Introduction

在前面的学习中我们只接触了数据可全部储存于内存的数据集，这些数据集对应的数据集类在创建对象时就将所有数据都加载到内存。然而在一些应用场景中，**数据集规模超级大，我们很难有足够大的内存完全存下所有数据**。因此需要**一个按需加载样本到内存的数据集类**。在此上半节内容中，我们将学习为一个包含上千万个图样本的数据集构建一个数据集类。


## 2. Dataset Base Class
在PyG中，我们通过继承[`torch_geometric.data.Dataset`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset)基类来自定义一个按需加载样本到内存的数据集类。此基类与Torchvision的`Dataset `类的概念密切相关，这与第6节中介绍的`torch_geometric.data.InMemoryDataset`基类是一样的。

**继承`torch_geometric.data.InMemoryDataset`基类要实现的方法，继承此基类同样要实现，此外还需要实现以下方法**：

- `len()`：返回数据集中的样本的数量。
- `get()`：实现加载单个图的操作。注意：在内部，`__getitem__()`返回通过调用`get()`来获取`Data`对象，并根据`transform`参数对它们进行选择性转换。

下面让我们通过一个简化的例子看**继承`torch_geometric.data.Dataset`基类的规范**：

```python
import os.path as osp

import torch
from torch_geometric.data import Dataset, download_url

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # list of file names of raw partial graphs
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        # list of file names of processed partial graphs
        return ['data_1.pt', 'data_2.pt', ...]

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(url, self.raw_dir)
        ...

    def process(self):
        i = 0
        # process each raw file to get corresponding processed file
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

```

其中，每个`Data`对象在`process()`方法中单独被保存，并在`get()`中通过指定索引进行加载。




## 3. 合并小图组成大图

图可以有任意数量的节点和边，它不是规整的数据结构，因此对图数据封装成批的操作与对图像和序列等数据封装成批的操作不同。PyTorch Geometric中采用的将多个图封装成批的方式是，将小图作为连通组件（connected component）的形式合并，构建一个大图。于是小图的邻接矩阵存储在大图邻接矩阵的对角线上。大图的邻接矩阵、属性矩阵、预测目标矩阵分别为：
$$
\begin{split}\mathbf{A} = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_n \end{bmatrix}, \qquad \mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \vdots \\ \mathbf{X}_n \end{bmatrix}, \qquad \mathbf{Y} = \begin{bmatrix} \mathbf{Y}_1 \\ \vdots \\ \mathbf{Y}_n \end{bmatrix}.\end{split}
$$

**此方法有以下关键的优势**：

- 依靠消息传递方案的GNN运算不需要被修改，因为消息仍然不能在属于不同图的两个节点之间交换。

- 没有额外的计算或内存的开销。例如，这个批处理程序的工作完全不需要对节点或边缘特征进行任何填充。请注意，邻接矩阵没有额外的内存开销，因为它们是以稀疏的方式保存的，只保留非零项，即边。

通过[`torch_geometric.data.DataLoader`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.DataLoader)类，多个小图被封装成一个大图。[`torch_geometric.data.DataLoader`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.DataLoader)是PyTorch的`DataLoader`的子类，它覆盖了`collate()`函数，该函数定义了一列表的样本是如何封装成批的。因此，所有可以传递给PyTorch `DataLoader`的参数也可以传递给PyTorch Geometric的 `DataLoader`，例如，`num_workers`。


## 4. Pairs of Graphs
如果你想在一个[`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)对象中存储多个图，例如用于图对等应用，我们需要确保所有这些图的正确封装成批行为。例如，考虑将两个图，一个源图$G_s$和一个目标图$G_t$，存储在一个[`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)类中，即



```python
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data import DataLoader

class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        # source graph
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        
        # target graph
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)



edge_index_s = torch.tensor([
    [0, 0, 0, 0],
    [1, 2, 3, 4],
])
x_s = torch.randn(5, 16)  # 5 nodes.
edge_index_t = torch.tensor([
    [0, 0, 0],
    [1, 2, 3],
])
x_t = torch.randn(4, 16)  # 4 nodes.

data = PairData(edge_index_s, x_s, edge_index_t, x_t)
data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)

print(batch.edge_index_s)

print(batch.edge_index_t)

```

    Batch(edge_index_s=[2, 8], edge_index_t=[2, 6], x_s=[10, 16], x_t=[8, 16])
    tensor([[0, 0, 0, 0, 5, 5, 5, 5],
            [1, 2, 3, 4, 6, 7, 8, 9]])
    tensor([[0, 0, 0, 4, 4, 4],
            [1, 2, 3, 5, 6, 7]])


### 在新的维度上做拼接

有时，`Data`对象的属性需要在一个新的维度上做拼接（如经典的封装成批），例如，图级别属性或预测目标。具体来说，形状为`[num_features]`的属性列表应该被返回为`[num_examples, num_features]`，而不是`[num_examples * num_features]`。PyTorch Geometric通过在[`__cat_dim__()`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data.__cat_dim__)中返回一个[`None`](https://docs.python.org/3/library/constants.html#None)的连接维度来实现这一点。




```python
class MyData(Data):
     def __cat_dim__(self, key, item):
         if key == 'foo':
             return None
         else:
             return super().__cat_dim__(key, item)

edge_index = torch.tensor([
   [0, 1, 1, 2],
   [1, 0, 2, 1],
])
foo = torch.randn(16)

data = MyData(edge_index=edge_index, foo=foo)
data_list = [data, data]
loader = DataLoader(data_list, batch_size=2)
batch = next(iter(loader))

print(batch)
```

    WARNING:root:The number of nodes in your data object can only be inferred by its edge indices, and hence may result in unexpected batch-wise behavior, e.g., in case there exists isolated nodes. Please consider explicitly setting the number of nodes for this data object by assigning it to data.num_nodes.
    WARNING:root:The number of nodes in your data object can only be inferred by its edge indices, and hence may result in unexpected batch-wise behavior, e.g., in case there exists isolated nodes. Please consider explicitly setting the number of nodes for this data object by assigning it to data.num_nodes.
    WARNING:root:The number of nodes in your data object can only be inferred by its edge indices, and hence may result in unexpected batch-wise behavior, e.g., in case there exists isolated nodes. Please consider explicitly setting the number of nodes for this data object by assigning it to data.num_nodes.
    WARNING:root:The number of nodes in your data object can only be inferred by its edge indices, and hence may result in unexpected batch-wise behavior, e.g., in case there exists isolated nodes. Please consider explicitly setting the number of nodes for this data object by assigning it to data.num_nodes.


    Batch(batch=[6], edge_index=[2, 8], foo=[2, 16], ptr=[3])


## Advanced Mini-Batching for large-scale  graph

[**PCQM4M-LSC**](https://ogb.stanford.edu/kddcup2021/pcqm4m/)是一个分子图的量子特性回归数据集，它包含了3,803,453个图。

注意以下代码依赖于`ogb`包，通过`pip install ogb`命令可安装此包。`ogb`文档可见于[Get Started | Open Graph Benchmark (stanford.edu)](https://ogb.stanford.edu/docs/home/)。

在生成一个该数据集类的对象时，程序首先会检查指定的文件夹下是否存在`data.csv.gz`文件，如果不在，则会执行`download`方法，这一过程是在运行`super`类的`__init__`方法中发生的。然后程序继续执行`__init__`方法的剩余部分，读取`data.csv.gz`文件，获取存储图信息的`smiles`格式的字符串，以及回归预测的目标`homolumogap`。我们将由`smiles`格式的字符串转成图的过程在`get()`方法中实现，这样我们在生成一个`DataLoader`变量时，通过指定`num_workers`可以实现并行执行生成多个图。



```python
import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set


import torch
import torch.nn.functional as F


import torch
from torch import nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr) # 先将类别型边属性转换为边表征
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out




# GNN to generate node embedding
class GINNodeEmbedding(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self, num_layers, emb_dim, drop_ratio=0.5, JK="last", residual=False):
        """GIN Node Embedding Module"""

        super(GINNodeEmbedding, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        # add residual connection or not
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(GINConv(emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr

        # computing input node embedding
        h_list = [self.atom_encoder(x)]  # 先将类别型原子属性转化为原子表征
        for layer in range(self.num_layers):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation



class GINGraphPooling(nn.Module):

    def __init__(self, num_tasks=1, num_layers=5, emb_dim=300, residual=False, drop_ratio=0, JK="last", graph_pooling="sum"):
        """GIN Graph Pooling Module
        Args:
            num_tasks (int, optional): number of labels to be predicted. Defaults to 1 (控制了图表征的维度，dimension of graph representation).
            num_layers (int, optional): number of GINConv layers. Defaults to 5.
            emb_dim (int, optional): dimension of node embedding. Defaults to 300.
            residual (bool, optional): adding residual connection or not. Defaults to False.
            drop_ratio (float, optional): dropout rate. Defaults to 0.
            JK (str, optional): 可选的值为"last"和"sum"。选"last"，只取最后一层的结点的嵌入，选"sum"对各层的结点的嵌入求和。Defaults to "last".
            graph_pooling (str, optional): pooling method of node embedding. 可选的值为"sum"，"mean"，"max"，"attention"和"set2set"。 Defaults to "sum".

        Out:
            graph representation
        """
        super(GINGraphPooling, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = GINNodeEmbedding(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual)

        # Pooling function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1)))
        elif graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        output = self.graph_pred_linear(h_graph)

        if self.training:
            return output
        else:
            # At inference time, relu is applied to output to ensure positivity
            # 因为预测目标的取值范围就在 (0, 50] 内
            return torch.clamp(output, min=0, max=50)
        
```

## 6. Practice with GIN Regression Task using PCQM4M Dataset


```python
# %load_ext tensorboard
# %tensorboard --logdir=runs

import os
import os.path as osp

import pandas as pd
import torch
from ogb.utils.mol import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.lsc import PCQM4MEvaluator
from ogb.utils.url import download_url, extract_zip
from rdkit import RDLogger
from torch_geometric.data import Data, Dataset


import shutil

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

RDLogger.DisableLog('rdApp.*')

class MyPCQM4MDataset(Dataset):

    def __init__(self, root):
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip'
        super(MyPCQM4MDataset, self).__init__(root)

        filepath = osp.join(root, 'raw/data.csv.gz')
        data_df = pd.read_csv(filepath)
        self.smiles_list = data_df['smiles']
        self.homolumogap_list = data_df['homolumogap']

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.move(osp.join(self.root, 'pcqm4m_kddcup2021/raw/data.csv.gz'), osp.join(self.root, 'raw/data.csv.gz'))

    def len(self):
        return len(self.smiles_list)

    def get(self, idx):
        smiles, homolumogap = self.smiles_list[idx], self.homolumogap_list[idx]
        graph = smiles2graph(smiles)
        assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
        assert(len(graph['node_feat']) == graph['num_nodes'])

        x = torch.from_numpy(graph['node_feat']).to(torch.int64)
        edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
        edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
        y = torch.Tensor([homolumogap])
        num_nodes = int(graph['num_nodes'])
        data = Data(x, edge_index, edge_attr, y, num_nodes=num_nodes)
        return data

    # 获取数据集划分
    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'pcqm4m_kddcup2021/split_dict.pt')))
        return split_dict

    
def train(model, optimizer, loss_f, loader, scheduler ,device ):
    model.train()
    total_loss = 0.
    total_nodes = 0.
    for batch in tqdm(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch).to(device)
        loss = loss_f(out, batch.y)
        loss.backward()
        optimizer.step()
        
        nodes = batch.num_nodes
        total_loss += loss.item() * nodes
        total_nodes += nodes
        scheduler.step()
        
    return total_loss/total_nodes


def evaluate(model, loader, device, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        out = model(batch).view(-1, ).to(device)
        y_true.append(batch.y.view(out.shape).detach().cpu())
        y_pred.append(out.detach().cpu())
    #convert list to tensor
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    dic ={'y_true': y_true, "y_pred":y_pred}
    return evaluator.eval(dic)['mae']
        
        
        
if __name__ == "__main__":
    dataset = MyPCQM4MDataset('dataset2')
    split_data = dataset.get_idx_split()
    output_file =open( "./task-7-runs/logging.txt" ,"a")
    from torch_geometric.data import DataLoader
    from tqdm import tqdm
    train_loader = DataLoader(dataset[split_data['train']], batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset[split_data['valid']], batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset[split_data['test']], batch_size=256, shuffle=True, num_workers=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("Using device: ",device)
    
    # loss used to train model, MSE loss
    loss_f = torch.nn.MSELoss()
    
    # evaluator used to evaluate regression output and prediction, MAE (mean absolute error)
    evaluator = PCQM4MEvaluator()
    
    model = GINGraphPooling(num_tasks=1, num_layers=4, emb_dim=300, residual=False, drop_ratio=0.5, JK="last",
                 graph_pooling="sum").to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr =1e-3, weight_decay= 1e-3 )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # tensorboard writer
    writer = SummaryWriter(log_dir ="task-7-runs" )
    epochs = 20
    for e in range(epochs):
        train_loss = train(model, optimizer, loss_f, train_loader, scheduler,device )
        
        print(f"Epoch: {e}, Train MAE: {train_loss} ",file=output_file, flush=True)
        writer.add_scalar("Loss/train", train_loss, e)
        if e%2==0:
            val_mae = evaluate(model, val_loader, device, evaluator)
            writer.add_scalar("MAE/val", val_mae, e)
            print(f"Epoch: {e}, Valid MAE: {val_mae} ",file=output_file, flush=True)
            
        
        

```

      0%|          | 0/11896 [00:00<?, ?it/s]

    Using device:  cuda


    100%|██████████| 11896/11896 [06:44<00:00, 29.38it/s]
    100%|██████████| 1487/1487 [00:51<00:00, 28.91it/s]
    100%|██████████| 11896/11896 [06:40<00:00, 29.72it/s]
    100%|██████████| 11896/11896 [06:43<00:00, 29.49it/s]
    100%|██████████| 1487/1487 [00:50<00:00, 29.23it/s]
    100%|██████████| 11896/11896 [06:43<00:00, 29.51it/s]
    100%|██████████| 11896/11896 [06:42<00:00, 29.58it/s]
    100%|██████████| 1487/1487 [00:51<00:00, 28.71it/s]
    100%|██████████| 11896/11896 [06:40<00:00, 29.74it/s]
    100%|██████████| 11896/11896 [06:39<00:00, 29.81it/s]
    100%|██████████| 1487/1487 [00:50<00:00, 29.54it/s]
    100%|██████████| 11896/11896 [06:40<00:00, 29.68it/s]
    100%|██████████| 11896/11896 [06:41<00:00, 29.65it/s]
    100%|██████████| 1487/1487 [00:50<00:00, 29.25it/s]
    100%|██████████| 11896/11896 [06:40<00:00, 29.67it/s]
    100%|██████████| 11896/11896 [06:46<00:00, 29.24it/s]
    100%|██████████| 1487/1487 [00:49<00:00, 29.96it/s]
    100%|██████████| 11896/11896 [06:38<00:00, 29.83it/s]
    100%|██████████| 11896/11896 [06:42<00:00, 29.57it/s]
    100%|██████████| 1487/1487 [00:49<00:00, 29.92it/s]
    100%|██████████| 11896/11896 [06:41<00:00, 29.63it/s]
    100%|██████████| 11896/11896 [06:47<00:00, 29.22it/s]
    100%|██████████| 1487/1487 [00:49<00:00, 29.78it/s]
    100%|██████████| 11896/11896 [06:39<00:00, 29.76it/s]
    100%|██████████| 11896/11896 [06:42<00:00, 29.53it/s]
    100%|██████████| 1487/1487 [00:49<00:00, 30.23it/s]
    100%|██████████| 11896/11896 [06:42<00:00, 29.56it/s]
    100%|██████████| 11896/11896 [06:42<00:00, 29.55it/s]
    100%|██████████| 1487/1487 [00:49<00:00, 29.90it/s]
    100%|██████████| 11896/11896 [06:35<00:00, 30.08it/s]


### Result
1. Training MSE Loss
<img src=train_loss.png>

2. MAE (Mean Absolute Error) for evaluation 
<img src=val_mae.png>



```python

```

## 7. Reference
[1] OGB document: https://ogb.stanford.edu/kddcup2021/pcqm4m/#evaluator

[2] `Dataset`类官方文档： [`torch_geometric.data.Dataset`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset)

[3] 将图样本封装成批（BATCHING）：[ADVANCED MINI-BATCHING](https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html)

[4] 分子图的量子特性回归数据集：[PCQM4M-LSC](https://ogb.stanford.edu/kddcup2021/pcqm4m/)

[5] [Get Started | Open Graph Benchmark (stanford.edu)](https://ogb.stanford.edu/docs/home/)

[6] Datawhale: https://github.com/datawhalechina/team-learning-nlp/blob/master/GNN/Markdown%E7%89%88%E6%9C%AC/9-1-%E6%8C%89%E9%9C%80%E8%8E%B7%E5%8F%96%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86%E7%B1%BB%E7%9A%84%E5%88%9B%E5%BB%BA.md

https://github.com/datawhalechina/team-learning-nlp/blob/master/GNN/Markdown%E7%89%88%E6%9C%AC/9-2-%E5%9B%BE%E9%A2%84%E6%B5%8B%E4%BB%BB%E5%8A%A1%E5%AE%9E%E8%B7%B5.md




```python

```




```python

```
