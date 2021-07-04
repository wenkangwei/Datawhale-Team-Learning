```python

```

# GNN-Task-6-GIN
## Introduction
在前5篇博客里面，考虑的都是node representation节点的表征，并且每个节点都有自己的特征向量代表这个节点对象的信息。而这次我们考虑的是图的表征，而不是节点的表征。我们要用GNN来学习图的特征(包括节点信息和图的结构)，要如何利用节点的特征来计算图的特征。这里我们首先考虑同构图的特征表达和学习。既然是同构图，那么就是涉及一下几个问题：
+ 什么是同构图(isomorphisc graph)?
+ 如何判断两个图是否同构？
+ 如何衡量两个图的相似度？
+ 怎么通过GIN(Graph Isomorphism Network)计算Graph Embedding?

对于这几个问题,这篇博客会先回答什么是通构图，然后提及过去测试同构图和图的相似度的方法(Weisfeiler-Lehman Test and subtree kernels)，之后会回答怎么用图神经网络GIN对graph representation进行学习。

## What is Isomorphic Graph
首先什么是同构图？根据WolframMathWorld的解释: `Two graphs which contain the same number of graph vertices connected in the same way are said to be isomorphic.` 如果两个图是同构就会满足以下特点(其中第2~4点意味两个图的连接方式一样):
1. 两个图的node个数相同
2. 两个图的edge边数相同
3. 两个图的node的degree序列都是一样的(两个图的node degree一一对应)
4. 如果一个图有环，那么总能在另外一个graph找到长度相同的对应的环

以下图为例，下面两个图里面的节点数(4个)和边的连接方式都是一样的(4条边，1个环)，所以下面两个图是同构图
<img src=isomorphic-graph.png>

**那么问题来了，如果图十分复杂没法用眼来观察时，怎么知道他们是同构图呢?** 这就先涉及到一个叫 WL test 和 graph kernel( Weisfeiler-Lehman Test and graph kernel)的测量方法。下面一节解释这个方法

## Weisfeiler-Lehman Test and subtree kernel
Weisfeiler-Lehman Test 的paper:https://www.jmlr.org/papers/volume12/shervashidze11a/shervashidze11a.pdf

### WL test
WL test(Weisfeiler-Lehman Test) 是一个用来判断两个图是否同构的方法。 WL Test 的一维形式，类似于图神经网络中的邻接节点聚合。WL Test步骤:
1. 对两个图的节点进行label(一般可以把相同degree的node打上标志)
2. 对每个node进行neighbor节点的label收集，并且排序(排序是为了确保节点表示的单射性，去除顺序带来的影响) 
3. 对每个node的节点的序列通过hashing映射到新的label。将聚合的标签散列（hash）成新标签，该过程形式化为下方的公式，
4. 不断重复迭代地聚合节点及其邻接节点的标签

$$
L^{h}_{u} \leftarrow \operatorname{hash}\left(L^{h-1}_{u} + \sum_{v \in \mathcal{N}(U)} L^{h-1}_{v}\right)
$$
在上方的公式中，$L^{h}_{u}$表示节点$u$的第$h$次迭代的标签，第$0$次迭代的标签为节点原始标签。

在迭代过程中，发现两个图之间的节点的标签不同时，就可以确定这两个图是非同构的。需要注意的是节点标签可能的取值只能是有限个数。**WL测试不能保证对所有图都有效，特别是对于具有高度对称性的图，如链式图、完全图、环图和星图，它会判断错误。**  下面c从图a到图d是WL-test的流程图:
<img src=WL-test.png>

### WL subtree kernel
WL-test 虽然能判断两个图是否同构但是不能测量两个图的相似度，并且有时候对高度对称的图容易判断错误。这种情况下，我们可以用WL subtree kernel方法对两个图的相似度。它的步骤是
1. 先对两个图做迭代多次的WL-test label，即按照上面一小节的图a~d不断对图进行relabel
2. 把两个图的多次迭代生成的所有label进行个数的统计，并将他们拼接成一个向量
3. 把两个图的向量做inner product内积进行相似度计算，从而得到kernel值
4. 这个kernel值越大代表两个图越相似，但是不一定是同构图。下图是subtree kernel的计算例子
$\phi$代表图的feature vector， $k_ {wlsubtree}$代表kernel值或两个图的相似度
<img src=WL-subtree-kernel.png>

## Graph Isomorphism Network（GIN）
paper: https://arxiv.org/pdf/1810.00826.pdf
### Motivaition
根据原文, GNN的设计目前都是根据以往经验以及启发式方法和通过实验试错得到的，但是对GNN的表达能力缺乏了研究分析也缺少理论证明。而这片文章主要描述GNN的表达能力以及对其分析，另外也通过把它和WL-test 结合设计了简单的同构图网络(GIN)。 这篇文章的**重点贡献**在
1. 理论上说明了GNN和WL-test 在图结构的识别上有同样的能力
2. 搭建了neighbor aggregation 和 readout 函数使GNN有和WL-test相同的识别网络结构的能力
3. 分析了那些GNN （像GCN，GraphSAGE等）识别不好的图结构
4. 设计了和WL-test有同样的网络结构识别能力的GIN网络

### How does GIN work
+ Node Representation learning
+ Graph Representation
#### 简单的Readout 函数
这篇paper也提出了readout函数用于把GNN最后一层的node representation通过把所有node的信息进行聚合从而得到一个graph的embedding。而这个readout函数一般是简单的排列不变性，和节点的特征排序无关, 比如summation， graph-level pooling。
$$
\mathbf{h_ {G}} = \textbf{READOUT}(\{\mathbf{h^{k}_ {v} | v \in \mathbf{G} }\})
$$

#### 图同构网络GIN的构建

能实现判断图同构性的图神经网络需要满足，只在两个节点自身标签一样且它们的邻接节点一样时，图神经网络将这两个节点映射到相同的表征，即映射是单射性的。**可重复集合（Multisets）指的是元素可重复的集合，元素在集合中没有顺序关系。** **一个节点的所有邻接节点是一个可重复集合，一个节点可以有重复的邻接节点，邻接节点没有顺序关系。**因此GIN模型中生成节点表征的方法遵循WL Test算法更新节点标签的过程。

在GIN里面node representation的update公式是
$$
h_ {v}^{k} = \text{MLP}^{k}((1+ \epsilon^{k})h_ {v}^{(k-1)} + \sum_ {u \in \mathbf{N}(v)} h_ {u}^{(k-1)})
$$

**在生成节点的表征后仍需要执行图池化（或称为图读出）操作得到图表征**，最简单的图读出操作是做求和。由于每一层的节点表征都可能是重要的，因此在图同构网络中，不同层的节点表征在求和后被拼接，其数学定义如下，
$$
h_{G} = \text{CONCAT}(\text{READOUT}\left(\{h_{v}^{(k)}|v\in G\}\right)|k=0,1,\cdots, K)
$$
**采用拼接而不是相加的原因在于不同层节点的表征属于不同的特征空间。**未做严格的证明，这样得到的图的表示与WL Subtree Kernel得到的图的表征是等价的。



```python

```

## 4. Coding
**以下代码参考Stanford SNAP的 molecular的例子。** 从[官方文档](https://github.com/datawhalechina/team-learning-nlp/blob/6f8cd26d2cff4f791bab7d553b06ed652b75b854/GNN/Markdown%E7%89%88%E6%9C%AC/codes/gin_regression/gin_node.py#L8)，我们可以找到一下的GIN的node embedding, graph representation以及GINConv layer的代码. 
这里先以stanford的Open Graph Benchmark (OGB)  的原子结构图的为例。
OBG library的AtomEncoder和BondEncoder为例。


由于在当前的例子中，节点（原子）和边（化学键）的属性都为离散值，它们属于不同的空间，无法直接将它们融合在一起。通过嵌入（Embedding），**我们可以将节点属性和边属性分别映射到一个新的空间，在这个新的空间中，我们就可以对节点和边进行信息融合**。在`GINConv`中，`message()`函数中的`x_j + edge_attr` 操作执行了节点信息和边信息的融合。

接下来，我们通过下方的代码中的`AtomEncoder`类，来分析将节点属性映射到一个新的空间是如何实现的：

- `full_atom_feature_dims` 是一个链表`list`，存储了节点属性向量每一维可能取值的数量，即`X[i]` 可能的取值一共有`full_atom_feature_dims[i]`种情况，`X`为节点属性；
- 节点属性有多少维，那么就需要有多少个嵌入函数，通过调用`torch.nn.Embedding(dim, emb_dim)`可以实例化一个嵌入函数；
- `torch.nn.Embedding(dim, emb_dim)`，第一个参数`dim`为被嵌入数据可能取值的数量，第一个参数`emb_dim`为要映射到的空间的维度。得到的嵌入函数接受一个大于`0`小于`dim`的数，输出一个维度为`emb_dim`的向量。嵌入函数也包含可训练参数，通过对神经网络的训练，嵌入函数的输出值能够表达不同输入值之间的相似性。
- 在`forward()`函数中，我们对不同属性值得到的不同嵌入向量进行了相加操作，实现了**将节点的的不同属性融合在一起**。

`BondEncoder`类与`AtomEncoder`类是类似的。




```python
import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   

```

#### 4.2 GIN 代码
下面的GIN的架构根据ogb里面的molecular的例子的AtomEncoder, BondEncoder先对node， edge进行embedding得到节点和边的特征向量。之后基于这些特征向量的encoder搭建了GINConv卷积layer， 基于GINConv layer而搭建的GINNodeEmbedding节点信息更新的网络,以及基于GINNodeEmbedding的节点信息而计算的GINGraphEmbedding的图向量表达。 这些模块可以参考stanford的ogb的mol的源码，**不过下面的代码把源码的class的名字更改了一下**
+ AtomEncoder and BondEncoder for mol example: https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/mol_encoder.py

+ GINConv layer: https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/conv.py

+  基于GINConv layer而搭建的GINNodeEmbedding： https://github.com/snap-stanford/ogb/blob/955f22515dc0e6a8231c0118f3c8760aa26c45a6/examples/graphproppred/mol/conv.py#L68

+ 基于GINNodeEmbedding 而搭建的GINGraphPooling网络，输出是graph embedding： https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/mol/gnn.py


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


```python


import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


class Args():
    def __init__(self):
        self.device = 0
        self.gnn='gin'
        self.drop_ratio = 0.5
        self.num_layers=5
        self.emb_dim = 300
        self.batch_size = 32
        self.epochs = 100
        self.num_workers=0
        self.dataset= "ogbg-molhiv"
        self.feature="full"
        self.filename=""

        
def get_terminal_args():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()
    return args
def main():
    # Training settings
    ## if obtain settings from terminal
    #args = get_terminal_args()
    args = Args()
    args.epochs = 5
    

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    if args.gnn == 'gin':
        model = GINGraphPooling( num_tasks = dataset.num_tasks, num_layers = args.num_layers, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio,).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    torch.manual_seed(2021)
    main()
```

    Iteration:   0%|          | 4/1029 [00:00<00:29, 34.25it/s]

    =====Epoch 1
    Training...


    Iteration: 100%|██████████| 1029/1029 [00:25<00:00, 40.35it/s]
    Iteration:   2%|▏         | 17/1029 [00:00<00:06, 160.89it/s]

    Evaluating...


    Iteration: 100%|██████████| 1029/1029 [00:06<00:00, 164.41it/s]
    Iteration: 100%|██████████| 129/129 [00:00<00:00, 131.22it/s]
    Iteration: 100%|██████████| 129/129 [00:01<00:00, 95.27it/s]
    Iteration:   0%|          | 4/1029 [00:00<00:27, 37.52it/s]

    {'Train': {'rocauc': 0.6604943642908611}, 'Validation': {'rocauc': 0.682172251616696}, 'Test': {'rocauc': 0.6643677938932772}}
    =====Epoch 2
    Training...


    Iteration: 100%|██████████| 1029/1029 [00:23<00:00, 43.43it/s]
    Iteration:   1%|▏         | 14/1029 [00:00<00:07, 136.74it/s]

    Evaluating...


    Iteration: 100%|██████████| 1029/1029 [00:06<00:00, 163.67it/s]
    Iteration: 100%|██████████| 129/129 [00:00<00:00, 165.12it/s]
    Iteration: 100%|██████████| 129/129 [00:00<00:00, 165.38it/s]
    Iteration:   0%|          | 4/1029 [00:00<00:30, 34.14it/s]

    {'Train': {'rocauc': 0.4996526571726294}, 'Validation': {'rocauc': 0.498015873015873}, 'Test': {'rocauc': 0.496861662063771}}
    =====Epoch 3
    Training...


    Iteration: 100%|██████████| 1029/1029 [00:25<00:00, 40.39it/s]
    Iteration:   2%|▏         | 16/1029 [00:00<00:06, 159.16it/s]

    Evaluating...


    Iteration: 100%|██████████| 1029/1029 [00:06<00:00, 163.72it/s]
    Iteration: 100%|██████████| 129/129 [00:00<00:00, 165.47it/s]
    Iteration: 100%|██████████| 129/129 [00:00<00:00, 166.95it/s]
    Iteration:   0%|          | 4/1029 [00:00<00:26, 39.08it/s]

    {'Train': {'rocauc': 0.5092237564450139}, 'Validation': {'rocauc': 0.5308641975308642}, 'Test': {'rocauc': 0.5}}
    =====Epoch 4
    Training...


    Iteration: 100%|██████████| 1029/1029 [00:25<00:00, 40.43it/s]
    Iteration:   1%|▏         | 15/1029 [00:00<00:06, 147.41it/s]

    Evaluating...


    Iteration: 100%|██████████| 1029/1029 [00:06<00:00, 163.42it/s]
    Iteration: 100%|██████████| 129/129 [00:00<00:00, 163.16it/s]
    Iteration: 100%|██████████| 129/129 [00:00<00:00, 164.34it/s]
    Iteration:   0%|          | 4/1029 [00:00<00:27, 37.85it/s]

    {'Train': {'rocauc': 0.5711309771569805}, 'Validation': {'rocauc': 0.6154636365863217}, 'Test': {'rocauc': 0.5926070414646865}}
    =====Epoch 5
    Training...


    Iteration: 100%|██████████| 1029/1029 [00:23<00:00, 43.58it/s]
    Iteration:   1%|▏         | 14/1029 [00:00<00:07, 140.00it/s]

    Evaluating...


    Iteration: 100%|██████████| 1029/1029 [00:06<00:00, 163.92it/s]
    Iteration: 100%|██████████| 129/129 [00:00<00:00, 166.01it/s]
    Iteration: 100%|██████████| 129/129 [00:00<00:00, 165.55it/s]

    {'Train': {'rocauc': 0.5522212973644183}, 'Validation': {'rocauc': 0.5579178301979228}, 'Test': {'rocauc': 0.6050927982386682}}
    Finished training!
    Best validation score: 0.682172251616696
    Test score: 0.6643677938932772


    



```python

```

## Assignment
请画出下方图片中的6号、3号和5号节点的从1层到3层的WL子树。

<img src=2560px-6n-graf.svg.png>


**6号、3号和5号节点的从1层到3层的WL子树：**

<img src=assignment-wl-subtree.png>


## Reference
[1] https://calcworkshop.com/trees-graphs/isomorphic-graph/

[2] https://mathworld.wolfram.com/IsomorphicGraphs.html

[3] Stanford OGB source code: https://github.com/snap-stanford/ogb

[4] Datawhale: https://github.com/datawhalechina/team-learning-nlp/blob/6f8cd26d2cff4f791bab7d553b06ed652b75b854/GNN/Markdown%E7%89%88%E6%9C%AC/8-%E5%9F%BA%E4%BA%8E%E5%9B%BE%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%9B%BE%E8%A1%A8%E7%A4%BA%E5%AD%A6%E4%B9%A0.md

[5] Pytorch_geometric: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers



```python

```




```python

```
