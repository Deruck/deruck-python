# 机器学习包
`deruck.ml`

包含模块与子包
- `datasets`：模块，包含加载各类数据集的`dataLoader`

## 1. 数据集
数据表示：
$$x=\left[ \begin{matrix}
| & | & {} & |  \\
{{x}^{(1)}} & {{x}^{(2)}} & \cdots  & {{x}^{(m)}}  \\
| & | & {} & |  \\
\end{matrix} \right]=\left[ \begin{matrix}
— & {{x}_{1}} & —  \\
— & {{x}_{2}} & —  \\
{} & {\vdots} & {}  \\
— & {{x}_{n}} & —  \\
\end{matrix} \right]\in {\mathbb{R}^{n\times m}}$$
其中，$x_i$ 表示输入的第 $i$ 个特征，$x^{(i)}$ 表示输入的第 $i$ 个变量/样本

### 1.1 回归数据集
```python
from deruck.ml.datasets import regDataLoader

reg_data_loader = regDataLoader()
X, y = reg_data_loader(data_name = "wine")
```
数据集及其说明
|数据集|说明|
|---|---|
|`wine`|[link](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)|
|`airfoil`|[link](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise)|


