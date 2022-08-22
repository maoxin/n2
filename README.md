# N2

基于新闻向量相似性及多元多元线性回归的故事构建。

## Usage

1. 同步 GDELT 数据集（到 csv），下载最新新闻文本（到 Mongodb），使用 REALM 模型编码文本为向量（到 Milvus）。

```python
python data/retrieve_data.py
```

2. 对于每条新闻，搜索其前 3 天向量内积相似性最高的10条新闻，以该条新闻向量为应变量，搜索出的 10 条新闻向量为自变量进行多元线性回归。选择 $R^2 > 0.7$ 的多元线性回归中 p-value $< 0.05$ 的新闻对为 edge 建 graph 作为故事，保存至 networkx pickle。

```python
python story/story_construction.py
```

3. 在`story_graph_vis.ipynb`中使用 plotly 进行故事可视化。


## Config

可在 `config.yaml` 中配置需要考虑的新闻源。