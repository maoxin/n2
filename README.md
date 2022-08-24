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

## TODO

### Feature
- [x] today's news
	- distinguish today's news with others
- [x] story
	- represent a story as a graph, whose nodes are an event together with the events caused that event directly or indirectly
- [x] story cluster
	- represent a story cluster as a set of stories having a big overlap of nodes (high IoU)
	- [ ] story cluster based on genealogy, i.e., news lineage
		- using Sankey diagram https://plotly.com/python/sankey-diagram/
		- ancestors within a given degree https://stackoverflow.com/questions/39930083/efficiently-identifying-ancestors-descendants-within-a-given-distance-in-networ, https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_tree.html
---
- [ ] story similarity
	- (choice 1) graph similarity (sub-graph based on time range or ancestor level can be used)
	- (choice 2) extend causes to history event, and using sibling relationship
---
- visualization
	- graph structure
		- [ ] tree structure (collapsable)
		- [ ] multi-level structure
	- story (cluster) relationship
		- [ ] timeline and geo
		- [ ] waterfall view
	- [ ] story quantification (word cloud, sentiment trend, etc.)
	- [ ] read vs unread
---
- [ ] information integration for analysis
	- [x] direct causes (show x below y?)
	- [ ] intermediate causes
	- [ ] source causes
- [ ] story classification
	- political, etc.
---
- [ ] support search


### Model
- [x] information completeness
	- use the first two paragraphs in embedding instead of the first one because the first paragraph contains only the info resource
- [x] information deduplication
	- remove duplicate news with the same titles (see MSN in different regions)
	- [ ] further deduplicate based on url and titles
- [ ] hyper-parameters search for R2 and p values
- [ ] time-window optimization
- [ ] multi-language supports


### Data
- [ ] sophisticated graph database

## Guidlines
- [x] guideline for reading the graph
	- in-degree first, which is the result, also the sink of info
	- edge nodes for root reason