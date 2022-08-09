from pathlib import Path

from torch import kaiser_window
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from data.database import MongoClient, StoryDataset


class StoryDashboard:
    def __init__(self, mongo_client: MongoClient, story_dataset: StoryDataset,
                 graph_layout="sfdp", edge_width=0.5, arrow_length=1.1, arrow_width=0.035):
        self.mongo_client = mongo_client
        self.story_dataset = story_dataset
        self.drawer = GraphDrawer(graph_layout=graph_layout, edge_width=edge_width, arrow_length=arrow_length,
                                  arrow_width=arrow_width)
        
        self.target_nodes = None
        self.graph_clusters = None  # graph_clusters[0] is the single tree, graph_clusters[1:] are the clusters
        self.node_text_map = None
        
    def get_graph(self, expr, cluster_iou_threshold=0.6):
        expr["storified"] = True
        records = self.mongo_client.find(expr)
        nodes = [r["global_event_id"] for r in records]
        records = self.mongo_client.find(expr)
        self.target_nodes = nodes
        
        graph_single_tree = self.story_dataset.get_family_tree_graph(nodes, mode="single_tree")
        graph_clusters = self.story_dataset.get_family_tree_graph(nodes, mode="tree_clusters", 
                                                                  cluster_iou_threshold=cluster_iou_threshold)
        graph_clusters = [graph_single_tree] + graph_clusters
        
        records = self.mongo_client.find(
            {"global_event_id": {"$in": list(graph_single_tree.nodes())}, "storified": True},
        )
        
        records = list(records)
        titles = {
            r["global_event_id"]: r["title"] for r in records
        }
        node_text_map = {}
        for r in records:
            global_event_id = r["global_event_id"]
            title = r["title"]
            date_added = r["date_added"] + timedelta(hours=8)
            url = r['url']
            titles_upstream = []
            for edge in graph_single_tree.in_edges(global_event_id):
                titles_upstream.append(titles[edge[0]])
            node_text_map[global_event_id] = f"{title}<br>{date_added}<br>{url}<br>{'<br>'.join(titles_upstream)}"
        
        # node_text_map = {
            # r["global_event_id"]: f"{r['title']}<br>{r['date_added'] + timedelta(hours=8)}<br>{r['url']}" for r in records
            # }
        
        self.graph_clusters = graph_clusters
        self.node_text_map = node_text_map
        
        return graph_single_tree, graph_clusters, node_text_map
    
    def get_graph_today(self, cluster_iou_threshold=0.6):
        expr = {"date_added": {"$gte": datetime.utcnow() - timedelta(days=1)}}
        return self.get_graph(expr, cluster_iou_threshold=cluster_iou_threshold)
    
    def __len__(self):
        if self.graph_clusters is None:
            return 0
        return len(self.graph_clusters)
    
    def __getitem__(self, idx):
        if len(self) == 0:
            print(f"no story loaded, run `get_graph` or `get_graph_today` first")
            return
        ancestors, descendants = self.story_dataset.split_ancestors_descendants(self.graph_clusters[idx],
                                                                                self.target_nodes)
        return self.drawer.draw(self.graph_clusters[idx], self.node_text_map, ancestors, descendants)


class GraphDrawer:
    def __init__(self, graph_layout="sfdp", edge_width=0.5, arrow_length=1.1, arrow_width=0.035):
        self.graph_layout = graph_layout
        self.edge_width = edge_width
        self.arrow_length = arrow_length
        self.arrow_width = arrow_width
    
    def draw(self, graph, node_text_map=None, ancestors=None, descendants=None):
        pos = nx.nx_agraph.graphviz_layout(graph, self.graph_layout)
        if ancestors is not None and descendants is not None:
            node_trace_ancestors = self.get_node_trace(graph, pos, node_text_map, ancestors)
            node_trace_descendents = self.get_node_trace(graph, pos, node_text_map, descendants, hightlight_nodes=True)
            node_traces = [node_trace_ancestors, node_trace_descendents]
        else:
            node_traces = [self.get_node_trace(graph, pos, node_text_map=node_text_map)]
        if len(graph.edges()) > 0:
            line_trace, arrow_trace = self.get_edge_trace(graph, pos)
            data = [line_trace, arrow_trace, *node_traces]
        else:
            data = node_traces
        fig = go.Figure(data=data,
                        layout=go.Layout(
                            title='<br>Story Dashboard',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
        return fig
        
    def get_edge_trace(self, graph, pos):
        line_trace = self.get_line_trace(graph, pos)
        arrow_trace = self.get_arrow_trace(graph, pos)
        
        return line_trace, arrow_trace
        
    def get_line_trace(self, graph, pos):
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        line_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=self.edge_width, color='#888'),
            hoverinfo='none',
            mode='lines')
        return line_trace
    
    def get_arrow_trace(self, graph, pos):
        p0s = []
        p1s = []    
        for edge in graph.edges():
            p0s.append(pos[edge[0]])
            p1s.append(pos[edge[1]])
        p0s = np.array(p0s)
        p1s = np.array(p1s)
        
        v_s = p1s - p0s
        w_s = v_s / np.linalg.norm(v_s, axis=1, keepdims=True)  
        u_s = np.vstack([-w_s[:, 1], w_s[:, 0]]).T * 10 # u orthogonal on  w

        pms = p1s / 4 * 3 + p0s / 4
        Ps = pms - self.arrow_length * w_s
        Ss = Ps - self.arrow_width * u_s
        Ts = Ps + self.arrow_width * u_s
        
        arrows_x = []
        arrows_y = []
        for P, S, T, pm in zip(Ps, Ss, Ts, pms):
            arrows_x += [S[0], T[0], pm[0], S[0], None]
            arrows_y += [S[1], T[1], pm[1], S[1], None]
        arrow_trace = go.Scatter(x=arrows_x, 
                                 y=arrows_y, 
                                 mode='lines', 
                                 fill='toself', 
                                 fillcolor='blue', 
                                 line_color='blue')
        return arrow_trace
        
    def get_node_trace(self, graph, pos, node_text_map=None, target_nodes=None, hightlight_nodes=False):
        if target_nodes is None:
            target_nodes = graph.nodes()
        node_x = []
        node_y = []
        for node in target_nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        marker = dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=5,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2)

        if hightlight_nodes:
            kwargs = {"marker_symbol": "circle", "marker_line_color": "red"}
            marker.pop("colorbar")
        else:
            kwargs = {}        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=marker, **kwargs)
        
        node_in_degrees = []
        node_texts = []
        for node in target_nodes:
            node_in_degrees.append(graph.in_degree(node))
            if node_text_map is not None:
                node_texts.append(node_text_map[node])

        node_trace.marker.color = node_in_degrees
        if node_texts:
            node_trace.text = node_texts
        
        return node_trace
    
    
if __name__ == "__main__":
    from tqdm import tqdm
    
    mongo_client = MongoClient()
    story_dataset = StoryDataset()

    story_dashboard = StoryDashboard(mongo_client, story_dataset)
    story_dashboard.get_graph_today()
    for fig in tqdm(story_dashboard):
        pass
    