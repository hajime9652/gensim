
# coding: utf-8

# # Topic Networks
# 
# このノートブックでは、ネットワークグラフを使用してトピックモデルを視覚化する方法を学習します。 ネットワークは、トピックモデルを探索するのに最適な方法です。 あるコンテキストに属するトピックが他のコンテキストでどのようにいくつかのトピックに関連し、それらの間の共通の要因を発見する方法をナビゲートするために使用することができます。 私たちはそれらを使って類似の話題のコミュニティを見つけ、最も影響力のあるトピックを特定することができます。 またはネットワーク分析用に設計されたその他のワークフローをいくつでも実行できます。
# 
# In this notebook, we will learn how to visualize topic model using network graphs. Networks can be a great way to explore topic models. We can use it to navigate that how topics belonging to one context may relate to some topics in other context and discover common factors between them. We can use them to find communities of similar topics and pinpoint the most influential topic that has large no. of connections or perform any number of other workflows designed for network analysis.

# In[ ]:

get_ipython().system("pip install plotly>=2.0.16 --ignore-installed# 2.0.16 need for support 'hovertext' argument from create_dendrogram function")


# In[ ]:

from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

import numpy as np


# ## Train Model
# このノートブックのkaggleの[fake news dataset](https://www.kaggle.com/mrisdal/fake-news)を使用します。 まず、データを前処理し、LDAを使用してトピックモデルをトレーニングします。 この [notebook](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/lda_training_tips.ipynb)を参照して、テキストデータの前処理に関するヒントや提案、および 良い結果を得るためのLDAモデルを訓練します。
# 
# We'll use the [fake news dataset](https://www.kaggle.com/mrisdal/fake-news) from kaggle for this notebook. First step is to preprocess the data and train our topic model using LDA. You can refer to this [notebook](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/lda_training_tips.ipynb) also for tips and suggestions of pre-processing the text data, and how to train LDA model for getting good results.

# In[ ]:

df_fake = pd.read_csv('fake.csv')
df_fake[['title', 'text', 'language']].head()
df_fake = df_fake.loc[(pd.notnull(df_fake.text)) & (df_fake.language=='english')]

# remove stopwords and punctuations
def preprocess(row):
    return strip_punctuation(remove_stopwords(row.lower()))
    
df_fake['text'] = df_fake['text'].apply(preprocess)

# Convert data to required input format by LDA
texts = []
for line in df_fake.text:
    lowered = line.lower()
    words = re.findall(r'\w+', lowered, flags=re.UNICODE|re.LOCALE)
    texts.append(words)
# Create a dictionary representation of the documents.
dictionary = Dictionary(texts)

# Filter out words that occur less than 2 documents, or more than 30% of the documents.
dictionary.filter_extremes(no_below=2, no_above=0.4)
# Bag-of-words representation of the documents.
corpus_fake = [dictionary.doc2bow(text) for text in texts]


# In[ ]:

lda_fake = LdaModel(corpus=corpus_fake, id2word=dictionary, num_topics=35, chunksize=1500, iterations=200, alpha='auto')
lda_fake.save('lda_35')


# In[ ]:

lda_fake = LdaModel.load('lda_35')


# ## Visualize topic network
# まず、各トピックペア間の距離を格納する距離行列を計算します。 ネットワークグラフのノードはトピックを表し、それらの間のエッジは2つの接続ノード/トピック間の距離に基づいて作成されます。
# 
# Firstly, a distance matrix is calculated to store distance between every topic pair. The nodes of the network graph will represent topics and the edges between them will be created based on the distance between two connecting nodes/topics.

# In[ ]:

# get topic distributions
topic_dist = lda_fake.state.get_lambda()

# get topic terms
num_words = 50
topic_terms = [{w for (w, _) in lda_fake.show_topic(topic, topn=num_words)} for topic in range(topic_dist.shape[0])]


# エッジを描画するために、gensimで使用可能なさまざまな種類の距離メトリックを使用して、各トピックのペア間の距離を計算することができます。 次に、それ以上の距離のトピックペアが接続されないように、距離値の閾値を定義しなければなりません。
# 
# To draw the edges, we can use different types of distance metrics available in gensim for calculating the distance between every topic pair. Next, we'd have to define a threshold of distance value such that the topic-pairs with distance above that does not get connected. 

# In[ ]:

from scipy.spatial.distance import pdist, squareform
from gensim.matutils import jensen_shannon
import networkx as nx
import itertools as itt

# calculate distance matrix using the input distance metric
def distance(X, dist_metric):
    return squareform(pdist(X, lambda u, v: dist_metric(u, v)))

topic_distance = distance(topic_dist, jensen_shannon)

# store edges b/w every topic pair along with their distance
edges = [(i, j, {'weight': topic_distance[i, j]})
         for i, j in itt.combinations(range(topic_dist.shape[0]), 2)]

# keep edges with distance below the threshold value
k = np.percentile(np.array([e[2]['weight'] for e in edges]), 20)
edges = [e for e in edges if e[2]['weight'] < k]


# これでエッジができたので、注釈付きネットワークグラフをプロットしましょう。 ノード上をホバリングすると、topic_idがトップの単語とともに表示され、エッジ上にマウスを置くと、接続する2つのトピックの交差/異なる単語が表示されます。
# 
# Now that we have our edges, let's plot the annotated network graph. On hovering over the nodes, we'll see the topic_id along with it's top words and on hovering over the edges, we'll see the intersecting/different words of the two  topics that it connects. 

# In[ ]:

import plotly.offline as py
from plotly.graph_objs import *
import plotly.figure_factory as ff

py.init_notebook_mode()

# add nodes and edges to graph layout
G = nx.Graph()
G.add_nodes_from(range(topic_dist.shape[0]))
G.add_edges_from(edges)

graph_pos = nx.spring_layout(G)


# In[ ]:

# initialize traces for drawing nodes and edges 
node_trace = Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=Marker(
        showscale=True,
        colorscale='YlOrRd',
        reversescale=True,
        color=[],
        size=10,
        colorbar=dict(
            thickness=15,
            xanchor='left'
        ),
        line=dict(width=2)))

edge_trace = Scatter(
    x=[],
    y=[],
    text=[],
    line=Line(width=0.5, color='#888'),
    hoverinfo='text',
    mode='lines')


# no. of terms to display in annotation
n_ann_terms = 10

# add edge trace with annotations
for edge in G.edges():
    x0, y0 = graph_pos[edge[0]]
    x1, y1 = graph_pos[edge[1]]
    
    pos_tokens = topic_terms[edge[0]] & topic_terms[edge[1]]
    neg_tokens = topic_terms[edge[0]].symmetric_difference(topic_terms[edge[1]])
    pos_tokens = list(pos_tokens)[:min(len(pos_tokens), n_ann_terms)]
    neg_tokens = list(neg_tokens)[:min(len(neg_tokens), n_ann_terms)]
    annotation = "<br>".join((": ".join(("+++", str(pos_tokens))), ": ".join(("---", str(neg_tokens)))))
    
    x_trace = list(np.linspace(x0, x1, 10))
    y_trace = list(np.linspace(y0, y1, 10))
    text_annotation = [annotation] * 10
    x_trace.append(None)
    y_trace.append(None)
    text_annotation.append(None)
    
    edge_trace['x'] += x_trace
    edge_trace['y'] += y_trace
    edge_trace['text'] += text_annotation

# add node trace with annotations
for node in G.nodes():
    x, y = graph_pos[node]
    node_trace['x'].append(x)
    node_trace['y'].append(y)
    node_info = ''.join((str(node+1), ': ', str(list(topic_terms[node])[:n_ann_terms])))
    node_trace['text'].append(node_info)
    
# color node according to no. of connections
for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color'].append(len(adjacencies))


# In[ ]:

fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(showlegend=False,
                hovermode='closest',
                xaxis=XAxis(showgrid=True, zeroline=False, showticklabels=True),
                yaxis=YAxis(showgrid=True, zeroline=False, showticklabels=True)))

py.iplot(fig)


# For the above graph, we just used the 20th percentile of all the distance values. But we can experiment with few different values also such that the graph doesn’t become too crowded or too sparse and we could get an optimum amount of information about similar topics or any interesting relations b/w different topics.
# 
# Or we can also get an idea of threshold from the dendrogram (with ‘single’ linkage function). You can refer to [this notebook](http://nbviewer.jupyter.org/github/parulsethi/gensim/blob/b9e7ab54dde98438b0e4f766ee764b81af704367/docs/notebooks/Topic_dendrogram.ipynb) for more details on topic dendrogram visualization. The y-values in the dendrogram represent the metric distances and if we choose a certain y-value then only those topics which are clustered below it would be connected. So let's plot the dendrogram now to see the sequential clustering process with increasing distance values.

# In[ ]:

from gensim.matutils import jensen_shannon
import scipy as scp
from scipy.cluster import hierarchy as sch
from scipy import spatial as scs

# get topic distributions
topic_dist = lda_fake.state.get_lambda()

# get topic terms
num_words = 300
topic_terms = [{w for (w, _) in lda_fake.show_topic(topic, topn=num_words)} for topic in range(topic_dist.shape[0])]

# no. of terms to display in annotation
n_ann_terms = 10

# use Jenson-Shannon distance metric in dendrogram
def js_dist(X):
    return pdist(X, lambda u, v: jensen_shannon(u, v))

# define method for distance calculation in clusters
linkagefun=lambda x: sch.linkage(x, 'single')

# calculate text annotations
def text_annotation(topic_dist, topic_terms, n_ann_terms, linkagefun):
    # get dendrogram hierarchy data
    d = js_dist(topic_dist)
    Z = linkagefun(d)
    P = sch.dendrogram(Z, orientation="bottom", no_plot=True)

    # store topic no.(leaves) corresponding to the x-ticks in dendrogram
    x_ticks = np.arange(5, len(P['leaves']) * 10 + 5, 10)
    x_topic = dict(zip(P['leaves'], x_ticks))

    # store {topic no.:topic terms}
    topic_vals = dict()
    for key, val in x_topic.items():
        topic_vals[val] = (topic_terms[key], topic_terms[key])

    text_annotations = []
    # loop through every trace (scatter plot) in dendrogram
    for trace in P['icoord']:
        fst_topic = topic_vals[trace[0]]
        scnd_topic = topic_vals[trace[2]]
        
        # annotation for two ends of current trace
        pos_tokens_t1 = list(fst_topic[0])[:min(len(fst_topic[0]), n_ann_terms)]
        neg_tokens_t1 = list(fst_topic[1])[:min(len(fst_topic[1]), n_ann_terms)]

        pos_tokens_t4 = list(scnd_topic[0])[:min(len(scnd_topic[0]), n_ann_terms)]
        neg_tokens_t4 = list(scnd_topic[1])[:min(len(scnd_topic[1]), n_ann_terms)]

        t1 = "<br>".join((": ".join(("+++", str(pos_tokens_t1))), ": ".join(("---", str(neg_tokens_t1)))))
        t2 = t3 = ()
        t4 = "<br>".join((": ".join(("+++", str(pos_tokens_t4))), ": ".join(("---", str(neg_tokens_t4)))))

        # show topic terms in leaves
        if trace[0] in x_ticks:
            t1 = str(list(topic_vals[trace[0]][0])[:n_ann_terms])
        if trace[2] in x_ticks:
            t4 = str(list(topic_vals[trace[2]][0])[:n_ann_terms])

        text_annotations.append([t1, t2, t3, t4])

        # calculate intersecting/diff for upper level
        intersecting = fst_topic[0] & scnd_topic[0]
        different = fst_topic[0].symmetric_difference(scnd_topic[0])

        center = (trace[0] + trace[2]) / 2
        topic_vals[center] = (intersecting, different)

        # remove trace value after it is annotated
        topic_vals.pop(trace[0], None)
        topic_vals.pop(trace[2], None)  
        
    return text_annotations

# get text annotations
annotation = text_annotation(topic_dist, topic_terms, n_ann_terms, linkagefun)

# Plot dendrogram
dendro = ff.create_dendrogram(topic_dist, distfun=js_dist, labels=range(1, 36), linkagefun=linkagefun, hovertext=annotation)
dendro['layout'].update({'width': 1000, 'height': 600})
py.iplot(dendro)


# From observing this dendrogram, we can try the threshold values between 0.3 to 0.35 for network graph, as the topics are clustered in distinct groups below them and this could plot separate clusters of related topics in the network graph.
# 
# But then why do we need to use network graph if the dendrogram already shows the topic clusters with a clear sequence of how topics joined one after the other. The problem is that we can't see the direct relation of any topic with another topic except if they are directly paired at the first hierarchy level. The network graph let's us explore the inter-topic distances and at the same time observe clusters of closely related topics.
