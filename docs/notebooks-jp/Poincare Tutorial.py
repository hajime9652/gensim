
# coding: utf-8

# # Tutorial on Poincaré Embeddings

# このノートブックでは、ポアンカレの埋め込みに関する基本的な考え方と使用例について説明し、どのような操作を行うことができるかを示します。 より包括的な技術の詳細と結果については、この [ブログ](https://rare-technologies.com/implementing-poincare-embeddings)がより適切なリソースになるかもしれません。
# 
# This notebook discusses the basic ideas and use-cases for Poincaré embeddings and demonstrates what kind of operations can be done with them. For more comprehensive technical details and results, this [blog post](https://rare-technologies.com/implementing-poincare-embeddings) may be a more appropriate resource.

# ## 1. Introduction

# ### 1.1 Concept and use-case
# ポアンカレ埋め込みは、グラフのノードのベクトル表現を学習する方法です。入力データは、ノード間の関係（エッジ）のリストの形式であり、ノードのベクトルがそれらの間の距離を正確に表すようにモデルは表現を学習しようとする。
# 
# 学習された埋め込みは、接続されたノードをお互いに接近させ、接続されていないノードを互いに遠くに置くことによって、_hierarchy_および_similarity_類似性の両方の概念を捕捉する。より階層の低いノードを原点から遠くに、すなわちより高い規範で配置することによって、階層を拡張することができる。
# 
# このモデルでは、このモデルを使用してWordNetの名詞階層にノードの埋め込みを学習し、評価のセクションで説明されている3つのタスク（再構成、リンク予測、語彙含意）を評価します。我々は、これらのタスクに関するPoincaréモデル実装の結果を、他のオープンソースの実装と論文の結果と比較しました。
# 
# また、指向性と非対称性を持つWordNetの名詞階層とは異なり、ノードの埋め込みを対称グラフに学習するPoincaréモデルの変種についても説明しています。このモデルの論文で使用されているデータセットは、ノードが研究者であり、2つの研究者が論文を共著していることを表す科学共同ネットワークです。
# 
# この亜種はまだ実装されていないため、チュートリアルや実験の一部ではありません。
# 
# 
# Poincaré embeddings are a method to learn vector representations of nodes in a graph. The input data is of the form of a list of relations (edges) between nodes, and the model tries to learn representations such that the vectors for the nodes accurately represent the distances between them.
# 
# The learnt embeddings capture notions of both _hierarchy_ and _similarity_ - similarity by placing connected nodes close to each other and unconnected nodes far from each other; hierarchy by placing nodes lower in the hierarchy farther from the origin, i.e. with higher norms.
# 
# The paper uses this model to learn embeddings of nodes in the WordNet noun hierarchy, and evaluates these on 3 tasks - reconstruction, link prediction and lexical entailment, which are described in the section on evaluation. We have compared the results of our Poincaré model implementation on these tasks to other open-source implementations and the results mentioned in the paper.
# 
# The paper also describes a variant of the Poincaré model to learn embeddings of nodes in a symmetric graph, unlike the WordNet noun hierarchy, which is directed and asymmetric. The datasets used in the paper for this model are scientific collaboration networks, in which the nodes are researchers and an edge represents that the two researchers have co-authored a paper.
# 
# This variant has not been implemented yet, and is therefore not a part of our tutorial and experiments.
# 
# 
# ### 1.2 Motivation
# ここでの主な新規性は、これらの埋め込みが、一般的に使用されるユークリッド空間ではなく、双曲線空間で学習されることです。 この背後にある理由は、双曲線空間がグラフに本質的に存在する階層情報をキャプチャするのに適しているからです。 ノード間の距離を保ちながらユークリッド空間にノードを埋め込むには、通常非常に多くの次元が必要です。 これを簡単に説明すると以下のようになります -
# 
# The main innovation here is that these embeddings are learnt in hyperbolic space, as opposed to the commonly used Euclidean space. The reason behind this is that hyperbolic space is more suitable for capturing any hierarchical information inherently present in the graph. Embedding nodes into a Euclidean space while preserving the distance between the nodes usually requires a very high number of dimensions. A simple illustration of this can be seen below - 
#  
#  ![Example tree](https://raw.githubusercontent.com/RaRe-Technologies/gensim/poincare_model_keyedvectors/docs/notebooks/poincare/example_tree.png)
#  
#  ここで、ノードの位置は、2次元ユークリッド空間におけるそれらのベクトルの位置を表します。理想的には、ノード `（A、D）`のベクトル間の距離は `（D、H）`と `H`との子ノード間の距離と同じでなければなりません。同様に、「H」のすべての子ノードは、ノード「A」から等しく離れていなければなりません。ユークリッド空間で木の次数と深さが大きくなるにつれて、これらの距離を正確に保存することはますます困難になります。階層構造にはクロスコネクト（事実上、有向グラフ）があり、より困難にすることもあります。
# 
# これらの距離を正確に反映することができる2次元ユークリッド空間におけるこの単純な木の表現は存在しない。これはさらに多くのディメンションを追加することで解決できますが、これは必要なディメンションの数が指数関数的に増加するにつれて計算上実行不可能になります。
# 双曲線空間は、距離が直線ではなく、曲線であるメトリック空間であり、このようなツリー状の階層構造は、低次元でもより正確に距離を取得する表現を持つことができます。
# 
# Here, the positions of nodes represent the positions of their vectors in 2-D euclidean space. Ideally, the distances between the vectors for nodes `(A, D)` should be the same as that between `(D, H)` and as that between `H` and its child nodes. Similarly, all the child nodes of `H` must be equally far away from node `A`. It becomes progressively hard to accurately preserve these distances in Euclidean space as the degree and depth of the tree grows larger. Hierarchical structures may also have cross-connections (effectively a directed graph), making this harder.
# 
# There is no representation of this simple tree in 2-dimensional Euclidean space which can reflect these distances correctly. This can be solved by adding more dimensions, but this becomes computationally infeasible as the number of required dimensions grows exponentially. 
# Hyperbolic space is a metric space in which distances aren't straight lines - they are curves, and this allows such tree-like hierarchical structures to have a representation that captures the distances more accurately even in low dimensions.

# ## 2. Training the embedding

# In[ ]:

get_ipython().magic('cd ../..')


# In[ ]:

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import os
import logging
import numpy as np

from gensim.models.poincare import PoincareModel, PoincareKeyedVectors, PoincareRelations

logging.basicConfig(level=logging.INFO)

poincare_directory = os.path.join(os.getcwd(), 'docs', 'notebooks-jp', 'poincare')
data_directory = os.path.join(poincare_directory, 'data')
wordnet_mammal_file = os.path.join(data_directory, 'wordnet_mammal_hypernyms.tsv')


# モデルは、リレーションの反復可能性を使用して初期化することができます。ここで、リレーションは単純にノードのペアで、
# 
# The model can be initialized using an iterable of relations, where a relation is simply a pair of nodes - 

# In[ ]:

model = PoincareModel(train_data=[('node.1', 'node.2'), ('node.2', 'node.3')])


# このモデルは、1行に1つのリレーションを含むcsvのようなファイルから初期化することもできます。 このモジュールは便利なクラス `PoincareRelations`を提供しています。
# 
# The model can also be initialized from a csv-like file containing one relation per line. The module provides a convenience class `PoincareRelations` to do so.

# In[ ]:

relations = PoincareRelations(file_path=wordnet_mammal_file, delimiter='\t')
model = PoincareModel(train_data=relations)


# 上記はモデルを初期化するだけであり、トレーニングを開始しないことに注意してください。 モデルをトレーニングするには、
# 
# Note that the above only initializes the model and does not begin training. To train the model - 

# In[ ]:

model = PoincareModel(train_data=relations, size=2, burn_in=0)
model.train(epochs=1, print_every=500)


# 同じモデルは、モデルがまだ収束していないとユーザが判断した場合に、より多くのエポックでさらに訓練することができます。
# 
# The same model can be trained further on more epochs in case the user decides that the model hasn't converged yet.

# In[ ]:

model.train(epochs=1, print_every=500)


# モデルは、2つの方法を使用して保存・ロードすることができます。
# 
# The model can be saved and loaded using two different methods - 

# In[ ]:

# Saves the entire PoincareModel instance, the loaded model can be trained further
model.save('/tmp/test_model')
PoincareModel.load('/tmp/test_model')


# In[ ]:

# Saves only the vectors from the PoincareModel instance, in the commonly used word2vec format
model.kv.save_word2vec_format('/tmp/test_vectors')
PoincareKeyedVectors.load_word2vec_format('/tmp/test_vectors')


# ## 3. What the embedding can be used for

# In[ ]:

# Load an example model
models_directory = os.path.join(poincare_directory, 'models')
test_model_path = os.path.join(models_directory, 'gensim_model_batch_size_10_burn_in_0_epochs_50_neg_20_dim_50')
model = PoincareModel.load(test_model_path)


# 学習された表現は、さまざまな種類の有用な操作を実行するために使用できます。 このセクションは2つに分かれています - 紙に直接記述されている簡単な操作と、ヒントを得た実験的な操作と、詳細を絞り込むための作業が必要な場合があります。
# 
# このセクションで使用されているモデルは、WordNetの上層グラフの推移的閉包について訓練されています。 推移的閉包は、WordNetグラフのすべての直接的および間接的な上位語のリストです。 直接上位語の例は `（seat.n.03、furniture.n.01）`であり、間接上位語の例は `（seat.n.03、physical_entity.n.01） ` です。
# 
# The learnt representations can be used to perform various kinds of useful operations. This section is split into two - some simple operations that are directly mentioned in the paper, as well as some experimental operations that are hinted at, and might require more work to refine.
# 
# The models that are used in this section have been trained on the transitive closure of the WordNet hypernym graph. The transitive closure is the list of all the direct and indirect hypernyms in the WordNet graph. An example of a direct hypernym is `(seat.n.03, furniture.n.01)` while an example of an indirect hypernym is `(seat.n.03, physical_entity.n.01)`.

# ### 3.1 Simple operations

# 次のすべての操作は、双曲線空間内の2つのノード間の距離の概念にのみ基づいています。
# 
# All the following operations are based simply on the notion of distance between two nodes in hyperbolic space.

# In[ ]:

# Distance between any two nodes
model.kv.distance('plant.n.02', 'tree.n.01')


# In[ ]:

model.kv.distance('plant.n.02', 'animal.n.01')


# In[ ]:

# Nodes most similar to a given input node
model.kv.most_similar('electricity.n.01')


# In[ ]:

model.kv.most_similar('man.n.01')


# In[ ]:

# Nodes closer to node 1 than node 2 is from node 1
model.kv.nodes_closer_than('dog.n.01', 'carnivore.n.01')


# In[ ]:

# Rank of distance of node 2 from node 1 in relation to distances of all nodes from node 1
model.kv.rank('dog.n.01', 'carnivore.n.01')


# In[ ]:

# Finding Poincare distance between input vectors
vector_1 = np.random.uniform(size=(100,))
vector_2 = np.random.uniform(size=(100,))
vectors_multiple = np.random.uniform(size=(5, 100))

# Distance between vector_1 and vector_2
print(PoincareKeyedVectors.vector_distance(vector_1, vector_2))
# Distance between vector_1 and each vector in vectors_multiple
print(PoincareKeyedVectors.vector_distance_batch(vector_1, vectors_multiple))


# ### 3.2 Experimental operations

# これらの演算は、ベクトルのノルムがその階層的位置を表すという概念に基づいています。 リーフノードは通常、最も高いノルムを持つ傾向があり、階層を上げると、ノルムは減少し、ルートノードは中心（または原点）に近くなります。
# 
# These operations are based on the notion that the norm of a vector represents its hierarchical position. Leaf nodes typically tend to have the highest norms, and as we move up the hierarchy, the norm decreases, with the root node being close to the center (or origin).

# In[ ]:

# Closest child node
model.kv.closest_child('person.n.01')


# In[ ]:

# Closest parent node
model.kv.closest_parent('person.n.01')


# In[ ]:

# Position in hierarchy - lower values represent that the node is higher in the hierarchy
print(model.kv.norm('person.n.01'))
print(model.kv.norm('teacher.n.01'))


# In[ ]:

# Difference in hierarchy between the first node and the second node
# Positive values indicate the first node is higher in the hierarchy
print(model.kv.difference_in_hierarchy('person.n.01', 'teacher.n.01'))


# In[ ]:

# One possible descendant chain
model.kv.descendants('mammal.n.01')


# In[ ]:

# One possible ancestor chain
model.kv.ancestors('dog.n.01')


# チェーンは対称ではないことに注意してください。最も近い子に再帰的に降下しながら、 `mammal`から始まり、` carnivore`の最も近い子は `dog`ですが、` dog`から最も近い親は`canine`.です。
# 
# Note that the chains are not symmetric - while descending to the closest child recursively, starting with `mammal`, the closest child of `carnivore` is `dog`, however, while ascending from `dog` to the closest parent, the closest parent to `dog` is `canine`. 

# これは、ポアンカレ距離が（メトリック空間内の任意の距離のように）対称であるという事実にもかかわらずです。 非対称性は、ノード「Y」が、「X」より高いノルム（階層の下位）を有するすべてのノードのうちのノード「X」に最も近いノードであっても、ノード「Y」よりも低いノルム（階層内で高い）を有する全てのノードの中でノード「X」に最も近いノードはノード「Y」というわけでないということに立脚します。
# 
# This is despite the fact that Poincaré distance is symmetric (like any distance in a metric space). The asymmetry stems from the fact that even if node `Y` is the closest node to node `X` amongst all nodes with a higher norm (lower in the hierarchy) than `X`, node `X` may not be the closest node to node `Y` amongst all the nodes with a lower norm (higher in the hierarchy) than `Y`.

# ## 4. Useful Links

# 1. [Original paper by Facebook AI Research](https://arxiv.org/pdf/1705.08039)
# 2. [Blog post describing technical challenges in implementation](https://rare-technologies.com/implementing-poincare-embeddings)
# 3. [Detailed evaluation notebook to reproduce results](https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/Poincare%20Evaluation.ipynb)
