
# coding: utf-8

# # TensorBoard Visualizations
# 
# このチュートリアルでは、さまざまなタイプのNLPベースの埋め込みをTensorBoardによって可視化する方法を学習します。 TensorBoardは、TensorFlowの実行とグラフを可視化して検査するためのデータ可視化化フレームワークです。 このチュートリアルでは、エンベディングプロジェクタと呼ばれるビルトインのテンソルボードビジュアライザを使用します。 埋め込みのような高次元のデータをインタラクティブに可視化して分析することができます。
# 
# In this tutorial, we will learn how to visualize different types of NLP based Embeddings via TensorBoard. TensorBoard is a data visualization framework for visualizing and inspecting the TensorFlow runs and graphs. We will use a built-in Tensorboard visualizer called *Embedding Projector* in this tutorial. It lets you interactively visualize and analyze high-dimensional data like embeddings.
# 

# ## Read Data 
# 
# このチュートリアルでは、変換されたMovieLensデータセット[1]が使用されています。 ここから最終的に準備されたcsvをダウンロードできます。
# 
# For this tutorial, a transformed MovieLens dataset<sup>[1]</sup> is used. You can download the final prepared csv from [here](https://github.com/parulsethi/DocViz/blob/master/movie_plots.csv).

# In[ ]:


import gensim
import pandas as pd
import numpy as np
import smart_open
import random
from smart_open import smart_open

# systhesis data
# s_data = np.loadtxt('bow_doc2500_word25_cluster4.txt')

# read data
dataframe = pd.read_csv('movie_plots.csv')
dataframe


# # 1. Visualizing Doc2Vec
# 
# このパートでは、TensorBoardを介して[Paragraph Vectors](https://arxiv.org/abs/1405.4053)と呼ばれるDoc2Vec埋め込みを視覚化する方法について学習します。 トレーニングのための入力ドキュメントは、Doc2Vecモデルが訓練されているムービーの概要です。
# 
# In this part, we will learn about visualizing Doc2Vec Embeddings aka [Paragraph Vectors](https://arxiv.org/abs/1405.4053) via TensorBoard. The input documents for training will be the synopsis of movies, on which Doc2Vec model is trained. 
# 
# <img src="Tensorboard.png">
# 
# ビジュアライゼーションは上の図に示すように散布図で、各データポイントはムービータイトルでラベル付けされ、対応するジャンルによって色付けされます。 また、上記のデータセットの埋め込みで設定された[Projector link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/DocViz/master/movie_plot_config.json)を参照することもできます。
# 
# The visualizations will be a scatterplot as seen in the above image, where each datapoint is labelled by the movie title and colored by it's corresponding genre. You can also visit this [Projector link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/DocViz/master/movie_plot_config.json) which is configured with my embeddings for the above mentioned dataset. 
# 
# 
# 
# ## Preprocess Text

# 以下では、トレーニング文書を読み込み、簡単なgensim前処理ツール（つまり、テキストを個々の単語にトークン化する、句読点を削除する、小文字に設定するなど）を使用して各文書を前処理し、単語のリストを返す関数を定義します 。 また、モデルをトレーニングするために、トレーニングコーパスの各ドキュメントにタグ/数値を関連付ける必要があります。 ここではタグは単にゼロから始まる行番号です。
# 
# Below, we define a function to read the training documents, pre-process each document using a simple gensim pre-processing tool (i.e., tokenize text into individual words, remove punctuation, set to lowercase, etc), and return a list of words. Also, to train the model, we'll need to associate a tag/number with each document of the training corpus. In our case, the tag is simply the zero-based line number.

# In[ ]:


def read_corpus(documents):
    for i, plot in enumerate(documents):
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(plot, max_len=30), [i])


# In[ ]:


train_corpus = list(read_corpus(dataframe.Plots))


# トレーニングコーパスを見てみましょう。
# 
# Let's take a look at the training corpus.

# In[ ]:


train_corpus[:2]


# ## Training the Doc2Vec Model
# 
# Doc2Vecのモデルを50ワードのベクトルサイズでインスタンス化し、トレーニングコーパスを55回反復処理します。 より高い頻度の単語に重みを与えるために、最小単語数を2に設定しました。 反復回数を増やすことでモデルの精度を向上させることができますが、大体の場合はトレーニング時間を増加させてしまいます。 このような短いドキュメントを持つ小さなデータセットは、より多くのトレーニングパスの恩恵を受けることができます。
# 
# We'll instantiate a Doc2Vec model with a vector size with 50 words and iterating over the training corpus 55 times. We set the minimum word count to 2 in order to give higher frequency words more weighting. Model accuracy can be improved by increasing the number of iterations but this generally increases the training time. Small datasets with short documents, like this one, can benefit from more training passes.

# In[ ]:


model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)


# さて、doctagあたりの文書埋め込みベクトルを保存します。
# 
# Now, we'll save the document embedding vectors per doctag.

# In[ ]:


model.save_word2vec_format('doc_tensor.w2v', doctag_vec=True, word_vec=False)  


# ## Prepare the Input files for Tensorboard

# Tensorboardは2つの入力ファイルを取ります。 1つは埋め込みベクトルを含み、もう1つは関連するメタデータを含む。 Gensimスクリプトを使用して、word2vec形式で保存された埋め込みファイルをTensorboardで必要なtsv形式に直接変換します。
# 
# Tensorboard takes two Input files. One containing the embedding vectors and the other containing relevant metadata. We'll use a gensim script to directly convert the embedding file saved in word2vec format above to the tsv format required in Tensorboard.

# In[ ]:


get_ipython().magic(u'run ../../gensim/scripts/word2vec2tensor.py -i doc_tensor.w2v -o movie_plot')


# 上記のスクリプトは、埋め込みベクトルを含む `movie_plot_tensor.tsv`とdoctagsを含む` movie_plot_metadata.tsv`の2つのファイルを生成します。 しかし、これらのdoctagsは一意のインデックス値であり、したがって、ドキュメントがどこに可視化されているかを解釈するのには有用ではありません。 したがって、 `movie_plot_metadata.tsv`を上書きして、2つの列を持つカスタムメタデータファイルを作成します。 最初の列はムービータイトル用で、2番目の列は対応するジャンル用です。
# 
# The script above generates two files, `movie_plot_tensor.tsv` which contain the embedding vectors and `movie_plot_metadata.tsv`  containing doctags. But, these doctags are simply the unique index values and hence are not really useful to interpret what the document was while visualizing. So, we will overwrite `movie_plot_metadata.tsv` to have a custom metadata file with two columns. The first column will be for the movie titles and the second for their corresponding genres.

# In[ ]:


with smart_open('movie_plot_metadata.tsv','w') as w:
    w.write('Titles\tGenres\n')
    for i,j in zip(dataframe.Titles, dataframe.Genres):
        w.write("%s\t%s\n" % (i,j))


# http://projector.tensorflow.org/ にアクセスして、左側のパネルで* Load data *をクリックして2つのファイルをアップロードできます。
# 
# デモの目的のために、私は上記の[here](https://github.com/parulsethi/DocViz) で訓練されたモデルから生成されたDoc2Vec埋め込みをアップロードしました。 これらのアップロードされた埋め込みで設定された埋め込みプロジェクタには、この[link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/DocViz/master/movie_plot_config.json)でアクセスできます。 。
# 
# Now you can go to http://projector.tensorflow.org/ and upload the two files by clicking on *Load data* in the left panel.
# 
# For demo purposes I have uploaded the Doc2Vec embeddings generated from the model trained above [here](https://github.com/parulsethi/DocViz). You can access the Embedding projector configured with these uploaded embeddings at this [link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/DocViz/master/movie_plot_config.json).

# # Using Tensorboard

# ビジュアライゼーションの目的で、上記のDoc2Vecモデルから得られる多次元の埋め込みは、2次元または3次元に縮小する場合があります。元の多次元埋め込みから情報を保持しようとする新しい2次元または3次元の埋め込みで基本的に終了するようにします。これらのベクトルがはるかに小さい次元に縮小されるので、それらの間の正確なコサイン/ユークリッド距離は保存されず、むしろ相対的であり、したがって、以下で見るように、最も近い類似結果が変化する可能性があります。
# 
# TensorBoardには、埋め込みを視覚化するための2つの一般的な次元削減方法があり、テキスト検索に基づくカスタムメソッドも提供されています。
# 
# - **主成分分析**：PCAは、データのグローバルな構造を探究することを目指しており、近傍とのローカルな類似点が失われる可能性があります。これは、より低い次元の部分空間における全分散を最大にし、したがって、より小さいものよりも大きい対の距離をしばしば保存する。 stackexchangeに関するこのうまく説明された[答え](https://stats.stackexchange.com/questions/176672/what-is-meant-by-pca-preserving-only-large-pairwise-distances) で、その背後にある直感を見てください。
# - **T-SNE**：T-SNEの考え方は、ローカルネイバー同士を近くに配置し、グローバル構造をほとんど完全に無視することです。地元の地域を探索したり、地元の集落を探すのに便利です。しかし、世界的な傾向は正確に表現されておらず、異なるグループ間の分離はしばしば保存されない（以下のデータのt-sneプロットを参照）。
# - **カスタムプロジェクション**：これは、異なる方向について定義したテキスト検索に基づくカスタム方法です。これは、ベクトル空間で意味のある方向、たとえば女性から男性、通貨などを見つけるのに便利です。
# 
# この[ドキュメント](https://www.tensorflow.org/get_started/embedding_viz)を参照すると、TensorBoardで使用できるさまざまなパネルを使用してナビゲートする方法が表示されます。
# 
# For the visualization purpose, the multi-dimensional embeddings that we get from the Doc2Vec model above, needs to be  downsized to 2 or 3 dimensions. So that we basically end up with a new 2d or 3d embedding which tries to preserve information from the original multi-dimensional embedding. As these vectors are reduced to a much smaller dimension, the exact cosine/euclidean distances between them are not preserved, but rather relative, and hence as you’ll see below the nearest similarity results may change.
# 
# TensorBoard has two popular dimensionality reduction methods for visualizing the embeddings and also provides a custom method based on text searches:
# 
# - **Principal Component Analysis**: PCA aims at exploring the global structure in data, and could end up losing the local similarities between neighbours. It maximizes the total variance in the lower dimensional subspace and hence, often preserves the larger pairwise distances better than the smaller ones. See an intuition behind it in this nicely explained [answer](https://stats.stackexchange.com/questions/176672/what-is-meant-by-pca-preserving-only-large-pairwise-distances) on stackexchange.
# 
# 
# - **T-SNE**: The idea of T-SNE is to place the local neighbours close to each other, and almost completely ignoring the global structure. It is useful for exploring local neighborhoods and finding local clusters. But the global trends are not represented accurately and the separation between different groups is often not preserved (see the t-sne plots of our data below which testify the same).
# 
# 
# - **Custom Projections**: This is a custom bethod based on the text searches you define for different directions. It could be useful for finding meaningful directions in the vector space, for example, female to male, currency to country etc.
# 
# You can refer to this [doc](https://www.tensorflow.org/get_started/embedding_viz) for instructions on how to use and navigate through different panels available in TensorBoard.

# ## Visualize using PCA
# 
# エンベディングプロジェクタは、上位10個の主成分を計算します。 左側のパネルのメニューでは、これらのコンポーネントを2つまたは3つの任意の組み合わせに投影できます。
# 
# The Embedding Projector computes the top 10 principal components. The menu at the left panel lets you project those components onto any combination of two or three. 
# 
# <img src="pca.png">
# 
# 上記のプロットは、最初の2つの主成分を使用して作成され、合計変動は36.5％になります。
# 
# The above plot was made using the first two principal components with total variance covered being 36.5%.

# 
# ## Visualize using T-SNE
# 
# データは、t-sneアルゴリズムの繰り返しごとにアニメーション化することによって可視化されます。 左のt-sneメニューでは、2つのハイパーパラメータの値を調整できます。 最初のものは** Perplexity **です。これは基本的に情報の尺度です。 これは有効な最近傍の数を設定するノブとして見ることができます<sup> [2] </sup>。 2番目は**学習率**で、新しい事例/データポイントに遭遇したときにアルゴリズムがどれくらい早く学習するかを定義します。
# 
# Data is visualized by animating through every iteration of the t-sne algorithm. The t-sne menu at the left lets you adjust the value of it's two hyperparameters. The first one is **Perplexity**, which is basically a measure of information. It may be viewed as a knob that sets the number of effective nearest neighbors<sup>[2]</sup>. The second one is **learning rate** that defines how quickly an algorithm learns on encountering new examples/data points.
# 
# <img src="tsne.png">
# 
# 上記のプロットは、perplexity 8、learning rate 10、iteration 500で生成されました。連続して実行すると結果が変わることがありますが、同じハイパーパラメータ設定で正確なプロットを得ることはできません。 しかし、いくつかの小クラスターは、上記のように、異なる方向性で形成を開始する。
# 
# The above plot was generated with perplexity 8, learning rate 10 and iteration 500. Though the results could vary on successive runs, and you may not get the exact plot as above with same hyperparameter settings. But some small clusters will start forming as above, with different orientations.

# # 2. Visualizing LDA
# 
# この部分では、TensorboardでLDAを視覚化する方法を見ていきます。 文書の埋め込みベクトルとして文書トピック分布を使用します。
# 基本的に、トピックをディメンションとして扱い、各ディメンションの値はドキュメント内のトピックのトピック割合を表します。
# 
# In this part, we will see how to visualize LDA in Tensorboard. We will be using the Document-topic distribution as the embedding vector of a document. Basically, we treat topics as the dimensions and the value in each dimension represents the topic proportion of that topic in the document.
# 
# ## Preprocess Text
# 
# 我々はコーパスの文書としてムービープロットを使用し、文書頻度に基づいてまれな単語や一般的な単語を削除します。 以下では、2つ以下のドキュメントまたは30％を超えるドキュメントに表示される単語を削除します。
# 
# We use the movie Plots as our documents in corpus and remove rare words and common words based on their document frequency. Below we remove words that appear in less than 2 documents or in more than 30% of the documents.

# In[ ]:


import pandas as pd
import re
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation
from gensim.models import ldamodel
from gensim.corpora.dictionary import Dictionary

# read data
dataframe = pd.read_csv('movie_plots.csv')

# remove stopwords and punctuations
def preprocess(row):
    return strip_punctuation(remove_stopwords(row.lower()))
    
dataframe['Plots'] = dataframe['Plots'].apply(preprocess)

# Convert data to required input format by LDA
texts = []
for line in dataframe.Plots:
    lowered = line.lower()
    words = re.findall(r'\w+', lowered, flags = re.UNICODE | re.LOCALE)
    texts.append(words)
# Create a dictionary representation of the documents.
dictionary = Dictionary(texts)

# Filter out words that occur less than 2 documents, or more than 30% of the documents.
dictionary.filter_extremes(no_below=2, no_above=0.3)
# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(text) for text in texts]


# ## Train LDA Model
# 

# In[ ]:


# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 50
iterations = 200
eval_every = None

# Train model
model = ldamodel.LdaModel(corpus=corpus, id2word=dictionary, chunksize=chunksize, alpha='auto', eta='auto', iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every)


# LDAモデルを習得する前に、[このノートブック](lda_training_tips.ipynb) を参照することもできます。 テキストデータの前処理のヒントや提案、LDAモデルを訓練して良い結果を得る方法について説明します。
# 
# You can refer to [this notebook](lda_training_tips.ipynb) also before training the LDA model. It contains tips and suggestions for pre-processing the text data, and how to train the LDA model to get good results.

# ## Doc-Topic distribution
# 
# 次に、ドキュメントのトピックの分布を推測する `get_document_topics`を使用します。 基本的には、入力コーパス内の各文書の（topic_id、topic_probability）のリストを返します。
# 
# Now we will use `get_document_topics` which infers the topic distribution of a document. It basically returns a list of (topic_id, topic_probability) for each document in the input corpus.

# In[ ]:


# Get document topics
all_topics = model.get_document_topics(corpus, minimum_probability=0)
all_topics[0]


# 上記の出力は、（topic_id、topic_probability）のリストとしてコーパス内の最初の文書のトピック分布を示しています。
# さて、文書のトピック分布をベクトル埋め込みとして使用して、Tensorboardを使用してコーパス内のすべての文書をプロットします。
# 
# The above output shows the topic distribution of first document in the corpus as a list of (topic_id, topic_probability).
# Now, using the topic distribution of a document as it's vector embedding, we will plot all the documents in our corpus using Tensorboard.

# ## Prepare the Input files for Tensorboard
# 
# Tensorboardは2つの入力ファイルを取ります.1つは埋め込みベクトルを含み、もう1つは関連するメタデータを含みます。 上記のように、文書のトピック分布を埋め込みベクトルとして使用します。 メタデータファイルは、そのジャンルのムービータイトルで構成されます。
# 
# Tensorboard takes two input files, one containing the embedding vectors and the other containing relevant metadata. As described above we will use the topic distribution of documents as their embedding vector. Metadata file will consist of Movie titles with their genres.

# In[ ]:


# create file for tensors
with smart_open('doc_lda_tensor.tsv','w') as w:
    for doc_topics in all_topics:
        for topics in doc_topics:
            w.write(str(topics[1])+ "\t")
        w.write("\n")
        
# create file for metadata
with smart_open('doc_lda_metadata.tsv','w') as w:
    w.write('Titles\tGenres\n')
    for j, k in zip(dataframe.Titles, dataframe.Genres):
        w.write("%s\t%s\n" % (j, k))


# http://projector.tensorflow.org/　にアクセスして、左側のパネルで"load data"をクリックしてこれら2つのファイルをアップロードできます。
# 
# デモの目的のために、私は上記の[ここ](https://github.com/parulsethi/LdaProjector/)で訓練されたモデルから生成されたLDAドキュメントトピック埋め込みをアップロードしました。 これらのアップロードされた埋め込みで設定された埋め込みプロジェクタには、[ここから](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/LdaProjector/master/doc_lda_config.json)でもアクセスできます。
# 
# 
# Now you can go to http://projector.tensorflow.org/ and upload these two files by clicking on Load data in the left panel.
# 
# For demo purposes I have uploaded the LDA doc-topic embeddings generated from the model trained above [here](https://github.com/parulsethi/LdaProjector/). You can also access the Embedding projector configured with these uploaded embeddings at this [link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/LdaProjector/master/doc_lda_config.json).

# ## Visualize using PCA
# 
# エンベディングプロジェクタは、上位10個の主成分を計算します。左側のパネルのメニューでは、これらのコンポーネントを2つまたは3つの任意の組み合わせに投影できます。
# 
# The Embedding Projector computes the top 10 principal components. The menu at the left panel lets you project those components onto any combination of two or three.
# <img src="doc_lda_pca.png">
# 
# PCAから、各データ点が文書を表すシンプレックス（この場合は四面体）が得られます。これらのデータポイントは、ムービーデータセットで与えられたGenresに従って色付けされています。
# 
# 私たちが見ることができるように、シンプレックスのコーナーには多くのポイントが集まります。これは、主に、使用しているベクトルの疎の度合いに起因します。コーナーの文書は主に単一のトピックに属します（したがって、1つの次元で大きな重みがあり、他の次元はおおよそゼロの重みを持ちます）。以下に説明するように、メタデータファイルを変更してムービータイトルとともに次元の重みを表示できます。
# 
# ここでは、クラスタのコーナーやエッジが主に属するトピックを調べるために、ドキュメントのタイトルに最も高い確率（topic_id、topic_probability）のトピックを追加します。そのためには、メタデータファイルを次のように上書きするだけです。
# 
# 
# From PCA, we get a simplex (tetrahedron in this case) where each data point represent a document. These data points are  colored according to their Genres which were given in the Movie dataset. 
# 
# As we can see there are a lot of points which cluster at the corners of the simplex. This is primarily due to the sparsity of vectors we are using. The documents at the corners primarily belongs to a single topic (hence, large weight in a single dimension and other dimensions have approximately zero weight.) You can modify the metadata file as explained below to see the dimension weights along with the Movie title.
# 
# Now, we will append the topics with highest probability (topic_id, topic_probability) to the document's title, in order to explore what topics do the cluster corners or edges dominantly belong to. For this, we just need to overwrite the metadata file as below:

# In[ ]:


tensors = []
for doc_topics in all_topics:
    doc_tensor = []
    for topic in doc_topics:
        if round(topic[1], 3) > 0:
            doc_tensor.append((topic[0], float(round(topic[1], 3))))
    # sort topics according to highest probabilities
    doc_tensor = sorted(doc_tensor, key=lambda x: x[1], reverse=True)
    # store vectors to add in metadata file
    tensors.append(doc_tensor[:5])

# overwrite metadata file
i=0
with smart_open('doc_lda_metadata.tsv','w') as w:
    w.write('Titles\tGenres\n')
    for j,k in zip(dataframe.Titles, dataframe.Genres):
        w.write("%s\t%s\n" % (''.join((str(j), str(tensors[i]))),k))
        i+=1


# 次に、以前のテンソルファイル "doc_lda_tensor.tsv"とこの新しいメタデータファイルをhttp://projector.tensorflow.org/　にアップロードします。
# 
# Next, we upload the previous tensor file "doc_lda_tensor.tsv" and this new metadata file to http://projector.tensorflow.org/ .
# <img src="topic_with_coordinate.png">
# 
# Voila！ ここでは、任意の点をクリックして、その文書内でそのタイトルと一緒にその可能性の高いトピックを表示することができます。 上記の例でわかるように、「ビバリーヒル警官」は主に0番目と1番目のトピックに属します。
# 
# Voila! Now we can click on any point to see it's top topics with their probabilty in that document, along with the title. As we can see in the above example, "Beverly hill cops" primarily belongs to the 0th and 1st topic as they have the highest probability amongst all.
# 
# 
# 
# ## Visualize using T-SNE
# T-SNEでは、データはt-sneアルゴリズムの各反復をアニメーション化することによって視覚化される。 左のt-sneメニューでは、2つのハイパーパラメータの値を調整できます。 最初のものはPerplexityで、基本的には情報の尺度です。 それは有効な最近隣の数を設定するノブとして見ることができる[2]。 2つ目は、新しいサンプル/データポイントに遭遇したときにアルゴリズムがどれほど迅速に学習するかを定義する学習率です。
# 
# さて、文書のトピック分布が埋め込みベクトルとして使用されると、t-sneは同じトピックに属する文書のクラスタを形成することになります。 これらのトピックのテーマを理解し、解釈するために、 `show_topic（）`を使ってトピックが構成する用語を調べることができます。
# 
# In T-SNE, the data is visualized by animating through every iteration of the t-sne algorithm. The t-sne menu at the left lets you adjust the value of it's two hyperparameters. The first one is Perplexity, which is basically a measure of information. It may be viewed as a knob that sets the number of effective nearest neighbors[2]. The second one is learning rate that defines how quickly an algorithm learns on encountering new examples/data points.
# 
# Now, as the topic distribution of a document is used as it’s embedding vector, t-sne ends up forming clusters of documents belonging to same topics. In order to understand and interpret about the theme of those topics, we can use `show_topic()` to explore the terms that the topics consisted of.
# 
# <img src="doc_lda_tsne.png">
# 上のプロットはperplexity 11、learning rate 10、iteration 1100で生成されました。連続して実行すると結果が変わることがありますが、同じハイパーパラメータ設定でも正確なプロットを得ることはできません。しかし、いくつかの小クラスターは、上記のように、異なる方向性で形成を開始します。
# 
# 上のクラスターの名前をムービーのジャンルに基づいて指定し、 `show_topic（）`を使用して、クラスター内で最も有益だったトピックの関連用語を表示しました。大部分のクラスタは、単一のトピックに支配的に属するdoocumetを持っていました。例えば、主にトピック0に属しているムービーを含むクラスタは、トピック0の下に表示される用語に基づいてファンタジー/ロマンスと名付けることができます。この[リンク](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/LdaProjector/master/doc_lda_config.json)、それが持っているムービーとその支配的なトピックに基づいてクラスターのラベルを締結しようとします。その上にマウスを置くと、各ポイントの上位5つのトピックが表示されます。
# 
# 今度は、上記の画像では10個以上のクラスターであることがわかりましたが、num_topics = 10というモデルを訓練しました。その理由は、それらのクラスターがほとんどなく、トピックに近いトピック確率値を持つ複数のトピックに属する文書があるからです。
# 
# The above plot was generated with perplexity 11, learning rate 10 and iteration 1100. Though the results could vary on successive runs, and you may not get the exact plot as above even with same hyperparameter settings. But some small clusters will start forming as above, with different orientations.
# 
# I named some clusters above based on the genre of it's movies and also using the `show_topic()` to see relevant terms of the topic which was most prevelant in a cluster. Most of the clusters had doocumets belonging dominantly to a single topic. For ex. The cluster with movies belonging primarily to topic 0 could be named Fantasy/Romance based on terms displayed below for topic 0. You can play with the visualization yourself on this [link](http://projector.tensorflow.org/?config=https://raw.githubusercontent.com/parulsethi/LdaProjector/master/doc_lda_config.json) and try to conclude a label for clusters based on movies it has and their dominant topic. You can see the top 5 topics of every point by hovering over it.
# 
# Now, we can notice that their are more than 10 clusters in the above image, whereas we trained our model for `num_topics=10`. It's because their are few clusters, which has documents belonging to more than one topic with an approximately close topic probability values.

# In[ ]:


model.show_topic(topicid=0, topn=15)


# トピックをより効率的に推論するためにpyLDAvisを使用することもできます。 それは個々のトピックに関連した用語のより深い点検を提供します。 このため、用語の関連性**と呼ばれる指標をトピックに使用します。これにより、ユーザーは有意義なトピック解釈に最も適した用語を柔軟にランク付けできます。 λという重みパラメータを調整して、トピックを効率的に区別するのに役立つ有益な用語を表示することができます。
# 
# You can even use pyLDAvis to deduce topics more efficiently. It provides a deeper inspection of the terms highly associated with each individual topic. For this, it uses a measure called **relevance** of a term to a topic that allows users to flexibly rank terms best suited for a meaningful topic interpretation. It's weight parameter called λ can be adjusted to display useful terms which could help in differentiating topics efficiently.

# In[ ]:


import pyLDAvis.gensim

viz = pyLDAvis.gensim.prepare(model, corpus, dictionary)
pyLDAvis.display(viz)


# 重みパラメータλは、トピック内の確率（λ= 1）に応じて単純にランク付けされるのか、コーパス（λ= 0）を横切る限界確率によって正規化されるのかに基づいて用語のランクを調整するノブとして見ることができる ）。 λ= 1に設定すると、大きな番号の場合も同様の順位のランキングが得られます。 したがって、それらを区別することが困難になり、λ= 0を設定することは、現在のトピックへの排他性にのみ基づいているため、単一のトピックでしか発生しないような希少な用語が生じる可能性があり、  [(Sievert and Shirley 2014)](https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf)は、ユーザー調査に基づいてλ= 0.6の最適値を示唆しています。
# 
# The weight parameter λ can be viewed as a knob to adjust the ranks of the terms based on whether they are simply ranked according to their probability in the topic (λ=1) or are normalized by their marginal probability across the corpus (λ=0). Setting λ=1 could result in similar ranking of terms for large no. of topics hence making it difficult to differentiate between them, and setting λ=0 ranks terms solely based on their exclusiveness to current topic which could result in such rare terms that occur in only a single topic and hence the topics may remain difficult to interpret. [(Sievert and Shirley 2014)](https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf) suggested the optimal value of λ=0.6 based on a user study.

# # Conclusion
# 
# Tensorboardのエンベディングプロジェクタを使用してドキュメント埋め込みとLDAドキュメントトピックの配布を視覚化する方法を学びました。 これは、単語埋め込み、文書埋め込み、または遺伝子発現および生物学的配列などの異なるタイプのデータを視覚化するための有用なツールである。 2Dテンソルの入力が必要なだけで、提供されたアルゴリズムを使用してデータを探索することができます。 最寄りの検索を実行して、クエリポイントと最も類似したデータポイントを見つけることもできます。
# 
# We learned about visualizing the Document Embeddings and LDA Doc-topic distributions through Tensorboard's Embedding Projector. It is a useful tool for visualizing different types of data for example, word embeddings, document embeddings or the gene expressions and biological sequences. It just needs an input of 2D tensors and then you can explore your data using provided algorithms. You can also perform nearest neighbours search to find most similar data points to your query point.
# 
# # References
#  1. https://grouplens.org/datasets/movielens/
#  2. https://lvdmaaten.github.io/tsne/
# 
