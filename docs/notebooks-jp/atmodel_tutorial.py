
# coding: utf-8

# # The author-topic model: LDA with metadata
# このチュートリアルでは、Gensimで著者 - トピックモデルを使用する方法を学習します。私たちは論文の著者についての洞察を得るために、科学論文からなるコーパスに適用します。
# 
# 著者 - トピックモデルは、潜在的ディリクレ割り当て（LDA）の拡張であり、コーパス内の著者のトピック表現を学習することを可能にします。このモデルは、Web上の投稿のタグなど、ドキュメント上の任意の種類のラベルに適用できます。このモデルは、データ検索の新しい方法、機械学習パイプラインの機能、著者（またはタグ）の予測、または既存のメタデータを使用してトピックモデルを簡単に活用するために使用できます。
# 
# 著者 - トピックモデルの理論的側面については、例えば[Rosen-Zvi and co-authors 2004](https://mimno.infosci.cornell.edu/info6150/readings/398.pdf)を参照してください。 Gensim実装で使用されているアルゴリズムに関するレポートは、すぐに利用可能になります。
# 
# 本チュートリアルでは、もちろん、トピックモデリング、LDA、Gensimに精通していることを前提とします。 LDAまたはそのGensim実装に慣れていない場合は、そこから開始することをお勧めします。これらのリソースのいくつかを考えてみましょう。
# * LDAモデルへの穏やかな紹介：http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/
# * GensimのLDA APIドキュメント：https://radimrehurek.com/gensim/models/ldamodel.html
# Gensimのトピックモデリング：http://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html
# * [Pre-processing and training LDA](lda_training_tips.ipynb)
# 
# 
# > **注：**
# >
# >このチュートリアルを自分で実行するには、Jupyter、Gensim、SpaCy、Scikit-Learn、Bokeh、Pandasをインストールします。ピップを使用して：
# >
# > `pip install jupyter gensim spacy sklearn bokeh pandas`
# >
# > `python -m spacy.en.download`を使ってSpaCyのデータをダウンロードする必要があることに注意してください。
# >
# >ノートブックをhttps://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks/atmodel_tutorial.ipynb からダウンロードしてください。
# 
# このチュートリアルでは、モデルのデータを準備する方法、モデルを訓練する方法、結果の表現をさまざまな方法で探索する方法について学習します。 Geoffrey HintonやYann LeCunのような有名な作家の話題を調べ、作者を次元削減によってプロットして類似性の比較を行うことで比較します。
# 
# In this tutorial, you will learn how to use the author-topic model in Gensim. We will apply it to a corpus consisting of scientific papers, to get insight about the authors of the papers.
# 
# The author-topic model is an extension of Latent Dirichlet Allocation (LDA), that allows us to learn topic representations of authors in a corpus. The model can be applied to any kinds of labels on documents, such as tags on posts on the web. The model can be used as a novel way of data exploration, as features in machine learning pipelines, for author (or tag) prediction, or to simply leverage your topic model with existing metadata.
# 
# To learn about the theoretical side of the author-topic model, see [Rosen-Zvi and co-authors 2004](https://mimno.infosci.cornell.edu/info6150/readings/398.pdf), for example. A report on the algorithm used in the Gensim implementation will be available soon.
# 
# Naturally, familiarity with topic modelling, LDA and Gensim is assumed in this tutorial. If you are not familiar with either LDA, or its Gensim implementation, I would recommend starting there. Consider some of these resources:
# * Gentle introduction to the LDA model: http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/
# * Gensim's LDA API documentation: https://radimrehurek.com/gensim/models/ldamodel.html
# * Topic modelling in Gensim: http://radimrehurek.com/topic_modeling_tutorial/2%20-%20Topic%20Modeling.html
# * [Pre-processing and training LDA](lda_training_tips.ipynb)
# 
# 
# > **NOTE:**
# >
# > To run this tutorial on your own, install Jupyter, Gensim, SpaCy, Scikit-Learn, Bokeh and Pandas, e.g. using pip:
# >
# > `pip install jupyter gensim spacy sklearn bokeh pandas`
# >
# > Note that you need to download some data for SpaCy using `python -m spacy.en.download`.
# >
# > Download the notebook at https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks/atmodel_tutorial.ipynb.
# 
# In this tutorial, we will learn how to prepare data for the model, how to train it, and how to explore the resulting representation in different ways. We will inspect the topic representation of some well known authors like Geoffrey Hinton and Yann LeCun, and compare authors by plotting them in reduced dimensionality and performing similarity queries.
# 
# ## Analyzing scientific papers
# 私たちが使用するデータは、神経情報処理システム会議（NIPS）の機械学習に関する科学論文から構成されます。前述の[Pre-processing and training LDA](lda_training_tips.ipynb)チュートリアルで使用されているのと同じデータセットです。
# 
# モデルの定性分析を行い、時にはデータの主題を理解する必要があります。このチュートリアルを自分で実行する場合は、よく知っている内容のデータセットに適用することを検討してください。たとえば、[StackExchange datadump datasets](https://archive.org/details/stackexchange)のいずれかを試してください。
# 
# Sam Roweisのウェブサイト（http://www.cs.nyu.edu/~roweis/data.html） からデータをダウンロードできます。または、以下のセルを実行するだけで、ダウンロードされ、 `tmpに展開されます。
# 
# The data we will be using consists of scientific papers about machine learning, from the Neural Information Processing Systems conference (NIPS). It is the same dataset used in the [Pre-processing and training LDA](lda_training_tips.ipynb) tutorial, mentioned earlier.
# 
# We will be performing qualitative analysis of the model, and at times this will require an understanding of the subject matter of the data. If you try running this tutorial on your own, consider applying it on a dataset with subject matter that you are familiar with. For example, try one of the [StackExchange datadump datasets](https://archive.org/details/stackexchange).
# 
# You can download the data from Sam Roweis' website (http://www.cs.nyu.edu/~roweis/data.html). Or just run the cell below, and it will be downloaded and extracted into your `tmp.

# In[ ]:


get_ipython().system("curl -o - 'http://www.cs.nyu.edu/~roweis/data/nips12raw_str602.tgz' > /tmp/nips12raw_str602.tgz")


# In[ ]:


import tarfile

filename = './tmp/nips12raw_str602.tgz'
tar = tarfile.open(filename, 'r:gz')
for item in tar:
    tar.extract(item, path='/tmp')


# 以下のセクションでは、実装の機能のいくつかを使用して、データをロードし、事前処理し、モデルをトレーニングし、結果を探索します。 このプロセスに精通していれば、今すぐに読み込みと前処理をスキップしても構いません。
# 
# In the following sections we will load the data, pre-process it, train the model, and explore the results using some of the implementation's functionality. Feel free to skip the loading and pre-processing for now, if you are familiar with the process.
# 
# ### Loading the data
# 下のセルでは、データセット内のフォルダとファイルをクロールし、ファイルをメモリに読み込みます。
# 
# In the cell below, we crawl the folders and files in the dataset, and read the files into memory.

# In[ ]:


import os, re, io
from smart_open import smart_open

# Folder containing all NIPS papers.
data_dir = './tmp/nipstxt/'  # Set this path to the data on your machine.

# Folders containin individual NIPS papers.
yrs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
dirs = ['nips' + yr for yr in yrs]

# Get all document texts and their corresponding IDs.
docs = []
doc_ids = []
for yr_dir in dirs:
    files = os.listdir(data_dir + yr_dir)  # List of filenames.
    for filen in files:
        # Get document ID.
        (idx1, idx2) = re.search('[0-9]+', filen).span()  # Matches the indexes of the start end end of the ID.
        doc_ids.append(yr_dir[4:] + '_' + str(int(filen[idx1:idx2])))
        
        # Read document text.
        # Note: ignoring characters that cause encoding errors.
        with smart_open(data_dir + yr_dir + '/' + filen, 'rb', encoding='utf-8', errors='ignore') as fid:
            txt = fid.read()
            
        # Replace any whitespace (newline, tabs, etc.) by a single space.
        txt = re.sub('\s', ' ', txt)
        
        docs.append(txt)


# 著者名からドキュメントIDへのマッピングを構築します。
# 
# Construct a mapping from author names to document IDs.

# In[ ]:


from smart_open import smart_open
filenames = [data_dir + 'idx/a' + yr + '.txt' for yr in yrs]  # Using the years defined in previous cell.

# Get all author names and their corresponding document IDs.
author2doc = dict()
i = 0
for yr in yrs:
    # The files "a00.txt" and so on contain the author-document mappings.
    filename = data_dir + 'idx/a' + yr + '.txt'
    for line in smart_open(filename, 'rb', errors='ignore', encoding='utf-8'):
        # Each line corresponds to one author.
        contents = re.split(',', line)
        author_name = (contents[1] + contents[0]).strip()
        # Remove any whitespace to reduce redundant author names.
        author_name = re.sub('\s', '', author_name)
        # Get document IDs for author.
        ids = [c.strip() for c in contents[2:]]
        if not author2doc.get(author_name):
            # This is a new author.
            author2doc[author_name] = []
            i += 1
        
        # Add document IDs to author.
        author2doc[author_name].extend([yr + '_' + id for id in ids])

# Use an integer ID in author2doc, instead of the IDs provided in the NIPS dataset.
# Mapping from ID of document in NIPS datast, to an integer ID.
doc_id_dict = dict(zip(doc_ids, range(len(doc_ids))))
# Replace NIPS IDs by integer IDs.
for a, a_doc_ids in author2doc.items():
    for i, doc_id in enumerate(a_doc_ids):
        author2doc[a][i] = doc_id_dict[doc_id]


# ### Pre-processing text
# 
# テキストは次の手順で事前処理されます。
# * テキストをトークン化する。
# * すべての空白を1つのスペースで置き換えます。
# * すべての句読点と数字を削除します。
# * ストップワードを削除します。
# * 単語を略語にする。
# * 複数単語の名前付きエンティティを追加します。
# * 頻繁なバイグラムを追加する。
# * 頻繁で珍しい言葉を削除する。
# 
# 重たい処理の多くは、素晴らしいパッケージ、Spacyによって行われます。 Spacyは「産業レベルの自然言語処理」として市場に出回っており、高速でマルチプロセッシングが可能で使いやすくなっています。 最初に、それをインポートし、NLPピンプラインを英語で読み込みましょう。
# 
# The text will be pre-processed using the following steps:
# * Tokenize text.
# * Replace all whitespace by single spaces.
# * Remove all punctuation and numbers.
# * Remove stopwords.
# * Lemmatize words.
# * Add multi-word named entities.
# * Add frequent bigrams.
# * Remove frequent and rare words.
# 
# A lot of the heavy lifting will be done by the great package, Spacy. Spacy markets itself as "industrial-strength natural language processing", is fast, enables multiprocessing, and is easy to use. First, let's import it and load the NLP pipline in english.

# In[ ]:


import spacy
nlp = spacy.load('en')


# 以下のコードでは、Spacyはアルファベット以外の文字の削除、ストップワードの削除、字形化および名前付きエンティティ認識をトークン化します。
# 
# 1つの名前のエンティティが既に存在するため、複数の単語で構成される名前付きエンティティのみを保持することに注意してください。
# 
# In the code below, Spacy takes care of tokenization, removing non-alphabetic characters, removal of stopwords, lemmatization and named entity recognition.
# 
# Note that we only keep named entities that consist of more than one word, as single word named entities are already there.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'processed_docs = []    \nfor doc in nlp.pipe(docs, n_threads=4, batch_size=100):\n    # Process document using Spacy NLP pipeline.\n    \n    ents = doc.ents  # Named entities.\n\n    # Keep only words (no numbers, no punctuation).\n    # Lemmatize tokens, remove punctuation and remove stopwords.\n    doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]\n\n    # Remove common words from a stopword list.\n    #doc = [token for token in doc if token not in STOPWORDS]\n\n    # Add named entities, but only if they are a compound of more than word.\n    doc.extend([str(entity) for entity in ents if len(entity) > 1])\n    \n    processed_docs.append(doc)')


# In[ ]:


docs = processed_docs
del processed_docs


# 以下では、Gensimモデルを使用してバイグラムを追加します。 これは、名前付きエンティティ認識と同じ目標を達成すること、つまり特に重要な隣接単語を見つけることに注意してください。
# 
# Below, we use a Gensim model to add bigrams. Note that this achieves the same goal as named entity recognition, that is, finding adjacent words that have some particular significance.

# In[ ]:


# Compute bigrams.
from gensim.models import Phrases
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)


# これで、ボキャブラリーが完成したので、辞書を作成する準備が整いました。 次に、一般的な単語（$50 \%$以上で発生している）とまれな単語（全体で$ 20 $回以下）が削除されます。
# 
# Now we are ready to construct a dictionary, as our vocabulary is finalized. We then remove common words (occurring $> 50\%$ of the time), and rare words (occur $< 20$ times in total).

# In[ ]:


# Create a dictionary representation of the documents, and filter out frequent and rare words.

from gensim.corpora import Dictionary
dictionary = Dictionary(docs)

# Remove rare and common tokens.
# Filter out words that occur too frequently or too rarely.
max_freq = 0.5
min_wordcount = 20
dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)

_ = dictionary[0]  # This sort of "initializes" dictionary.id2token.


# BoWを計算することによって、著者 - トピックモデルを供給するために、文書のベクトル化された表現を生成する。
# 
# We produce the vectorized representation of the documents, to supply the author-topic model with, by computing the bag-of-words.

# In[ ]:


# Vectorize data.

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]


# 私たちのデータの次元を調べましょう。
# 
# Let's inspect the dimensionality of our data.

# In[ ]:


print('Number of authors: %d' % len(author2doc))
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# ### Train and use model
# 
# 私たちは、前のセクションで準備したデータをもとに著者 - トピックモデルを訓練します。
# 
# 著者 - トピックモデルへのインタフェースは、GensimのLDAのインタフェースに非常に似ています。 著者 - トピックモデルには、コーパス、IDから単語へのマッピング（ `id2word`）、トピックの数（` num_topics`）の他に、IDマッピングを文書化する（ `author2doc`）か、もとに戻す（` doc2author `）。
# 
# 以下では、これもスキップしています。
# * 最適化問題の収束を向上させるために、データセットに対する「パス」の数を増やしました。
# * 各文書の「反復回数」を減らしました（上記に関連しています）。
# * ミニバッチサイズ（ `chunksize`）を指定しました（主にトレーニングをスピードアップするため）。
# * バウンド評価（ `eval_every`）をオフにしました（計算に時間がかかるため）。
# * 最適化問題の収束を改善するために、 `alpha`と` eta`priorsの自動学習を有効にしました。
# * 乱数ジェネレータのランダムな状態（ `random_state`）を設定します（これらの実験を再現可能にするため）。
# 
# モデルをロードしてトレーニングします。
# 
# We train the author-topic model on the data prepared in the previous sections. 
# 
# The interface to the author-topic model is very similar to that of LDA in Gensim. In addition to a corpus, ID to word mapping (`id2word`) and number of topics (`num_topics`), the author-topic model requires either an author to document ID mapping (`author2doc`), or the reverse (`doc2author`).
# 
# Below, we have also (this can be skipped for now):
# * Increased the number of `passes` over the dataset (to improve the convergence of the optimization problem).
# * Decreased the number of `iterations` over each document (related to the above).
# * Specified the mini-batch size (`chunksize`) (primarily to speed up training).
# * Turned off bound evaluation (`eval_every`) (as it takes a long time to compute).
# * Turned on automatic learning of the `alpha` and `eta` priors (to improve the convergence of the optimization problem).
# * Set the random state (`random_state`) of the random number generator (to make these experiments reproducible).
# 
# We load the model, and train it.

# In[ ]:


from gensim.models import AuthorTopicModel
get_ipython().run_line_magic('time', 'model = AuthorTopicModel(corpus=corpus, num_topics=10, id2word=dictionary.id2token,                 author2doc=author2doc, chunksize=2000, passes=1, eval_every=0,                 iterations=1, random_state=1)')


# モデルが収束していないと思われる場合は、 `model.update（）`を使ってトレーニングを続けることができます。 追加の文書や著者がいる場合は `model.update（corpus、author2doc）`を呼び出します。
# 
# モデルを調べる前に、そのモデルを改善しようとしましょう。 これを行うために、私たちは乱数ジェネレータ（ `random_state`）に異なる種を与えることによって、異なるランダム初期化を持ついくつかのモデルを訓練します。 [top_topics](https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel.top_topics)メソッドを使用してトピックの一貫性を評価し、最もトピックのコンシステンシーが高いモデルを選択します
# 
# If you believe your model hasn't converged, you can continue training using `model.update()`. If you have additional documents and/or authors call `model.update(corpus, author2doc)`.
# 
# Before we explore the model, let's try to improve upon it. To do this, we will train several models with different random initializations, by giving different seeds for the random number generator (`random_state`). We evaluate the topic coherence of the model using the [top_topics](https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel.top_topics) method, and pick the model with the highest topic coherence.
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'model_list = []\nfor i in range(5):\n    model = AuthorTopicModel(corpus=corpus, num_topics=10, id2word=dictionary.id2token, \\\n                    author2doc=author2doc, chunksize=2000, passes=100, gamma_threshold=1e-10, \\\n                    eval_every=0, iterations=1, random_state=i)\n    top_topics = model.top_topics(corpus)\n    tc = sum([t[1] for t in top_topics])\n    model_list.append((model, tc))')


# トピックのコンシステンシーが最も高いモデルを選択します。
# 
# Choose the model with the highest topic coherence.

# In[ ]:


model, tc = max(model_list, key=lambda x: x[1])
print('Topic coherence: %.3e' %tc)


# モデルを保存して、再度訓練する必要がなくなる、再度ロードする方法も示します。
# 
# We save the model, to avoid having to train it again, and also show how to load it again.

# In[ ]:


# Save model.
model.save('/tmp/model.atmodel')


# In[ ]:


# Load model.
model = AuthorTopicModel.load('/tmp/model.atmodel')


# ### Explore author-topic representation
# 
# モデルを訓練したので、著者とトピックを調べ始めることができます。
# 
# まず、トピックで最も重要な単語を単に出力してみましょう。 以下では、トピック0を印刷しています。各トピックは単語のセットに関連付けられており、各単語はそのトピックで表現される可能性があります。
# 
# Now that we have trained a model, we can start exploring the authors and the topics.
# 
# First, let's simply print the most important words in the topics. Below we have printed topic 0. As we can see, each topic is associated with a set of words, and each word has a probability of being expressed under that topic.

# In[ ]:


model.show_topic(0)


# 以下では、それぞれのトピックに直感的に近いものと思われるものに基づいて、各トピックにラベルを付けました。
# 
# Below, we have given each topic a label based on what each topic seems to be about intuitively. 

# In[ ]:


topic_labels = ['Circuits', 'Neuroscience', 'Numerical optimization', 'Object recognition',                'Math/general', 'Robotics', 'Character recognition',                 'Reinforcement learning', 'Speech recognition', 'Bayesian modelling']


# 単に `model.show_topics（num_topics = 10）`を呼び出すのではなく、出力を少しフォーマットして、概要を得る方が簡単です。
# 
# Rather than just calling `model.show_topics(num_topics=10)`, we format the output a bit so it is easier to get an overview.

# In[ ]:


for topic in model.show_topics(num_topics=10):
    print('Label: ' + topic_labels[topic[0]])
    words = ''
    for word, prob in model.show_topic(topic[0]):
        words += word + ' '
    print('Words: ' + words)
    print()


# これらのトピックは決して完璧ではありません。 *連載トピック*、*侵入単語*、*ランダムトピック*、*不均衡トピック*（ [Mimno and co-authors 2011](https://people.cs.umass.edu/~wallach/publications/mimno11optimizing.pdf)）のような問題を抱えています。 しかし、このチュートリアルの目的のために行います。以下では、 `model [name]`構文を使用して、著者のトピックの分布を取得します。 各トピックには、特定の著者が与えられたときに表現される確率がありますが、特定のしきい値を超えるトピックのみが表示されます。
# 
# These topics are by no means perfect. They have problems such as *chained topics*, *intruded words*, *random topics*, and *unbalanced topics* (see [Mimno and co-authors 2011](https://people.cs.umass.edu/~wallach/publications/mimno11optimizing.pdf)). They will do for the purposes of this tutorial, however.
# 
# Below, we use the `model[name]` syntax to retrieve the topic distribution for an author. Each topic has a probability of being expressed given the particular author, but only the ones above a certain threshold are shown.

# In[ ]:


model['YannLeCun']


# いくつかの著者のトップトピックを出力しましょう。 まず、これをより簡単に行うための機能を作ります。
# 
# Let's print the top topics of some authors. First, we make a function to help us do this more easily.

# In[ ]:


from pprint import pprint

def show_author(name):
    print('\n%s' % name)
    print('Docs:', model.author2doc[name])
    print('Topics:')
    pprint([(topic_labels[topic[0]], topic[1]) for topic in model[name]])


# 以下では、高プロファイルの研究者をいくつか出力し、それらを検査します。 これらのうちの3つ、Yann LeCun、Geoffrey E. Hinton、Christof Kochがあります。
# 
# しかしTerrence J. Sejnowskiの結果は驚くべきことである。 彼は神経科学者ですから、彼は "神経科学"のラベルを得ると期待しています。 これは、Sejnowskiが視覚認知の神経科学の側面で働いていること、またはおそらくトピックを誤ってラベル付けしていること、あるいは単にこのトピックがあまり有益ではないことを示している可能性があります。
# 
# Below, we print some high profile researchers and inspect them. Three of these, Yann LeCun, Geoffrey E. Hinton and Christof Koch, are spot on. 
# 
# Terrence J. Sejnowski's results are surprising, however. He is a neuroscientist, so we would expect him to get the "neuroscience" label. This may indicate that Sejnowski works with the neuroscience aspects of visual perception, or perhaps that we have labeled the topic incorrectly, or perhaps that this topic simply is not very informative.

# In[ ]:


show_author('YannLeCun')


# In[ ]:


show_author('GeoffreyE.Hinton')


# In[ ]:


show_author('TerrenceJ.Sejnowski')


# In[ ]:


show_author('ChristofKoch')


# #### Simple model evaluation methods
# モデルの予測パフォーマンスの尺度である単語単位の境界を計算することができます（再構成エラーとも言えます）。
# 
# そのためには、自動的に構築できる `doc2author`辞書が必要です。
# 
# 
# We can compute the per-word bound, which is a measure of the model's predictive performance (you could also say that it is the reconstruction error).
# 
# To do that, we need the `doc2author` dictionary, which we can build automatically.

# In[ ]:


from gensim.models import atmodel
doc2author = atmodel.construct_doc2author(model.corpus, model.author2doc)


# では、単語単位の境界を評価しましょう。
# 
# Now let's evaluate the per-word bound.

# In[ ]:


# Compute the per-word bound.
# Number of words in corpus.
corpus_words = sum(cnt for document in model.corpus for _, cnt in document)

# Compute bound and divide by number of words.
perwordbound = model.bound(model.corpus, author2doc=model.author2doc,                            doc2author=model.doc2author) / corpus_words
print(perwordbound)


# LDAクラスのように、トピックのcoherenceを計算することによって、トピックの品質を評価することができます。 これをたとえばに使用します。 どのトピックが低品質であるか、またはモデル選択の基準として見つけます。
# 
# We can evaluate the quality of the topics by computing the topic coherence, as in the LDA class. Use this to e.g. find out which of the topics are poor quality, or as a metric for model selection.

# In[ ]:


get_ipython().run_line_magic('time', 'top_topics = model.top_topics(model.corpus)')


# #### Plotting the authors
# 今、私たちは、太平洋列島のような種類のプロットを次のように作成します。 このプロットの目的は、著者トピック表現を直観的に探求する方法を提供することです。
# 
# 私たちは、（`model.state.gamma`に格納されている）すべての著者 - トピック分布をとり、それらを2D空間に埋め込みます。 これを行うために、t-SNEを使用してこのデータの次元性を減らします。
# 
# t-SNEは、ポイント間の距離を維持しながら、データセットの次元数を減らそうとする方法です。 つまり、2人の著者が下のプロットで近くにいると、そのトピックの分布は似ています。
# 
# 以下のセルでは、著者 - トピック表現をt-SNE空間に変換します。 少数の文書ですべての著者を表示したくない場合は、 `smallest_author`値を増やすことができます。
# 
# 
# Now we're going to produce the kind of pacific archipelago looking plot below. The goal of this plot is to give you a way to explore the author-topic representation in an intuitive manner.
# 
# We take all the author-topic distributions (stored in `model.state.gamma`) and embed them in a 2D space. To do this, we reduce the dimensionality of this data using t-SNE. 
# 
# t-SNE is a method that attempts to reduce the dimensionality of a dataset, while maintaining the distances between the points. That means that if two authors are close together in the plot below, then their topic distributions are similar.
# 
# In the cell below, we transform the author-topic representation into the t-SNE space. You can increase the `smallest_author` value if you do not want to view all the authors with few documents.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.manifold import TSNE\ntsne = TSNE(n_components=2, random_state=0)\nsmallest_author = 0  # Ignore authors with documents less than this.\nauthors = [model.author2id[a] for a in model.author2id.keys() if len(model.author2doc[a]) >= smallest_author]\n_ = tsne.fit_transform(model.state.gamma[authors, :])  # Result stored in tsne.embedding_')


# 我々は今、プロットをする準備が整いました。
# 
# このノートブックを自分で実行すると、別のグラフが表示されます。 モデルのランダムな初期化は異なるため、結果はある程度異なります。 データの表現がまったく異なる場合や、同じ解釈が若干異なる場合があります。
# 
# プロットが見えない場合は、Jupiterノートでこのチュートリアルを表示している可能性があります。 http://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb　の代わりにnbviewerで表示してください。
# 
# We are now ready to make the plot.
# 
# Note that if you run this notebook yourself, you will see a different graph. The random initialization of the model will be different, and the result will thus be different to some degree. You may find an entirely different representation of the data, or it may show the same interpretation slightly differently.
# 
# If you can't see the plot, you are probably viewing this tutorial in a Jupyter Notebook. View it in an nbviewer instead at http://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb.

# In[ ]:


# Tell Bokeh to display plots inside the notebook.
from bokeh.io import output_notebook
output_notebook()


# In[ ]:


from bokeh.models import HoverTool
from bokeh.plotting import figure, show, ColumnDataSource

x = tsne.embedding_[:, 0]
y = tsne.embedding_[:, 1]
author_names = [model.id2author[a] for a in authors]

# Radius of each point corresponds to the number of documents attributed to that author.
scale = 0.1
author_sizes = [len(model.author2doc[a]) for a in author_names]
radii = [size * scale for size in author_sizes]

source = ColumnDataSource(
        data=dict(
            x=x,
            y=y,
            author_names=author_names,
            author_sizes=author_sizes,
            radii=radii,
        )
    )

# Add author names and sizes to mouse-over info.
hover = HoverTool(
        tooltips=[
        ("author", "@author_names"),
        ("size", "@author_sizes"),
        ]
    )

p = figure(tools=[hover, 'crosshair,pan,wheel_zoom,box_zoom,reset,save,lasso_select'])
p.scatter('x', 'y', radius='radii', source=source, fill_alpha=0.6, line_color=None)
show(p)


# 上のプロットの円は個々の著者であり、そのサイズは対応する著者に帰属する文書の数を表します。マウスをサークル上に置くと、著者の名前とサイズがわかります。著者の大きなクラスターは関心のある重複を反映する傾向があります。
# 
# このモデルは、重複する著者を密接させる傾向があることがわかります。たとえば、Terrence J. SejnowkiとT. J. Sejnowskiは同じ人物で、そのベクトルは同じ場所にあります（プロットの$（-10、-10）$を参照）。
# 
# 約$（〜15、-10）$では、Christof KochやJames M. Bowerのような神経科学者の集団があります。
# 
# 先に述べたように、「物体認識」のトピックはSejnowskiに割り当てられました。 Sejnoskiの近隣のピーター・ダーヤン（Peter Dayan）のような他の作家の話題を取り上げると、同じ話題があります。さらに、このクラスターは上記の「神経科学」クラスターに近く、このトピックは脳における視覚知覚に関するものであることがさらに分かります。
# 
# 他のクラスターには、約$（-5,8）$の強化学習クラスターと約$（8、-12）$のベイジアンモデルクラスターがあります。
# 
# The circles in the plot above are individual authors, and their sizes represent the number of documents attributed to the corresponding author. Hovering your mouse over the circles will tell you the name of the authors and their sizes. Large clusters of authors tend to reflect some overlap in interest. 
# 
# We see that the model tends to put duplicate authors close together. For example, Terrence J. Sejnowki and T. J. Sejnowski are the same person, and their vectors end up in the same place (see about $(-10, -10)$ in the plot).
# 
# At about $(-15, -10)$ we have a cluster of neuroscientists like Christof Koch and James M. Bower. 
# 
# As discussed earlier, the "object recognition" topic was assigned to Sejnowski. If we get the topics of the other authors in Sejnoski's neighborhood, like Peter Dayan, we also get this same topic. Furthermore, we see that this cluster is close to the "neuroscience" cluster discussed above, which is further indication that this topic is about visual perception in the brain.
# 
# Other clusters include a reinforcement learning cluster at about $(-5, 8)$, and a Bayesian modelling cluster at about $(8, -12)$.
# 
# #### Similarity queries
# 
# このセクションでは、著者の名前をとり、最も似ている著者を生み出すシステムを構築しようとしています。 この機能性は、情報検索（すなわち、何らかの検索エンジン）の構成要素として、または作成者予測システム、すなわち、ラベル付けされていない文書を取り上げ、それを書いた著者を予測するシステムで使用することができる。
# 
# 著者 - トピック空間で最も近いベクトルを検索するだけでよい。 この意味で、アプローチは上記のt-SNEプロットと似ています。
# 
# 以下に、Gensimの組み込みの類似性フレームワークを使用した類似性クエリを示します。
# 
# In this section, we are going to set up a system that takes the name of an author and yields the authors that are most similar. This functionality can be used as a component in an information retrieval (i.e. a search engine of some kind), or in an author prediction system, i.e. a system that takes an unlabelled document and predicts the author(s) that wrote it.
# 
# We simply need to search for the closest vector in the author-topic space. In this sense, the approach is similar to the t-SNE plot above.
# 
# Below we illustrate a similarity query using a built-in similarity framework in Gensim.

# In[ ]:


from gensim.similarities import MatrixSimilarity

# Generate a similarity object for the transformed corpus.
index = MatrixSimilarity(model[list(model.id2author.values())])

# Get similarities to some author.
author_name = 'YannLeCun'
sims = index[model[author_name]]


# しかし、このフレームワークはコサイン距離を使用しますが、Hellinger距離を使用します。 Hellinger距離は、2つの確率分布間の距離（すなわち不一致）を測定する自然な方法である。 その離散バージョンは、
# $$
# H(p, q) = \frac{1}{\sqrt{2}} \sqrt{\sum_{i=1}^K (\sqrt{p_i} - \sqrt{q_i})^2},
# $$
# ここで、$ p $と$ q $は、2人の異なる著者の両方のトピック配布です。 類似度は次のように定義します。
# $$
# S(p, q) = \frac{1}{1 + H(p, q)}.
# $$
# 下のセルでは、Hellinger距離に基づいて類似クエリを実行するために必要なものすべてを準備します。
# 
# However, this framework uses the cosine distance, but we want to use the Hellinger distance. The Hellinger distance is a natural way of measuring the distance (i.e. dis-similarity) between two probability distributions. Its discrete version is defined as
# $$
# H(p, q) = \frac{1}{\sqrt{2}} \sqrt{\sum_{i=1}^K (\sqrt{p_i} - \sqrt{q_i})^2},
# $$
# 
# where $p$ and $q$ are both topic distributions for two different authors. We define the similarity as
# $$
# S(p, q) = \frac{1}{1 + H(p, q)}.
# $$
# 
# In the cell below, we prepare everything we need to perform similarity queries based on the Hellinger distance.

# In[ ]:


# Make a function that returns similarities based on the Hellinger distance.

from gensim import matutils
import pandas as pd

# Make a list of all the author-topic distributions.
author_vecs = [model.get_author_topics(author) for author in model.id2author.values()]

def similarity(vec1, vec2):
    '''Get similarity between two vectors'''
    dist = matutils.hellinger(matutils.sparse2full(vec1, model.num_topics),                               matutils.sparse2full(vec2, model.num_topics))
    sim = 1.0 / (1.0 + dist)
    return sim

def get_sims(vec):
    '''Get similarity of vector to all authors.'''
    sims = [similarity(vec, vec2) for vec2 in author_vecs]
    return sims

def get_table(name, top_n=10, smallest_author=1):
    '''
    Get table with similarities, author names, and author sizes.
    Return `top_n` authors as a dataframe.
    
    '''
    
    # Get similarities.
    sims = get_sims(model.get_author_topics(name))

    # Arrange author names, similarities, and author sizes in a list of tuples.
    table = []
    for elem in enumerate(sims):
        author_name = model.id2author[elem[0]]
        sim = elem[1]
        author_size = len(model.author2doc[author_name])
        if author_size >= smallest_author:
            table.append((author_name, sim, author_size))
            
    # Make dataframe and retrieve top authors.
    df = pd.DataFrame(table, columns=['Author', 'Score', 'Size'])
    df = df.sort_values('Score', ascending=False)[:top_n]
    
    return df


# 今、特定の作者に最も類似した著者を見つけることができます。 我々はパンダのライブラリを使用して、見栄えの良い表で結果を出力します。
# 
# Now we can find the most similar authors to some particular author. We use the Pandas library to print the results in a nice looking tables.

# In[ ]:


get_table('YannLeCun')


# 前と同じように、著者の最小サイズを指定することができます。
# 
# As before, we can specify the minimum author size.

# In[ ]:


get_table('JamesM.Bower', smallest_author=3)


# ### Serialized corpora
# 
# `AuthorTopicModel`クラスは、直列化されたコーパス、すなわちメモリではなくハードドライブに格納されたコーパスを受け入れます。これは通常、コーパスが大きすぎてメモリに収まらない場合に実行されます。ただし、この機能にはいくつかの注意点がありますが、ここで説明します。これらの警告は、この機能を理想よりも低くしているため、将来的には改善される可能性があります。
# 
# シリアライズされたコーパスを使用するつもりがない場合は、このセクションを読む必要はありません。
# 
# 以下では、説明の後に例と要約を示す。
# 
# コーパスがシリアライズされている場合、ユーザーは `serialized = True`を指定する必要があります。入力コーパスは、任意のタイプの反復可能または生成元になります。
# 
# モデルは入力コーパスを受け取り、 `MmCorpus`フォーマットでシリアル化します。これは[Gensimでサポートされています]（https://radimrehurek.com/gensim/corpora/mmcorpus.html）です。
# 
# モデルでは、すべての入力ドキュメントをシリアル化するパスを指定する必要があります（例： `serialization_path = '/ tmp / model_serializer.mm'`)。重要なデータを誤って上書きしないようにするため、 `serialization_path`にファイルがすでに存在する場合、モデルはエラーを発生させます。この場合、別のパスを選択するか、古いファイルを削除してください。
# 
# 新しいデータを訓練し、 `model.update（corpus、author2doc）`を呼び出す場合は、古いデータと新しいデータをすべて再直列化する必要があります。これはもちろん計算的に要求の厳しいものである可能性があるので、必要な場合にのみ*これを行うことをお勧めします。つまり、新しいドキュメントごとにモデルを更新するのではなく、できるだけ多くの新しいデータが更新されるまで待ってください。
# 
# The `AuthorTopicModel` class accepts serialized corpora, that is, corpora that are stored on the hard-drive rather than in memory. This is usually done when the corpus is too big to fit in memory. There are, however, some caveats to this functionality, which we will discuss here. As these caveats make this functionality less than ideal, it may be improved in the future.
# 
# It is not necessary to read this section if you don't intend to use serialized corpora.
# 
# In the following, an explanation, followed by an example and a summarization will be given.
# 
# If the corpus is serialized, the user must specify `serialized=True`. Any input corpus can then be any type of iterable or generator.
# 
# The model will then take the input corpus and serialize it in the `MmCorpus` format, which is [supported in Gensim](https://radimrehurek.com/gensim/corpora/mmcorpus.html).
# 
# The user must specify the path where the model should serialize all input documents, for example `serialization_path='/tmp/model_serializer.mm'`. To avoid accidentally overwriting some important data, the model will raise an error if there already exists a file at `serialization_path`; in this case, either choose another path, or delete the old file.
# 
# When you want to train on new data, and call `model.update(corpus, author2doc)`, all the old data and the new data have to be re-serialized. This can of course be quite computationally demanding, so it is recommended that you do this *only* when necessary; that is, wait until you have as much new data as possible to update, rather than updating the model for every new document.

# In[ ]:


get_ipython().run_line_magic('time', "model_ser = AuthorTopicModel(corpus=corpus, num_topics=10, id2word=dictionary.id2token,                                author2doc=author2doc, random_state=1, serialized=True,                                serialization_path='/tmp/model_serialization.mm')")


# In[ ]:


# Delete the file, once you're done using it.
import os
os.remove('/tmp/model_serialization.mm')


# 要約すると、シリアライズされたコーパスを使用する場合：
# * `serialized = True`を設定します。
# * `serialization_path`をまだファイルが入っていないパスに設定します。
# * `model.update（corpus、author2doc）`を呼び出す前に、たくさんのデータがあるまで待ってください。
# *完了したら、それがもはや必要でないなら `serialization_path`でファイルを削除してください。
# 
# In summary, when using serialized corpora:
# * Set `serialized=True`.
# * Set `serialization_path` to a path that doesn't already contain a file.
# * Wait until you have lots of data before you call `model.update(corpus, author2doc)`.
# * When done, delete the file at `serialization_path` if it's not needed anymore.

# ## What to try next
# 
# [StackExchange data dump](https://archive.org/details/stackexchange)のデータセットの1つでモデルを試してください。 記事のタグを著者として扱い、 "タグ - トピック"モデルを鍛えることができます。 統計から料理、哲学まで、さまざまなカテゴリがありますので、好きなものを選ぶことができます。 このデータセットのタグを使用する[Kaggle competition](https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags)を試すことさえできます。
# 
# Try the model on one of the datasets in the [StackExchange data dump](https://archive.org/details/stackexchange). You can treat the tags on the posts as authors and train a "tag-topic" model. There are many different categories, from statistics to cooking to philosophy, so you can pick on that you like. You can even try your hand at a [Kaggle competition](https://www.kaggle.com/c/transfer-learning-on-stack-exchange-tags) that uses tags in this dataset.
# 
