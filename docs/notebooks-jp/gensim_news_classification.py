
# coding: utf-8

# # News classification with topic models in gensim
# ニュース記事の分類は、世界中のニュースエージェンシーが大規模に行うタスクです。トピックのモデリングを使って、ニュース記事をスポーツ、技術、政治などの異なるカテゴリに正確に分類する方法を検討します。
# 
# このチュートリアルの目的は、私たちが簡単に理解できるトピックを生成できるトピックモデルを考え出すことです。このようなトピックモデルは、コーパス内の隠れた構造を発見するために使用することができ、ある１つのトピックへのニュース記事のメンバーシップを決定するために使用することもできます。
# 
# このチュートリアルでは、[Lee Background Corpus](http://www.socsci.uci.edu/~mdlee/lee_pincombe_welsh_document.PDF)の短縮バージョンであるリーコーパスを使用します。短縮版は、Australian Broadcasting Corporationのニュースメールサービスから選択された300の文書で構成されています。 2000年から2001年までの主要な記事のテキストで構成されています。
# 
# 付随するスライドは [こちら](https://speakerdeck.com/dsquareindia/pycon-delhi-lightening)にあります。
# 
# News article classification is a task which is performed on a huge scale by news agencies all over the world. We will be looking into how topic modeling can be used to accurately classify news articles into different categories such as sports, technology, politics etc.
# 
# Our aim in this tutorial is to come up with some topic model which can come up with topics that can easily be interpreted by us. Such a topic model can be used to discover hidden structure in the corpus and can also be used to determine the membership of a news article into one of the topics.
# 
# For this tutorial, we will be using the Lee corpus which is a shortened version of the [Lee Background Corpus](http://www.socsci.uci.edu/~mdlee/lee_pincombe_welsh_document.PDF). The shortened version consists of 300 documents selected from the Australian Broadcasting Corporation's news mail service. It consists of texts of headline stories from around the year 2000-2001.
# 
# Accompanying slides can be found [here](https://speakerdeck.com/dsquareindia/pycon-delhi-lightening).
# 
# ### Requirements
# このチュートリアルでは、 [gensim](https://radimrehurek.com/gensim/)を使用してさまざまなトピックモデルを簡単に作成する方法について説明します。
# 以下に、このチュートリアルの依存関係を示します。
#      - Gensim Version> = 0.13.1（ここではトピックコヒーレンスメトリックを広く使用するために推奨します。）
#      - matplotlib
#      - nltk.stopwordsおよびnltk.wordnet
#      - pyLDAVis
# ここでは4つの異なるトピックモデルで遊んでいきます：
#      - LSI（潜在セマンティックインデックス）
#      - HDP（Hierarchical Dirichlet Process）
#      - LDA（Latent Dirichlet Allocation）
#      - LDA（トピックの一貫性を調整して最適なトピック数を見つける）
#      - トピックコヒーレンスメトリクスの助けを借りてLSIとしてのLDA
# 最初に、これらのトピックモデルを既存のデータに合わせて、それぞれを比較し、人間の解釈可能性の点でランク付けする方法を見ていきます。
# 
# すべてgensimで見ることができ、プラグアンドプレイの方法で簡単に使用できます。 Roederらによる[この](http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)論文に基づいて、gensimで新たに追加されたトピックコヒーレンスメトリクスを使用してLDAモデルを調整し、結果として得られるトピックモデルは、既存のトピックモデルと比較されます。
# 
# In this tutorial we look at how different topic models can be easily created using [gensim](https://radimrehurek.com/gensim/).
# Following are the dependencies for this tutorial:
#     - Gensim Version >=0.13.1 would be preferred since we will be using topic coherence metrics extensively here.
#     - matplotlib
#     - nltk.stopwords and nltk.wordnet
#     - pyLDAVis
# We will be playing around with 4 different topic models here:
#     - LSI (Latent Semantic Indexing)
#     - HDP (Hierarchical Dirichlet Process)
#     - LDA (Latent Dirichlet Allocation)
#     - LDA (tweaked with topic coherence to find optimal number of topics) and
#     - LDA as LSI with the help of topic coherence metrics
# First we'll fit those topic models on our existing data then we'll compare each against the other and see how they rank in terms of human interpretability.
# 
# All can be found in gensim and can be easily used in a plug-and-play fashion. We will tinker with the LDA model using the newly added topic coherence metrics in gensim based on [this](http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) paper by Roeder et al and see how the resulting topic model compares with the exsisting ones.

# In[ ]:

import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

import nltk
nltk.download('stopwords') # Let's make sure the 'stopword' package is downloaded & updated
nltk.download('wordnet') # Let's also download wordnet, which will be used for lemmatization

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
from pprint import pprint
from smart_open import smart_open

get_ipython().magic('matplotlib inline')


# In[ ]:

test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'


# コーパスの分析。
#      - 最初の文書は、ニューサウスウェールズ州で発生した暴動について語っています。
#      - ２つ目の文書は、カシミールにおけるインドとパキスタンの紛争について語っています。
#      - ３つ目の文書は、ニューサウスウェールズ州の道路事故に関する記事です。
#      - ４つ目の文書は、アルゼンチンの経済・政治危機について語っています。
#      - 最後は、シドニーの病院で助産婦による薬の使用について語っています。
# 私たちの最後のトピックモデルは、わかりやすく簡単な要約を出すことができるキーワードを私たちに与えてくれるはずです。 これがなければ、トピックモデルはあまり実用的ではありません。
# 
# Analysing our corpus.
#     - The first document talks about a bushfire that had occured in New South Wales.
#     - The second talks about conflict between India and Pakistan in Kashmir.
#     - The third talks about road accidents in the New South Wales area.
#     - The fourth one talks about Argentina's economic and political crisis during that time.
#     - The last one talks about the use of drugs by midwives in a Sydney hospital.
# Our final topic model should be giving us keywords which we can easily interpret and make a small summary out of. Without this the topic model cannot be of much practical use.

# In[ ]:

with smart_open(lee_train_file, 'rb') as f:
    for n, l in enumerate(f):
        if n < 5:
            print([l])


# In[ ]:

def build_texts(fname):
    """
    Function to build tokenized texts from file
    
    Parameters:
    ----------
    fname: File to be read
    
    Returns:
    -------
    yields preprocessed line
    """
    with smart_open(fname, 'rb') as f:
        for line in f:
            yield gensim.utils.simple_preprocess(line, deacc=True, min_len=3)


# In[ ]:

train_texts = list(build_texts(lee_train_file))


# In[ ]:

len(train_texts)


# ### Preprocessing our data. Remember: Garbage In Garbage Out
#                                         "NLP is 80% preprocessing."
#                                                                 -Lev Konstantinovskiy
# これは、良いトピックモデリングシステムを設定する際の最も重要な一歩です。前処理がうまくいかない場合、アルゴリズムは多くのノイズを与えるのであまり効果がありません。このチュートリアルでは、次の手順をこの順に各行に適用してノイズをフィルタリングします。
# 1. NLTKの英語ストップワードデータセットを使用したストップワードの削除。
# 2. gensimの[Phrases](https://radimrehurek.com/gensim/models/phrases.html)を使用して、Bigramコロケーション検出（頻繁に共存するトークン）を行います。これは、コーパス内の隠れた構造を見つける最初の試みです。トリグラムコロケーション検出を試みることもできます。
# 3.名詞を保持するためにのみ使用する（gensimの  [`lemmatize`](https://radimrehurek.com/gensim/utils.html#gensim.utils.lemmatize)）。一般化された後の単語は依然として控えめなので、トピックモデリングの場合には、ステミングよりも一般にレメリズムが優れています。しかし、データがベクトル化に繰り込まれて、表示されることを意図していない場合には、一般的にステミングが好まれる可能性があります。
# 
# 
# This is the single most important step in setting up a good topic modeling system. If the preprocessing is not good, the algorithm can't do much since we would be feeding it a lot of noise. In this tutorial, we will be filtering out the noise using the following steps in this order for each line:
# 1. Stopword removal using NLTK's english stopwords dataset.
# 2. Bigram collocation detection (frequently co-occuring tokens) using gensim's [Phrases](https://radimrehurek.com/gensim/models/phrases.html). This is our first attempt to find some hidden structure in the corpus. You can even try trigram collocation detection.
# 3. Lemmatization (using gensim's [`lemmatize`](https://radimrehurek.com/gensim/utils.html#gensim.utils.lemmatize)) to only keep the nouns. Lemmatization is generally better than stemming in the case of topic modeling since the words after lemmatization still remain understable. However, generally stemming might be preferred if the data is being fed into a vectorizer and isn't intended to be viewed.

# In[ ]:

bigram = gensim.models.Phrases(train_texts)  # for bigram collocation detection


# In[ ]:

bigram[['new', 'york', 'example']]


# In[ ]:

from gensim.utils import lemmatize
from nltk.corpus import stopwords


# In[ ]:

stops = set(stopwords.words('english'))  # nltk stopwords list


# In[ ]:

def process_texts(texts):
    """
    Function to process texts. Following are the steps we take:
    
    1. Stopword Removal.
    2. Collocation detection.
    3. Lemmatization (not stem since stemming can reduce the interpretability).
    
    Parameters:
    ----------
    texts: Tokenized texts.
    
    Returns:
    -------
    texts: Pre-processed tokenized texts.
    """
    texts = [[word for word in line if word not in stops] for line in texts]
    texts = [bigram[line] for line in texts]
    
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    texts = [[word for word in lemmatizer.lemmatize(' '.join(line), pos='v').split()] for line in texts]
    return texts


# In[ ]:

train_texts = process_texts(train_texts)
train_texts[5:6]


# 辞書とコーパスを完成させます。
# 
# Finalising our dictionary and corpus

# In[ ]:

dictionary = Dictionary(train_texts)
corpus = [dictionary.doc2bow(text) for text in train_texts]


# ### Topic modeling with LSI
# これは、トピック自体をランク付けできる点で、有用なトピックモデリングアルゴリズムです。 したがって、ランク付けされた順序でトピックを出力します。 しかし、SVD後の潜在次元の数を決定するためには、 `num_topics`パラメータ（デフォルトでは200に設定）が必要です。
# 
# This is a useful topic modeling algorithm in that it can rank topics by itself. Thus it outputs topics in a ranked order. However it does require a `num_topics` parameter (set to 200 by default) to determine the number of latent dimensions after the SVD.

# In[ ]:

lsimodel = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary)


# In[ ]:

lsimodel.show_topics(num_topics=5)  # Showing only the top 5 topics


# In[ ]:

lsitopics = lsimodel.show_topics(formatted=False)


# ### Topic modeling with [HDP](http://jmlr.csail.mit.edu/proceedings/papers/v15/wang11a/wang11a.pdf)
# HDPモデルは完全に教師なしです。 また、事後推論を通じて必要なトピックの理想的な数を決定することもできます。
# 
# An HDP model is fully unsupervised. It can also determine the ideal number of topics it needs through posterior inference.

# In[ ]:

hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)


# In[ ]:

hdpmodel.show_topics()


# In[ ]:

hdptopics = hdpmodel.show_topics(formatted=False)


# ### Topic modeling using [LDA](https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf)
# これは今日最も人気のあるトピックモデリングアルゴリズムの1つです。 これは、各文書がトピックの混合物であると仮定し、各トピックが単語の混合物であると仮定して生成モデルとなります。 それをよりよく理解するには、David Bleiの[この](https://www.youtube.com/watch?v=DDq3OVp9dNA)講義をご覧ください。 これを初期化するために10のトピックを選択しましょう。
# 
# This is one the most popular topic modeling algorithms today. It is a generative model in that it assumes each document is a mixture of topics and in turn, each topic is a mixture of words. To understand it better you can watch [this](https://www.youtube.com/watch?v=DDq3OVp9dNA) lecture by David Blei. Let's choose 10 topics to initialize this.

# In[ ]:

ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)


# pyLDAvisは、LDAモデルを視覚化するための優れた方法です。 要約すると、円の面積はトピックの優位性を表します。 右側のバーの長さは、特定のトピックの単語のメンバーシップを表します。 pyLDAvisは[この](http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf) 論文に基づいています。
# 
# pyLDAvis is a great way to visualize an LDA model. To summarize in short, the area of the circles represent the prevelance of the topic. The length of the bars on the right represent the membership of a term in a particular topic. pyLDAvis is based on [this](http://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf) paper.

# In[ ]:

import pyLDAvis.gensim


# In[ ]:

pyLDAvis.enable_notebook()


# In[ ]:

pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)


# In[ ]:

ldatopics = ldamodel.show_topics(formatted=False)


# ### Finding out the optimal number of topics
# __Introduction to topic coherence__:
# <img src="https://rare-technologies.com/wp-content/uploads/2016/06/pipeline.png">
# 本質的にトピックのcoherenceは、トピックモデルの人間の解釈可能性を測定します。 トピックモデルを評価するには伝統的に[perplexityが使用されていますが](http://qpleple.com/perplexity-to-evaluate-topic-models/) トピックの一貫性は、人間の解釈可能性をはるかに高く保証したトピックモデルを評価するもう1つの方法です。 したがって、これを使用して、他の多くのユースケースで異なるトピックモデルを比較することができます。 ここで私はトピックのCoherenceを説明した短いブログです：[What is topic coherence?](https://rare-technologies.com/what-is-topic-coherence/)
# 
# Topic coherence in essence measures the human interpretability of a topic model. Traditionally [perplexity has been used](http://qpleple.com/perplexity-to-evaluate-topic-models/) to evaluate topic models however this does not correlate with human annotations at times. Topic coherence is another way to evaluate topic models with a much higher guarantee on human interpretability. Thus this can be used to compare different topic models among many other use-cases. Here's a short blog I wrote explaining topic coherence:
# [What is topic coherence?](https://rare-technologies.com/what-is-topic-coherence/)

# In[ ]:

def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        
    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()
    
    return lm_list, c_v


# In[ ]:

get_ipython().run_cell_magic('time', '', 'lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=train_texts, limit=10)')


# In[ ]:

pyLDAvis.gensim.prepare(lmlist[2], corpus, dictionary)


# In[ ]:

lmtopics = lmlist[5].show_topics(formatted=False)


# ### LDA as LSI

# LDAの問題の1つは、多数のトピックでトレーニングすると、トピックの中でトピックが「失われる」ということです。 私たちが作ることができる最高のLDAモデルから最高の話題を見つけ出すことができるかどうかを見てみましょう。 以下の関数を使用して、LDAモデルの品質を制御することができます。
# 
# One of the problem with LDA is that if we train it on a large number of topics, the topics get "lost" among the numbers. Let us see if we can dig out the best topics from the best LDA model we can produce. The function below can be used to control the quality of the LDA model we produce.

# In[ ]:

def ret_top_model():
    """
    Since LDAmodel is a probabilistic model, it comes up different topics each time we run it. To control the
    quality of the topic model we produce, we can see what the interpretability of the best topic is and keep
    evaluating the topic model until this threshold is crossed. 
    
    Returns:
    -------
    lm: Final evaluated topic model
    top_topics: ranked topics in decreasing order. List of tuples
    """
    top_topics = [(0, 0)]
    while top_topics[0][1] < 0.97:
        lm = LdaModel(corpus=corpus, id2word=dictionary)
        coherence_values = {}
        for n, topic in lm.show_topics(num_topics=-1, formatted=False):
            topic = [word for word, _ in topic]
            cm = CoherenceModel(topics=[topic], texts=train_texts, dictionary=dictionary, window_size=10)
            coherence_values[n] = cm.get_coherence()
        top_topics = sorted(coherence_values.items(), key=operator.itemgetter(1), reverse=True)
    return lm, top_topics


# In[ ]:

lm, top_topics = ret_top_model()


# In[ ]:

print(top_topics[:5])


# ### Inference
# 第1話は__cinema__について、第2は__email malware__について、第3は2000年にオーストラリアの__Larrakia原住民コミュニティ__に与えられた土地についてのものです。__オーストラリアのクリケット__があります。 LSIとしてのLDAは、LDAの中から最善の話題を見つけ出すのにすばらしい努力をしてきました。
# 
# We can clearly see below that the first topic is about __cinema__, second is about __email malware__, third is about the land which was given back to the __Larrakia aboriginal community of Australia__ in 2000. Then there's one about __Australian cricket__. LDA as LSI has worked wonderfully in finding out the best topics from within LDA.

# In[ ]:

pprint([lm.show_topic(topicid) for topicid, c_v in top_topics[:10]])


# In[ ]:

lda_lsi_topics = [[word for word, prob in lm.show_topic(topicid)] for topicid, c_v in top_topics]


# ### Evaluating all the topic models
# トピック単語を思いつくことができるどのようなトピックモデルも、コヒーレンスパイプラインに差し込むことができます。 scikit-learnで作成した[NMF topic model](http://derekgreene.com/nmf-topic/) をプラグインすることもできます。
# 
# Any topic model which can come up with topic terms can be plugged into the coherence pipeline. You can even plug in an [NMF topic model](http://derekgreene.com/nmf-topic/) created with scikit-learn.

# In[ ]:

lsitopics = [[word for word, prob in topic] for topicid, topic in lsitopics]

hdptopics = [[word for word, prob in topic] for topicid, topic in hdptopics]

ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]

lmtopics = [[word for word, prob in topic] for topicid, topic in lmtopics]


# In[ ]:

lsi_coherence = CoherenceModel(topics=lsitopics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()

hdp_coherence = CoherenceModel(topics=hdptopics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()

lda_coherence = CoherenceModel(topics=ldatopics, texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()

lm_coherence = CoherenceModel(topics=lmtopics, texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()

lda_lsi_coherence = CoherenceModel(topics=lda_lsi_topics[:10], texts=train_texts, dictionary=dictionary, window_size=10).get_coherence()


# In[ ]:

def evaluate_bar_graph(coherences, indices):
    """
    Function to plot bar graph.
    
    coherences: list of coherence values
    indices: Indices to be used to mark bars. Length of this and coherences should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label=indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')


# In[ ]:

evaluate_bar_graph([lsi_coherence, hdp_coherence, lda_coherence, lm_coherence, lda_lsi_coherence],
                   ['LSI', 'HDP', 'LDA', 'LDA_Mod', 'LDA_LSI'])


# ### Customizing the topic coherence measure
# 今までは、 `c_v`コヒーレンス測定値だけを使用しました。 `u_mass`、` c_uci`、 `c_npmi`のようなものがあります。 これらのすべては、異なる方法でCoherenceを計算します。 `c_v`は人間の評価と最も一致しますが、` u_mass`よりもはるかに遅くなる可能性があります。
# 
# Till now we only used the `c_v` coherence measure. There are others such as `u_mass`, `c_uci`, `c_npmi`. All of these calculate coherence in a different way. `c_v` is found to be most in line with human ratings but can be much slower than `u_mass` since it uses a sliding window over the texts.

# ### Making your own coherence measure
# `s_one_one`セグメンテーションの代わりに` s_one_pre`を使う `c_uci`を修正しましょう。
# 
# Let's modify `c_uci` to use `s_one_pre` instead of `s_one_one` segmentation

# In[ ]:

from gensim.topic_coherence import (segmentation, probability_estimation,
                                    direct_confirmation_measure, indirect_confirmation_measure,
                                    aggregation)
from gensim.matutils import argsort
from collections import namedtuple


# In[ ]:

make_pipeline = namedtuple('Coherence_Measure', 'seg, prob, conf, aggr')


# In[ ]:

measure = make_pipeline(segmentation.s_one_one,
                        probability_estimation.p_boolean_sliding_window,
                        direct_confirmation_measure.log_ratio_measure,
                        aggregation.arithmetic_mean)


# トピックモデルからトピックを取得するには：
# 
# To get topics out of the topic model:

# In[ ]:

topics = []
for topic in lm.state.get_lambda():
    bestn = argsort(topic, topn=10, reverse=True)
topics.append(bestn)


# __Step 1__: Segmentation

# In[ ]:

# Perform segmentation
segmented_topics = measure.seg(topics)


# __Step 2__: Probability estimation

# In[ ]:

# Since this is a window-based coherence measure we will perform window based prob estimation
per_topic_postings, num_windows = measure.prob(texts=train_texts, segmented_topics=segmented_topics,
                                               dictionary=dictionary, window_size=2)


# __Step 3__: Confirmation Measure

# In[ ]:

confirmed_measures = measure.conf(segmented_topics, per_topic_postings, num_windows, normalize=False)


# __Step 4__: Aggregation

# In[ ]:

print(measure.aggr(confirmed_measures))


# # How this topic model can be used further
# ここで最高のトピックモデルは、ニュース記事の分類のスタンドアロンとして使用できます。 しかし、トピックモデルは、分類子に供給するための次元削減アルゴリズムとしても使用できます。 良い話題モデルは、雑音から信号を効率的に抽出することができ、分類器の性能を向上させることができるはずである。
# 
# The best topic model here can be used as a standalone for news article classification. However a topic model can also be used as a dimensionality reduction algorithm to feed into a classifier. A good topic model should be able to extract the signal from the noise efficiently, hence improving the performance of the classifier.
