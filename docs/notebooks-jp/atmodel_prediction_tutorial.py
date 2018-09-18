
# coding: utf-8

# # Authorship prediction with the author-topic model

# In this tutorial, you will learn how to use the author-topic model in Gensim for authorship prediction, based on the topic distributions and mesuring their similarity.
# We will train the author-topic model on a Reuters dataset, which contains 50 authors, each with 50 documents for trianing and another 50 documents for testing: https://archive.ics.uci.edu/ml/datasets/Reuter_50_50 .

# If you wish to learn more about the Author-topic model and LDA and how to train them, you should check out these tutorials beforehand. A lot of the preprocessing and configuration here has been done using their example:
# * [LDA training tips](https://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/lda_training_tips.ipynb)
# * [Training the author-topic model](https://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb)

# > **NOTE:**
# >
# > To run this tutorial on your own, install Jupyter, Gensim, SpaCy, Scikit-Learn, Bokeh and Pandas, e.g. using pip:
# >
# > `pip install jupyter gensim spacy sklearn bokeh pandas`
# >
# > Note that you need to download some data for SpaCy using `python -m spacy.en.download`.
# >
# > Download the notebook at https://github.com/RaRe-Technologies/gensim/tree/develop/docs/notebooks/atmodel_prediction_tutorial.ipynb.
# 

# Predicting the author of a document is a difficult task, where current approaches usually turn to neural networks. These base a lot of their predictions on learing stylistic and syntactic preferences of the authors and also other features which help rather identify the author. 
# 
# In our case, we first model the domain knowledge of a certain author, based on what the author writes about. We do this by calculating the topic distributions for each author using the author-topic model.
# After that, we perform the [new author inference](https://github.com/RaRe-Technologies/gensim/pull/1766) on the held-out subset. This again calculates a topic distribution for this new unknown author. 
# In order to perform the prediction, we find out of all known authors, the most similar one to the new unknown. Mathematically speaking, we find the author, whose topic distribution is the closest to the topic distribution of the new author, by a certrain distrance function or metric. 
# Here we explore the [Hellinger distance](https://en.wikipedia.org/wiki/Hellinger_distance) for the measuring the distance between two discrete multinomial topic distributions.

# We start off by downloading the dataset. You can do it manually using the aforementioned link, or run the following code cell.

# In[ ]:


get_ipython().system('wget -O - "https://archive.ics.uci.edu/ml/machine-learning-databases/00217/C50.zip" > /tmp/C50.zip')


# In[ ]:


import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


# In[ ]:


import zipfile

filename = '/tmp/C50.zip'

zip_ref = zipfile.ZipFile(filename, 'r')
zip_ref.extractall("/tmp/")
zip_ref.close()


# We wrap all the preprocessing steps, that you can find more about in the [author-topic notebook](https://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb) , in one fucntion so that we are able to iterate over different preprocessing parameters.

# In[ ]:


import os, re, io
def preprocess_docs(data_dir):
    doc_ids = []
    author2doc = {}
    docs = []
    
    folders = os.listdir(data_dir)  # List of filenames.
    for authorname in folders:
        files = file = os.listdir(data_dir + '/' + authorname)
        for filen in files:
            (idx1, idx2) = re.search('[0-9]+', filen).span()  # Matches the indexes of the start end end of the ID.
            if not author2doc.get(authorname):
                # This is a new author.
                author2doc[authorname] = []
            doc_id = str(int(filen[idx1:idx2]))
            doc_ids.append(doc_id)
            author2doc[authorname].extend([doc_id])

            # Read document text.
            # Note: ignoring characters that cause encoding errors.
            with io.open(data_dir + '/' + authorname + '/' + filen, errors='ignore', encoding='utf-8') as fid:
                txt = fid.read()

            # Replace any whitespace (newline, tabs, etc.) by a single space.
            txt = re.sub('\s', ' ', txt)
            docs.append(txt)
            
    doc_id_dict = dict(zip(doc_ids, range(len(doc_ids))))
    # Replace dataset IDs by integer IDs.
    for a, a_doc_ids in author2doc.items():
        for i, doc_id in enumerate(a_doc_ids):
            author2doc[a][i] = doc_id_dict[doc_id]
    import spacy
    nlp = spacy.load('en')
    
    get_ipython().run_line_magic('time', '')
    processed_docs = []
    for doc in nlp.pipe(docs, n_threads=4, batch_size=100):
        # Process document using Spacy NLP pipeline.

        ents = doc.ents  # Named entities.

        # Keep only words (no numbers, no punctuation).
        # Lemmatize tokens, remove punctuation and remove stopwords.
        doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

        # Remove common words from a stopword list.
        #doc = [token for token in doc if token not in STOPWORDS]

        # Add named entities, but only if they are a compound of more than word.
        doc.extend([str(entity) for entity in ents if len(entity) > 1])
        processed_docs.append(doc)
    docs = processed_docs
    del processed_docs
    
    # Compute bigrams.

    from gensim.models import Phrases

    # Add bigrams and trigrams to docs (only ones that appear 20 times or more).
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    return docs, author2doc


# We create the corpus of the train and test data using two separate functions, since each corpus is tied to a certain dictionary which maps the words to their ids. Also in order to create the test corpus, we use the dictionary from the train data, since the trained model has have the same id2word reference as the new test data. Otherwise token with id 1 from the test data wont't mean the same as the trained upon token with id 1 in the model.

# In[ ]:


def create_corpus_dictionary(docs, max_freq=0.5, min_wordcount=20):
    # Create a dictionary representation of the documents, and filter out frequent and rare words.
    from gensim.corpora import Dictionary
    dictionary = Dictionary(docs)

    # Remove rare and common tokens.
    # Filter out words that occur too frequently or too rarely.
    max_freq = max_freq
    min_wordcount = min_wordcount
    dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)

    _ = dictionary[0]  # This sort of "initializes" dictionary.id2token.

    # Vectorize data.
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    return corpus, dictionary

def create_test_corpus(train_dictionary, docs):
    # Create test corpus using the dictionary from the train data.
    return [train_dictionary.doc2bow(doc) for doc in docs]


# For our first training, we specify that we want the parameters max_freq and min_wordcoun to be 50 and 20, as proposed by the original notebook tutorial. We will find out if this configuration is good enough for us.

# In[ ]:


traindata_dir = "/tmp/C50train"
train_docs, train_author2doc = preprocess_docs(traindata_dir)
train_corpus_50_20, train_dictionary_50_20 = create_corpus_dictionary(train_docs, 0.5, 20)


# In[ ]:


print('Number of unique tokens: %d' % len(train_dictionary_50_20))


# In[ ]:


testdata_dir = "/tmp/C50test"
test_docs, test_author2doc = preprocess_docs(testdata_dir)
test_corpus_50_20 = create_test_corpus(train_dictionary_50_20, test_docs)


# We wrap the model training also in a function, in order to, again, be able to iterate over different parametrizations.

# In[ ]:


def train_model(corpus, author2doc, dictionary, num_topics=20, eval_every=0, iterations=50, passes=20):
    from gensim.models import AuthorTopicModel
    
    model = AuthorTopicModel(corpus=corpus, num_topics=num_topics, id2word=dictionary.id2token,                     author2doc=author2doc, chunksize=2500, passes=passes,                     eval_every=eval_every, iterations=iterations, random_state=1)
    top_topics = model.top_topics(corpus)
    tc = sum([t[1] for t in top_topics]) 
    print(tc / num_topics)
    return model


# In[ ]:


# NOTE: Author of the logic of this function is the Olavur Mortensen, from his notebook tutorial.

def predict_author(new_doc, atmodel, top_n=10, smallest_author=1):
    from gensim import matutils
    import pandas as pd

    def similarity(vec1, vec2):
        '''Get similarity between two vectors'''
        dist = matutils.hellinger(matutils.sparse2full(vec1, atmodel.num_topics),                                   matutils.sparse2full(vec2, atmodel.num_topics))
        sim = 1.0 / (1.0 + dist)
        return sim

    def get_sims(vec):
        '''Get similarity of vector to all authors.'''
        sims = [similarity(vec, vec2) for vec2 in author_vecs]
        return sims

    author_vecs = [atmodel.get_author_topics(author) for author in atmodel.id2author.values()]
    new_doc_topics = atmodel.get_new_author_topics(new_doc)
    # Get similarities.
    sims = get_sims(new_doc_topics)

    # Arrange author names, similarities, and author sizes in a list of tuples.
    table = []
    for elem in enumerate(sims):
        author_name = atmodel.id2author[elem[0]]
        sim = elem[1]
        author_size = len(atmodel.author2doc[author_name])
        if author_size >= smallest_author:
            table.append((author_name, sim, author_size))

    # Make dataframe and retrieve top authors.
    df = pd.DataFrame(table, columns=['Author', 'Score', 'Size'])
    df = df.sort_values('Score', ascending=False)[:top_n]

    return df


# We define a custom function, which measures the prediction accuracy, following the [precision at k](https://en.wikipedia.org/wiki/Information_retrieval#Precision_at_K) principle. We parametrize the accuracy by a parameter k, k=1 meaning we need an exact match in order to be accurate, k=5 meaning our prediction has be in the top 5 results, ordered by similarity.

# In[ ]:


def prediction_accuracy(test_author2doc, test_corpus, model, k=5):

    print("Precision@k: top_n={}".format(k))
    matches=0
    tries = 0
    for author in test_author2doc:
        author_id = model.author2id[author]
        for doc_id in test_author2doc[author]:
            predicted_authors = predict_author(test_corpus[doc_id:doc_id+1], atmodel=model, top_n=k)
            tries = tries+1
            if author_id in predicted_authors["Author"]:
                matches=matches+1

    accuracy = matches/tries
    print("Prediction accuracy: {}".format(accuracy))
    return accuracy, k


# In[ ]:


def plot_accuracy(scores1, label1, scores2=None, label2=None):
    
    import matplotlib.pyplot as plt
    s = [score*100 for score in scores1.values()]
    t = list(scores1.keys())

    plt.plot(t, s, "b-", label=label1)
    plt.plot(t, s, "r^", label=label1+" data points")
    
    if scores2 is not None:
        s2 = [score*100 for score in scores2.values()]
        plt.plot(t, s2, label=label2)
        plt.plot(t, s2, "o", label=label2+" data points")
        
    plt.legend(loc="lower right")

    plt.xlabel('parameter k')
    plt.ylabel('prediction accuracy')
    plt.title('Precision at k')
    plt.xticks(t)
    plt.grid(True)
    plt.yticks([30,40,50,60,70,80,90,100])
    plt.axis([0, 11, 30, 100])
    plt.show()


# We calculate the accuracy for a range of values for k=[1,2,3,4,5,6,8,10] and plot how exactly the prediction accuracy naturally rises with higher k.

# In[ ]:


atmodel_standard = train_model(train_corpus_50_20, train_author2doc, train_dictionary_50_20)


# We run our first training and observe that the **passes** and **iterations** parameters are set high enough, so that the model converges.
# 
# 07:47:24 INFO:PROGRESS: pass 15, at document #2500/2500
# 
# 07:47:24 DEBUG:performing inference on a chunk of 2500 documents 
# 
# 07:47:27 DEBUG:2500/2500 documents converged within 50 iterations 
# 
# Tells us that the model indeed conveges well.

# In[ ]:


accuracy_scores_20topic={}
for i in [1,2,3,4,5,6,8,10]:
    accuracy, k = prediction_accuracy(test_author2doc, test_corpus_50_20, atmodel_standard, k=i)
    accuracy_scores_20topic[k] = accuracy
    
plot_accuracy(scores1=accuracy_scores_20topic, label1="20 topics")


# This is a rather poor accuracy performace. We increase the number of topic to 100.

# In[ ]:


atmodel_100topics = train_model(train_corpus_50_20, train_author2doc, train_dictionary_50_20, num_topics=100, eval_every=0, iterations=50, passes=10)


# In[ ]:


accuracy_scores_100topic={}
for i in [1,2,3,4,5,6,8,10]:
    accuracy, k = prediction_accuracy(test_author2doc, test_corpus_50_20, atmodel_100topics, k=i)
    accuracy_scores_100topic[k] = accuracy
    
plot_accuracy(scores1=accuracy_scores_20topic, label1="20 topics", scores2=accuracy_scores_100topic, label2="100 topics")


# The 100-topic model is much more accurate than the 20-topic model. We continue to increase the topic until convergence.

# In[ ]:


atmodel_150topics = train_model(train_corpus_50_20, train_author2doc, train_dictionary_50_20, num_topics=150, eval_every=0, iterations=50, passes=15)


# In[ ]:


accuracy_scores_150topic={}
for i in [1,2,3,4,5,6,8,10]:
    accuracy, k = prediction_accuracy(test_author2doc, test_corpus_50_20, atmodel_150topics, k=i)
    accuracy_scores_150topic[k] = accuracy
    
plot_accuracy(scores1=accuracy_scores_100topic, label1="100 topics", scores2=accuracy_scores_150topic, label2="150 topics")


# The 150-topic model is also slightly better, especially in the lower end of k. But we clearly see convergence. We try with 200 topic to be sure.

# In[ ]:


atmodel_200topics = train_model(train_corpus_50_20, train_author2doc, train_dictionary_50_20, num_topics=200, eval_every=0, iterations=50, passes=15)


# In[ ]:


accuracy_scores_200topic={}
for i in [1,2,3,4,5,6,8,10]:
    accuracy, k = prediction_accuracy(test_author2doc, test_corpus_50_20, atmodel_200topics, k=i)
    accuracy_scores_200topic[k] = accuracy
    
plot_accuracy(scores1=accuracy_scores_150topic, label1="150 topics", scores2=accuracy_scores_200topic, label2="200 topics")


# The 200-topic seems to be performing a bit better for lower k, might be due to a slight overrepresentation with high topic number. So let us stop here with the topic number increase and focus some more on the dictionary. We choose either one of the models.
# Currently we are filtering out tokens, that appear in more 50% of all documents and no more than 20 times overall, which drastically decreaces the size of our dictionary. 
# We know about this dataset, that the underlying topic are not so diverse and are structed around corporate/industrial topic class. Thus it makes sense to increase the dictionary by filtering less tokens.

# We set the parameters set max_freq=25%, min_wordcount=10

# In[ ]:


train_corpus_25_10, train_dictionary_25_10 = create_corpus_dictionary(train_docs, 0.25, 10)


# In[ ]:


test_corpus_25_10 = create_test_corpus(train_dictionary_25_10, test_docs)


# In[ ]:


print('Number of unique tokens: %d' % len(train_dictionary_25_10))


# We now have now nearly doubled the tokens. Let's train and evaluate.

# In[ ]:


atmodel_150topics_25_10 = train_model(train_corpus_25_10, train_author2doc, train_dictionary_25_10, num_topics=150, eval_every=0, iterations=50, passes=15)


# In[ ]:


accuracy_scores_150topic_25_10={}
for i in [1,2,3,4,5,6,8,10]:
    accuracy, k = prediction_accuracy(test_author2doc, test_corpus_25_10, atmodel_150topics_25_10, k=i)
    accuracy_scores_150topic_25_10[k] = accuracy
    
plot_accuracy(scores1=accuracy_scores_150topic_25_10, label1="150 topics, max_freq=25%, min_wordcount=10", scores2=accuracy_scores_150topic, label2="150 topics, standard")


# The results seem rather ambigious and do not show a clear trend. Which is why we would stop here for the iterations.
