
# coding: utf-8

# # Word2Vec Tutorial
# 
# In case you missed the buzz, word2vec is a widely featured as a member of the “new wave” of machine learning algorithms based on neural networks, commonly referred to as "deep learning" (though word2vec itself is rather shallow). Using large amounts of unannotated plain text, word2vec learns relationships between words automatically. The output are vectors, one vector per word, with remarkable linear relationships that allow us to do things like vec(“king”) – vec(“man”) + vec(“woman”) =~ vec(“queen”), or vec(“Montreal Canadiens”) – vec(“Montreal”) + vec(“Toronto”) resembles the vector for “Toronto Maple Leafs”.
# 
# Word2vec is very useful in [automatic text tagging](https://github.com/RaRe-Technologies/movie-plots-by-genre), recommender systems and machine translation.
# 
# Check out an [online word2vec demo](http://radimrehurek.com/2014/02/word2vec-tutorial/#app) where you can try this vector algebra for yourself. That demo runs `word2vec` on the Google News dataset, of **about 100 billion words**.
# 
# ## This tutorial
# 
# In this tutorial you will learn how to train and evaluate word2vec models on your business data.
# 

# ## Preparing the Input
# Starting from the beginning, gensim’s `word2vec` expects a sequence of sentences as its input. Each sentence is a list of words (utf8 strings):

# In[ ]:


# import modules & set up logging
import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[ ]:


sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1)


# Keeping the input as a Python built-in list is convenient, but can use up a lot of RAM when the input is large.
# 
# Gensim only requires that the input must provide sentences sequentially, when iterated over. No need to keep everything in RAM: we can provide one sentence, process it, forget it, load another sentence…
# 
# For example, if our input is strewn across several files on disk, with one sentence per line, then instead of loading everything into an in-memory list, we can process the input file by file, line by line:

# In[ ]:


# create some toy data to use with the following example
import smart_open, os

if not os.path.exists('./data/'):
    os.makedirs('./data/')

filenames = ['./data/f1.txt', './data/f2.txt']

for i, fname in enumerate(filenames):
    with smart_open.smart_open(fname, 'w') as fout:
        for line in sentences[i]:
            fout.write(line + '\n')


# In[ ]:


from smart_open import smart_open
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in smart_open(os.path.join(self.dirname, fname), 'rb'):
                yield line.split()


# In[ ]:


sentences = MySentences('./data/') # a memory-friendly iterator
print(list(sentences))


# In[ ]:


# generate the Word2Vec model
model = gensim.models.Word2Vec(sentences, min_count=1)


# In[ ]:


print(model)
print(model.wv.vocab)


# Say we want to further preprocess the words from the files — convert to unicode, lowercase, remove numbers, extract named entities… All of this can be done inside the `MySentences` iterator and `word2vec` doesn’t need to know. All that is required is that the input yields one sentence (list of utf8 words) after another.
# 
# **Note to advanced users:** calling `Word2Vec(sentences, iter=1)` will run **two** passes over the sentences iterator. In general it runs `iter+1` passes. By the way, the default value is `iter=5` to comply with Google's word2vec in C language. 
#   1. The first pass collects words and their frequencies to build an internal dictionary tree structure. 
#   2. The second pass trains the neural model.
# 
# These two passes can also be initiated manually, in case your input stream is non-repeatable (you can only afford one pass), and you’re able to initialize the vocabulary some other way:

# In[ ]:


# build the same model, making the 2 steps explicit
new_model = gensim.models.Word2Vec(min_count=1)  # an empty model, no training
new_model.build_vocab(sentences)                 # can be a non-repeatable, 1-pass generator     
new_model.train(sentences, total_examples=new_model.corpus_count, epochs=new_model.iter)                       
# can be a non-repeatable, 1-pass generator


# In[ ]:


print(new_model)
print(model.wv.vocab)


# ### More data would be nice
# For the following examples, we'll use the [Lee Corpus](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/test/test_data/lee_background.cor) (which you already have if you've installed gensim):

# In[ ]:


# Set file names for train and test data
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
lee_train_file = test_data_dir + 'lee_background.cor'


# In[ ]:


class MyText(object):
    def __iter__(self):
        for line in open(lee_train_file):
            # assume there's one document per line, tokens separated by whitespace
            yield line.lower().split()

sentences = MyText()

print(sentences)


# ## Training
# `Word2Vec` accepts several parameters that affect both training speed and quality.
# 
# ### min_count
# `min_count` is for pruning the internal dictionary. Words that appear only once or twice in a billion-word corpus are probably uninteresting typos and garbage. In addition, there’s not enough data to make any meaningful training on those words, so it’s best to ignore them:

# In[ ]:


# default value of min_count=5
model = gensim.models.Word2Vec(sentences, min_count=10)


# ### size
# `size` is the number of dimensions (N) of the N-dimensional space that gensim Word2Vec maps the words onto.
# 
# Bigger size values require more training data, but can lead to better (more accurate) models. Reasonable values are in the tens to hundreds.

# In[ ]:


# default value of size=100
model = gensim.models.Word2Vec(sentences, size=200)


# ### workers
# `workers`, the last of the major parameters (full list [here](http://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec)) is for training parallelization, to speed up training:

# In[ ]:


# default value of workers=3 (tutorial says 1...)
model = gensim.models.Word2Vec(sentences, workers=4)


# The `workers` parameter only has an effect if you have [Cython](http://cython.org/) installed. Without Cython, you’ll only be able to use one core because of the [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) (and `word2vec` training will be [miserably slow](http://rare-technologies.com/word2vec-in-python-part-two-optimizing/)).

# ## Memory
# At its core, `word2vec` model parameters are stored as matrices (NumPy arrays). Each array is **#vocabulary** (controlled by min_count parameter) times **#size** (size parameter) of floats (single precision aka 4 bytes).
# 
# Three such matrices are held in RAM (work is underway to reduce that number to two, or even one). So if your input contains 100,000 unique words, and you asked for layer `size=200`, the model will require approx. `100,000*200*4*3 bytes = ~229MB`.
# 
# There’s a little extra memory needed for storing the vocabulary tree (100,000 words would take a few megabytes), but unless your words are extremely loooong strings, memory footprint will be dominated by the three matrices above.

# ## Evaluating
# `Word2Vec` training is an unsupervised task, there’s no good way to objectively evaluate the result. Evaluation depends on your end application.
# 
# Google has released their testing set of about 20,000 syntactic and semantic test examples, following the “A is to B as C is to D” task. It is provided in the 'datasets' folder.
# 
# For example a syntactic analogy of comparative type is bad:worse;good:?. There are total of 9 types of syntactic comparisons in the dataset like plural nouns and nouns of opposite meaning.
# 
# The semantic questions contain five types of semantic analogies, such as capital cities (Paris:France;Tokyo:?) or family members (brother:sister;dad:?). 

# Gensim supports the same evaluation set, in exactly the same format:

# In[ ]:


model.accuracy('./datasets/questions-words.txt')


# This `accuracy` takes an 
# [optional parameter](http://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.accuracy) `restrict_vocab` 
# which limits which test examples are to be considered.
# 
# 

# In the December 2016 release of Gensim we added a better way to evaluate semantic similarity.
# 
# By default it uses an academic dataset WS-353 but one can create a dataset specific to your business based on it. It contains word pairs together with human-assigned similarity judgments. It measures the relatedness or co-occurrence of two words. For example, 'coast' and 'shore' are very similar as they appear in the same context. At the same time 'clothes' and 'closet' are less similar because they are related but not interchangeable.

# In[ ]:


model.evaluate_word_pairs(test_data_dir + 'wordsim353.tsv')


# Once again, **good performance on Google's or WS-353 test set doesn’t mean word2vec will work well in your application, or vice versa**. It’s always best to evaluate directly on your intended task. For an example of how to use word2vec in a classifier pipeline, see this [tutorial](https://github.com/RaRe-Technologies/movie-plots-by-genre).

# ## Storing and loading models
# You can store/load models using the standard gensim methods:

# In[ ]:


from tempfile import mkstemp

fs, temp_path = mkstemp("gensim_temp")  # creates a temp file

model.save(temp_path)  # save the model


# In[ ]:


new_model = gensim.models.Word2Vec.load(temp_path)  # open the model


# which uses pickle internally, optionally `mmap`‘ing the model’s internal large NumPy matrices into virtual memory directly from disk files, for inter-process memory sharing.
# 
# In addition, you can load models created by the original C tool, both using its text and binary formats:
# ```
#   model = gensim.models.KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)
#   # using gzipped/bz2 input works too, no need to unzip:
#   model = gensim.models.KeyedVectors.load_word2vec_format('/tmp/vectors.bin.gz', binary=True)
# ```

# ## Online training / Resuming training
# Advanced users can load a model and continue training it with more sentences and [new vocabulary words](online_w2v_tutorial.ipynb):

# In[ ]:


model = gensim.models.Word2Vec.load(temp_path)
more_sentences = [['Advanced', 'users', 'can', 'load', 'a', 'model', 'and', 'continue', 'training', 'it', 'with', 'more', 'sentences']]
model.build_vocab(more_sentences, update=True)
model.train(more_sentences, total_examples=model.corpus_count, epochs=model.iter)

# cleaning up temp
os.close(fs)
os.remove(temp_path)


# You may need to tweak the `total_words` parameter to `train()`, depending on what learning rate decay you want to simulate.
# 
# Note that it’s not possible to resume training with models generated by the C tool, `KeyedVectors.load_word2vec_format()`. You can still use them for querying/similarity, but information vital for training (the vocab tree) is missing there.
# 
# ## Using the model
# `Word2Vec` supports several word similarity tasks out of the box:

# In[ ]:


model.most_similar(positive=['human', 'crime'], negative=['party'], topn=1)


# In[ ]:


model.doesnt_match("input is lunch he sentence cat".split())


# In[ ]:


print(model.similarity('human', 'party'))
print(model.similarity('tree', 'murder'))


# You can get the probability distribution for the center word given the context words as input:

# In[ ]:


print(model.predict_output_word(['emergency', 'beacon', 'received']))


# The results here don't look good because the training corpus is very small. To get meaningful results one needs to train on 500k+ words.

# If you need the raw output vectors in your application, you can access these either on a word-by-word basis:

# In[ ]:


model['tree']  # raw NumPy vector of a word


# …or en-masse as a 2D NumPy matrix from `model.wv.syn0`.

# ## Training Loss Computation
# 
# The parameter `compute_loss` can be used to toggle computation of loss while training the Word2Vec model. The computed loss is stored in the model attribute `running_training_loss` and can be retrieved using the function `get_latest_training_loss` as follows : 

# In[ ]:


# instantiating and training the Word2Vec model
model_with_loss = gensim.models.Word2Vec(sentences, min_count=1, compute_loss=True, hs=0, sg=1, seed=42)

# getting the training loss value
training_loss = model_with_loss.get_latest_training_loss()
print(training_loss)


# ### Benchmarks to see effect of training loss compuation code on training time
# 
# We first download and setup the test data used for getting the benchmarks.

# In[ ]:


input_data_files = []

def setup_input_data():
    # check if test data already present
    if os.path.isfile('./text8') is False:

        # download and decompress 'text8' corpus
        import zipfile
        get_ipython().system(" wget 'http://mattmahoney.net/dc/text8.zip'")
        get_ipython().system(" unzip 'text8.zip'")
    
        # create 1 MB, 10 MB and 50 MB files
        get_ipython().system(' head -c1000000 text8 > text8_1000000')
        get_ipython().system(' head -c10000000 text8 > text8_10000000')
        get_ipython().system(' head -c50000000 text8 > text8_50000000')
                
    # add 25 KB test file
    input_data_files.append(os.path.join(os.getcwd(), '../../gensim/test/test_data/lee_background.cor'))

    # add 1 MB test file
    input_data_files.append(os.path.join(os.getcwd(), 'text8_1000000'))

    # add 10 MB test file
    input_data_files.append(os.path.join(os.getcwd(), 'text8_10000000'))

    # add 50 MB test file
    input_data_files.append(os.path.join(os.getcwd(), 'text8_50000000'))

    # add 100 MB test file
    input_data_files.append(os.path.join(os.getcwd(), 'text8'))

setup_input_data()
print(input_data_files)


# We now compare the training time taken for different combinations of input data and model training parameters like `hs` and `sg`.

# In[ ]:


logging.getLogger().setLevel(logging.ERROR)


# In[ ]:


# using 25 KB and 50 MB files only for generating o/p -> comment next line for using all 5 test files
input_data_files = [input_data_files[0], input_data_files[-2]]
print(input_data_files)

import time
import numpy as np
import pandas as pd

train_time_values = []
seed_val = 42
sg_values = [0, 1]
hs_values = [0, 1]

for data_file in input_data_files:
    data = gensim.models.word2vec.LineSentence(data_file) 
    for sg_val in sg_values:
        for hs_val in hs_values:
            for loss_flag in [True, False]:
                time_taken_list = []
                for i in range(3):
                    start_time = time.time()
                    w2v_model = gensim.models.Word2Vec(data, compute_loss=loss_flag, sg=sg_val, hs=hs_val, seed=seed_val) 
                    time_taken_list.append(time.time() - start_time)

                time_taken_list = np.array(time_taken_list)
                time_mean = np.mean(time_taken_list)
                time_std = np.std(time_taken_list)
                train_time_values.append({'train_data': data_file, 'compute_loss': loss_flag, 'sg': sg_val, 'hs': hs_val, 'mean': time_mean, 'std': time_std})

train_times_table = pd.DataFrame(train_time_values)
train_times_table = train_times_table.sort_values(by=['train_data', 'sg', 'hs', 'compute_loss'], ascending=[False, False, True, False])
print(train_times_table)


# ### Adding Word2Vec "model to dict" method to production pipeline
# Suppose, we still want more performance improvement in production. 
# One good way is to cache all the similar words in a dictionary.
# So that next time when we get the similar query word, we'll search it first in the dict.
# And if it's a hit then we will show the result directly from the dictionary.
# otherwise we will query the word and then cache it so that it doesn't miss next time.

# In[ ]:


logging.getLogger().setLevel(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[ ]:


most_similars_precalc = {word : model.wv.most_similar(word) for word in model.wv.index2word}
for i, (key, value) in enumerate(most_similars_precalc.iteritems()):
    if i==3:
        break
    print key, value


# ### Comparison with and without caching

# for time being lets take 4 words randomly

# In[ ]:


import time
words = ['voted','few','their','around']


# Without caching

# In[ ]:


start = time.time()
for word in words:
    result = model.wv.most_similar(word)
    print(result)
end = time.time()
print(end-start)


# Now with caching

# In[ ]:


start = time.time()
for word in words:
    if 'voted' in most_similars_precalc:
        result = most_similars_precalc[word]
        print(result)
    else:
        result = model.wv.most_similar(word)
        most_similars_precalc[word] = result
        print(result)
    
end = time.time()
print(end-start)


# Clearly you can see the improvement but this difference will be even larger when we take more words in the consideration.

# # Visualising the Word Embeddings

# The word embeddings made by the model can be visualised by reducing dimensionality of the words to 2 dimensions using tSNE.
# 
# Visualisations can be used to notice semantic and syntactic trends in the data.
# 
# Example: Semantic- words like cat, dog, cow, etc. have a tendency to lie close by
#          Syntactic- words like run, running or cut, cutting lie close together.
# Vector relations like vKing - vMan = vQueen - vWoman can also be noticed.
# 
# Additional dependencies : 
# - sklearn
# - numpy
# - plotly
# 
# The function below can be used to plot the embeddings in an ipython notebook.
# It requires the model as the necessary parameter. If you don't have the model, you can load it by
# 
# `model = gensim.models.Word2Vec.load('path/to/model')`
# 
# If you don't want to plot inside a notebook, set the `plot_in_notebook` parameter to `False`.
# 
# Note: the model used for the visualisation is trained on a small corpus. Thus some of the relations might not be so clear
# 
# Beware : This sort dimension reduction comes at the cost of loss of information.

# In[ ]:


from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling

from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

def reduce_dimensions(model, plot_in_notebook = True):

    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = []        # positions in vector space
    labels = []         # keep track of words to label our data again later
    for word in model.wv.vocab:
        vectors.append(model[word])
        labels.append(word)


    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)
    
    # reduce using t-SNE
    vectors = np.asarray(vectors)
    logging.info('starting tSNE dimensionality reduction. This may take some time.')
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
        
    # Create a trace
    trace = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='text',
        text=labels
        )
    
    data = [trace]
    
    logging.info('All done. Plotting.')
    
    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


# In[ ]:


reduce_dimensions(model)


# ## Conclusion
# 
# In this tutorial we learned how to train word2vec models on your custom data and also how to evaluate it. Hope that you too will find this popular tool useful in your Machine Learning tasks!
# 
# ## Links
# 
# 
# Full `word2vec` API docs [here](http://radimrehurek.com/gensim/models/word2vec.html); get [gensim](http://radimrehurek.com/gensim/) here. Original C toolkit and `word2vec` papers by Google [here](https://code.google.com/archive/p/word2vec/).
