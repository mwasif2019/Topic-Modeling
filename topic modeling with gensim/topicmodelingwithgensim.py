#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df=pd.read_csv("20200325_counsel_chat.csv")
df.head(3)


# In[2]:


#data.drop(columns=["Unnamed: 0","questionID","questionLink","topic","therapistInfo","therapistURL","upvotes","views","split"],inplace=True)


# In[3]:


df["qqatext"]=df["questionTitle"]+df["questionText"]+df["answerText"]


# In[4]:


df.qqatext


# In[5]:


df.qqatext[1]


# In[6]:


df.drop(columns=["Unnamed: 0","questionID","questionLink","questionTitle","questionText","topic","answerText","therapistInfo","therapistURL","upvotes","views","split"],inplace=True)


# In[7]:


df


# In[8]:


df.qqatext[1]


# In[9]:


# Clean it up a little bit, removing non-word characters (numbers and ___ etc)
df.qqatext = df.qqatext.str.replace("[^A-Za-z ]", " ")


# In[10]:


df


# In[11]:


import gensim
from gensim.utils import simple_preprocess
#tokenizing
texts = df.qqatext.apply(simple_preprocess)


# In[12]:


texts[2]


# In[13]:


from gensim import corpora

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.5)

corpus = [dictionary.doc2bow(text) for text in texts]


# In[14]:


dictionary.token2id


# In[15]:


corpus


# In[16]:


from gensim import models

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]


# In[17]:


n_topics = 5

# Build an LSI model
lsi_model = models.LsiModel(corpus_tfidf,
                            id2word=dictionary,
                            num_topics=n_topics)


# In[18]:


lsi_model.print_topics()

Gensim is all about how important each word is to the category. Why not visualize it? First we'll make a dataframe that shows each topic, its top five words, and its values.[ ]
# In[19]:


n_words = 15

topic_words = pd.DataFrame({})

for i, topic in enumerate(lsi_model.get_topics()):
    top_feature_ids = topic.argsort()[-n_words:][::-1]
    feature_values = topic[top_feature_ids]
    words = [dictionary[id] for id in top_feature_ids]
    topic_df = pd.DataFrame({'value': feature_values, 'word': words, 'topic': i})
    topic_words = pd.concat([topic_words, topic_df], ignore_index=True)

topic_words.head()


# In[20]:


#Then we'll use seaborn to visualize it.
import seaborn as sns
import matplotlib.pyplot as plt

g = sns.FacetGrid(topic_words, col="topic", col_wrap=3, sharey=False)
g.map(plt.barh, "word", "value")


# In[ ]:





# In[21]:


import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words


# ## LDA_MODEL with Gensim

# In[22]:


from gensim.utils import simple_preprocess
texts = df.qqatext.apply(simple_preprocess)
texts[0]


# In[23]:


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# In[24]:


data=remove_stopwords(texts)


# In[25]:


get_ipython().system('pip install --upgrade gensim')

import gensim
from gensim.utils import lemmatize
lemmatized_out = [wd.decode('utf-8').split('.')[0]for wd in lemmatize(data)]
# In[26]:



from gensim import corpora

dictionary = corpora.Dictionary(data)
dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=2000)
corpus = [dictionary.doc2bow(text) for text in data]

from gensim import models
"""
n_topics = 15
lda_model = models.LdaModel(corpus=corpus, num_topics=n_topics)

lda_model.print_topics()
"""
# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = models.LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)


# In[27]:


model.print_topics()


# In[28]:


get_ipython().system('pip install pyLDAvis')


# In[29]:


import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()
lda_viz = gensimvis.prepare(model, corpus, dictionary)
lda_viz


# In[30]:


top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

