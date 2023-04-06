#!/usr/bin/env python
# coding: utf-8

# In[31]:


import spacy


# In[32]:


nlp = spacy.load('en_core_web_md')


# In[33]:


nlp.vocab['minimalist'].vector


# In[34]:


def vec(s):
    return nlp.vocab[s].vector


# In[35]:


bio = nlp(open("bio.txt").read())


# In[36]:


get_ipython().system('curl -L -O https://raw.githubusercontent.com/aparrish/wordfreq-en-25000/main/wordfreq-en-25000-log.json')


# In[37]:


import json
prob_lookup = dict(json.load(open("./wordfreq-en-25000-log.json")))


# In[38]:


import sys
get_ipython().system('{sys.executable} -m pip install simpleneighbors')


# In[39]:


import sys
get_ipython().system('{sys.executable} -m pip install annoy==1.16.3')


# In[40]:


from simpleneighbors import SimpleNeighbors


# In[41]:


import random


# In[42]:


lookup = SimpleNeighbors(300)
for word in prob_lookup.keys():
    if nlp.vocab[word].has_vector:
        lookup.add_one(word, vec(word))
lookup.build()


# In[43]:


output = []
for word in bio:
    if word.is_alpha and word.pos_ in ('NOUN', 'VERB', 'ADJ'):
        new_word = random.choice(lookup.nearest(word.vector, 3))
        output.append(new_word)
    else:
        output.append(word.text)
    output.append(word.whitespace_)
print(''.join(output))


# In[44]:


target_word = 'minimalist'
factor = 0.6


# In[45]:


output = []
for word in bio:
    if word.is_alpha and word.pos_ in ('NOUN', 'VERB', 'ADJ'):
        new_word = random.choice(
            lookup.nearest((word.vector*(1-factor)) + (vec(target_word)*factor), 5))
        output.append(new_word)
    else:
        output.append(word.text)
    output.append(word.whitespace_)
print(''.join(output))


# In[ ]:





# In[ ]:




