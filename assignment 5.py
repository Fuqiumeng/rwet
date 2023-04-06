#!/usr/bin/env python
# coding: utf-8

# In[13]:


import spacy


# In[14]:


nlp = spacy.load('en_core_web_md')


# In[15]:


nlp.vocab['minimalist'].vector


# In[16]:


def vec(s):
    return nlp.vocab[s].vector


# In[17]:


bio = nlp(open("bio.txt").read())


# In[144]:


Rilke = nlp(open("Rilke.txt").read())


# In[18]:


get_ipython().system('curl -L -O https://raw.githubusercontent.com/aparrish/wordfreq-en-25000/main/wordfreq-en-25000-log.json')


# In[19]:


import json
prob_lookup = dict(json.load(open("./wordfreq-en-25000-log.json")))


# In[20]:


import sys
get_ipython().system('{sys.executable} -m pip install simpleneighbors')


# In[21]:


import sys
get_ipython().system('{sys.executable} -m pip install annoy==1.16.3')


# In[22]:


from simpleneighbors import SimpleNeighbors


# In[23]:


import random


# In[24]:


lookup = SimpleNeighbors(300)
for word in prob_lookup.keys():
    if nlp.vocab[word].has_vector:
        lookup.add_one(word, vec(word))
lookup.build()


# In[145]:


output = []
for word in bio:
    if word.is_alpha and word.pos_ in ('NOUN', 'VERB', 'ADJ'):
        new_word = random.choice(lookup.nearest(word.vector, 3))
        output.append(new_word)
    else:
        output.append(word.text)
    output.append(word.whitespace_)
print(''.join(output))


# In[ ]:





# In[29]:


target_word = 'cosmo'
factor = 0.6


# In[172]:


output1 = []
for word in bio:
    if word.is_alpha and word.pos_ in ('NOUN', 'VERB', 'ADJ'):
        new_word = random.choice(
            lookup.nearest((word.vector*(1-factor)) + (vec(target_word)*factor), 5))
        output1.append(new_word)
    else:
        output1.append(word.text)
    output1.append(word.whitespace_)
print(''.join(output1))


# In[120]:


text_hisao =print(''.join(output))


# In[ ]:


target_word = 'cosmo'
factor = 0.6


# In[178]:


output2 = []
for word in Rilke:
    if word.is_alpha and word.pos_ in ('NOUN', 'VERB', 'ADJ'):
        new_word = random.choice(
            lookup.nearest((word.vector*(1-factor)) + (vec(target_word)*factor), 5))
        output2.append(new_word)
    else:
        output2.append(word.text)
    output2.append(word.whitespace_)
print(''.join(output2))


# In[174]:





# In[ ]:





# In[111]:


import sys
get_ipython().system('{sys.executable} -m pip install markovify')


# In[196]:


import markovify


# In[197]:


generator_a = markovify.Text(''.join(output1))


# In[198]:


generator_b=markovify.Text(''.join(output2))


# In[199]:


print(generator_a.make_sentence())


# In[200]:


print(generator_a.make_short_sentence(1))


# In[201]:


print(generator_a.make_short_sentence(10, tries=500))


# In[211]:


generator_a = markovify.Text(''.join(output1))
generator_b = markovify.Text(''.join(output2))
combo = markovify.combine([generator_a, generator_b], [0.2, 0.8])


# In[212]:


print(combo.make_sentence())


# In[213]:


class SentencesByChar(markovify.Text):
    def word_split(self, sentence):
        return list(sentence)
    def word_join(self, words):
        return "".join(words)


# In[214]:


# change to "word" for a word-level model
level = "char"
# controls the length of the n-gram
order = 7
# controls the number of lines to output
output_n = 14
# weights between the models; text A first, text B second.
# if you want to completely exclude one model, set its corresponding value to 0
weights = [0.6, 0.4]
# limit sentence output to this number of characters
length_limit = 280


# In[215]:


model_cls = markovify.Text if level == "word" else SentencesByChar
gen_a = model_cls(''.join(output1), state_size=order)
gen_b = model_cls(''.join(output2), state_size=order)
gen_combo = markovify.combine([gen_a, gen_b], weights)
for i in range(output_n):
    out = gen_combo.make_short_sentence(length_limit, test_output=False)
    out = out.replace("\n", " ")
    print(out)
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:





# In[36]:





# In[37]:





# In[ ]:





# In[ ]:




