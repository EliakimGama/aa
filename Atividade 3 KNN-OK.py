#!/usr/bin/env python
# coding: utf-8

# In[111]:


from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# In[173]:


df_wine = pd.read_csv('wine_dataset.csv')


# In[113]:


df_wine.head().T


# In[114]:


df_wine.info()


# In[115]:


df_wine['style'] = df_wine['style'].replace('red', 0)
df_wine['style'] = df_wine['style'].replace('white', 1)


# In[116]:


df_wine['style'].value_counts()


# In[117]:


X_train, X_test, y_train, y_test = train_test_split(df_wine.drop('style',axis=1), df_wine['style'], test_size=0.3)


# In[118]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[119]:


knn.fit(X_train, y_train)


# In[120]:


resultado = knn.predict(X_test)
resultado


# In[121]:


print (pd.crosstab(y_test,resultado, rownames=['Real'], colnames=['Predito'], margins=True))


# In[163]:


print(metrics.classification_report(y_test,resultado))


# In[164]:


k_list = list(range(1,31))
weight_list = ("uniform",'distance')
p_list = [1,2]


# In[156]:


parametros = dict(n_neighbors=k_list)


# In[161]:


parametros


# In[157]:


grid = GridSearchCV(knn, parametros, cv=5, scoring='accuracy')


# In[158]:


grid.fit(df_wine.drop('style',axis=1),df_wine['style'])


# In[165]:


grid.cv_results_


# In[166]:


scores = grid.cv_results_.get('mean_test_score')


# In[167]:


k_rank = grid.cv_results_.get('rank_test_score')


# In[169]:


print("Melhores parametros {} com o valor de acurácia {} ".format(grid.best_params_,grid.best_score_))


# In[170]:


plt.figure(figsize=(10,6))
plt.plot(k_list,scores,color='red',linestyle='dashed',marker='o')


# In[ ]:




