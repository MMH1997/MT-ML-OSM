#!/usr/bin/env python
# coding: utf-8

# ### We import libraries

# In[1]:


import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# ### We load datasets of tags from different cities

# In[2]:


buiP = pd.read_csv('buildingP3.csv', delimiter=";")
buiP = buiP.iloc[:,0:5]
ameP = pd.read_csv('amenityP3.csv', delimiter=";")
ameP = ameP.iloc[:,0:5]
touP = pd.read_csv('highwayP3.csv', delimiter=";")
touP = touP.iloc[:,0:5]
higp = pd.read_csv('tourismP3.csv', delimiter=";")
higp = higp.iloc[:,0:5]


# In[3]:


Paris = buiP.append(ameP, ignore_index=True)
Paris = Paris.append(touP, ignore_index = True)
Paris = Paris.append(higp, ignore_index = True)
Paris.to_csv('Paris.csv', index = False)  


# In[4]:


buiBr = pd.read_csv('buildingBr3.csv', delimiter=";")
buiBr = buiBr.iloc[:,0:5]
ameBr = pd.read_csv('amenityBr3.csv', delimiter=";")
ameBr = ameBr.iloc[:,0:5]
touBr = pd.read_csv('tourismBr3.csv', delimiter=";")
touBr = touBr.iloc[:,0:5]
higBr = pd.read_csv('highwayBr3.csv', delimiter=";")
higBr = higBr.iloc[:,0:5]
Bruselas = buiBr.append(ameBr, ignore_index=True)
Bruselas = Bruselas.append(touBr, ignore_index = True)
Bruselas = Bruselas.append(higBr, ignore_index = True)
Bruselas = Bruselas.drop(Bruselas[(Bruselas['error'] != 'no') & (Bruselas['error'] != 'yes')].index)
Bruselas.to_csv('Bruselas.csv', index = False)  
Bruselas = pd.read_csv('Bruselas.csv')


# In[6]:


buiB = pd.read_csv('buildingB3.csv', delimiter=";")
buiB = buiB.iloc[:,0:5]
ameB = pd.read_csv('amenityB3.csv', delimiter=";")
ameB = ameB.iloc[:,0:5]
touB = pd.read_csv('highwayB3.csv', delimiter=";")
touB = touB.iloc[:,0:5]
higB = pd.read_csv('tourismB3.csv', delimiter=";")
higB = higB.iloc[:,0:5]
Barcelona = buiB.append(ameB, ignore_index=True)
Barcelona = Barcelona.append(touB, ignore_index = True)
Barcelona = Barcelona.append(higB, ignore_index = True)
Barcelona.to_csv('Barcelona.csv', index = False)  
Barcelona = pd.read_csv('Barcelona.csv', delimiter = ';')


# In[7]:


buiH = pd.read_csv('buildingH3.csv', delimiter=";")
buiH = buiH.iloc[:,0:5]
ameH = pd.read_csv('amenityH3.csv', delimiter=";")
ameH = ameH.iloc[:,0:5]
touH = pd.read_csv('highwayH3.csv', delimiter=";")
touH = touH.iloc[:,0:5]
higH = pd.read_csv('tourismH3.csv', delimiter=";")
higH = higH.iloc[:,0:5]
Habana = buiH.append(ameH, ignore_index=True)
Habana = Habana.append(touH, ignore_index = True)
Habana = Habana.append(higH, ignore_index = True)
Habana.to_csv('Habana.csv', index = False)  
Habana = pd.read_csv('Habana.csv')


# In[8]:


# buiBo = pd.read_csv('buildingBo3.csv', delimiter=";")
# buiBo = buiBo.iloc[:,0:5]
# ameBo = pd.read_csv('amenityBo3.csv', delimiter=";")
# ameBo = ameBo.iloc[:,0:5]
# touBo = pd.read_csv('highwayBo3.csv', delimiter=";")
# touBo = touBo.iloc[:,0:5]
# higBo = pd.read_csv('tourismBo3.csv', delimiter=";")
# higBo = higBo.iloc[:,0:5]
# Boston = buiBo.append(ameBo, ignore_index=True)
# Boston = Boston.append(touBo, ignore_index = True)
# Boston = Boston.append(higBo, ignore_index = True)
# Boston.to_csv('Boston.csv', index = False)  
Boston = pd.read_csv('Boston.csv')


# In[9]:


# buiSC = pd.read_csv('buildingSC3.csv', delimiter=";")
# buiSC = buiSC.iloc[:,0:5]
# ameSC = pd.read_csv('amenitySC3.csv', delimiter=";")
# ameSC = ameSC.iloc[:,0:5]
# touSC = pd.read_csv('highwaySC3.csv', delimiter=";")
# touSC = touSC.iloc[:,0:5]
# higSC = pd.read_csv('tourismSC3.csv', delimiter=";")
# higSC = higSC.iloc[:,0:5]
# SChile = buiSC.append(ameSC, ignore_index=True)
# SChile = SChile.append(touSC, ignore_index = True)
# SChile = SChile.append(higSC, ignore_index = True)
# SChile.to_csv('SChile.csv', index = False)  
SChile = pd.read_csv('SChile.csv')


# In[10]:


# buiLu = pd.read_csv('buildingLu3.csv', delimiter=";")
# buiLu = buiLu.iloc[:,0:5]
# ameLu = pd.read_csv('amenityLu3.csv', delimiter=";")
# ameLu = ameLu.iloc[:,0:5]
# touLu = pd.read_csv('highwayLu3.csv', delimiter=";")
# touLu = touLu.iloc[:,0:5]
# higLu = pd.read_csv('tourismLu3.csv', delimiter=";")
# higLu = higLu.iloc[:,0:5]
# Luanda = buiLu.append(ameLu, ignore_index=True)
# Luanda = Luanda.append(touLu, ignore_index = True)
# Luanda = Luanda.append(higLu, ignore_index = True)
# Luanda.to_csv('Luanda.csv', index = False)  
Luanda = pd.read_csv('Luanda.csv')


# In[11]:


# buiWu = pd.read_csv('buildingWu3.csv', delimiter=";")
# buiWu = buiWu.iloc[:,0:5]
# ameWu = pd.read_csv('amenityWu3.csv', delimiter=";")
# ameWu = ameWu.iloc[:,0:5]
# touWu = pd.read_csv('highwayWu3.csv', delimiter=";")
# touWu = touWu.iloc[:,0:5]
# higWu = pd.read_csv('tourismWu3.csv', delimiter=";")
# higWu = higWu.iloc[:,0:5]
# Wuhan = buiWu.append(ameWu, ignore_index=True)
# Wuhan = Wuhan.append(touWu, ignore_index = True)
# Wuhan = Wuhan.append(higWu, ignore_index = True)
# Wuhan.to_csv('Wuhan.csv', index = False)  
Wuhan = pd.read_csv('Wuhan.csv')


# In[12]:


buiR = pd.read_csv('buildiingR3.csv', delimiter=";")
buiR = buiR.iloc[:,0:5]
ameR = pd.read_csv('amenityR3.csv', delimiter=";")
ameR = ameR.iloc[:,0:5]
touR = pd.read_csv('highwayR3.csv', delimiter=";")
touR = touR.iloc[:,0:5]
higR = pd.read_csv('tourismR3.csv', delimiter=";")
higR = higR.iloc[:,0:5]
Roma = buiR.append(ameR, ignore_index=True)
Roma = Roma.append(touR, ignore_index = True)
Roma = Roma.append(higR, ignore_index = True)
Roma = Roma.drop(Roma[(Roma['error'] != 'no') & (Roma['error'] != 'yes')].index)
Roma.to_csv('Roma.csv', index = False)  
Roma = pd.read_csv('Roma.csv')


# In[13]:


# buiSC = pd.read_csv('buildingSC3.csv', delimiter=";")
# buiSC = buiSC.iloc[:,0:5]
# ameSC = pd.read_csv('amenitySC3.csv', delimiter=";")
# ameSC = ameSC.iloc[:,0:5]
# touSC = pd.read_csv('highwaySC3.csv', delimiter=";")
# touSC = touSC.iloc[:,0:5]
# higSC = pd.read_csv('tourismSC3.csv', delimiter=";")
# higSC = higSC.iloc[:,0:5]
# Cairo = buiSC.append(ameSC, ignore_index=True)
# Cairo = Cairo.append(touSC, ignore_index = True)
# Cairo = Cairo.append(higSC, ignore_index = True)
# Cairo.to_csv('Cairo.csv', index = False)  
Cairo = pd.read_csv('Cairo.csv')


# In[15]:


# buiO = pd.read_csv('buildingO3.csv', delimiter=";")
# buiO = buiO.iloc[:,0:5]
# ameO = pd.read_csv('amenityO3.csv', delimiter=";")
# ameO = ameO.iloc[:,0:5]
# higO = pd.read_csv('highwayO3.csv', delimiter=";")
# higO = higO.iloc[:,0:5]
# Otawa = buiO.append(ameO, ignore_index=True)
# Otawa = Otawa.append(higO, ignore_index = True)
# Otawa.to_csv('Otawa.csv', index = False)  
Otawa = pd.read_csv('Otawa.csv')


# ### We create the function

# In[18]:


def error(dataset, deli):
    data = pd.read_csv(dataset, delimiter = deli)
    # data = pd.read_csv('Wuhan.csv', delimiter = ',')
    index = data.iloc[:,0]
    obj = data['error']
    data = data[['alpha','alphap']]
    data1 = list(data['alpha'])
    data2 = list(data['alphap'])
    file = open("dict_all.obj",'rb')
    dict_all_loaded = pickle.load(file)
    file = open("target.obj",'rb')
    dict_obj_loaded = pickle.load(file)
    file.close()
    file.close()
    for col in data.columns[0:2]:
            data.replace(dict_all_loaded[col], inplace=True)
    data['aux1']=data1 
    data['aux2']=data2        
    def get_key(val):
        for key, value in dict_obj_loaded.items():
             if val == value:
                 return key
        return "key doesn't exist"
    rf = joblib.load("./random_forest.joblib")
    lista = []
    for i in range(len(data)):
            if data.iloc[i,3][0:5] == 'name:':
                lista.append(np.array([dict_obj_loaded['no']]))
            elif data.iloc[i,3][0:5] == 'fuel:':
                lista.append(np.array([dict_obj_loaded['no']])) 
            elif data.iloc[i,3][0:12] == 'description:':
                lista.append(np.array([dict_obj_loaded['no']]))   
            elif data.iloc[i,3][0:4] == 'ref:':
                lista.append(np.array([dict_obj_loaded['no']]))   
            elif data.iloc[i,3][0:8] == 'network:':
                lista.append(np.array([dict_obj_loaded['no']]))  
            elif data.iloc[i,3][0:10] == 'recycling:':
                lista.append(np.array([dict_obj_loaded['no']]))
            elif data.iloc[i,3][0:9] == 'currency:':
                lista.append(np.array([dict_obj_loaded['no']]))
            elif data.iloc[i,3][0:9] == 'alt_name:':
                lista.append(np.array([dict_obj_loaded['no']])) 
            elif data.iloc[i,3][0:7] == 'source:':
                lista.append(np.array([dict_obj_loaded['no']]))                 
            elif data.iloc[i,3][0:8] == 'surface:':
                lista.append(np.array([dict_obj_loaded['no']]))              
            elif data.iloc[i,3][0:8] == 'contact:':
                lista.append(np.array([dict_obj_loaded['no']])) 
            elif data.iloc[i,3][0:10] == 'wikipedia:':
                lista.append(np.array([dict_obj_loaded['no']])) 
            elif data.iloc[i,3][0:9] == 'wikidata:':
                lista.append(np.array([dict_obj_loaded['no']]))
            elif data.iloc[i,3][0:8] == 'massgis:':
                lista.append(np.array([dict_obj_loaded['no']]))  
            elif data.iloc[i,3][0:9] == 'wikidata:':
                lista.append(np.array([dict_obj_loaded['no']]))
            elif data.iloc[i,3][0:11] == 'short_name:':
                lista.append(np.array([dict_obj_loaded['no']]))     
            elif data.iloc[i,3][0:5] == 'gnis:':
                lista.append(np.array([dict_obj_loaded['no']]))               
            elif data.iloc[i,3][0:5] == 'addr:' and data.iloc[i,3] != 'addr:housename' and data.iloc[i,2] != 'highway':
                lista.append(np.array([dict_obj_loaded['no']])) 
            elif data.iloc[i,1] not in list(dict_all_loaded.get('alphap').values()):
                lista.append(np.array([dict_obj_loaded['yes']]))
            else:
                pred = pd.DataFrame(data.iloc[i,0:2]).T
                a = rf.predict(pred)
                lista.append(a)

    lista1=[]
    for i in lista: lista1.append(i[0])
    obj1 = [dict_obj_loaded[x] for x in obj]
    list2 = []
    for i in lista1:
        list2.append(get_key(i))
    df = {'ID': index, 'Error': list2, 'Error real': obj}
    df = pd.DataFrame(df)
#     print(df)
#     print(accuracy_score(lista1,obj1))
#     print(confusion_matrix(obj1, lista1))
    return(df,accuracy_score(lista1,obj1),recall_score(lista1,obj1),
           f1_score(lista1,obj1),confusion_matrix(obj1, lista1))


# ### We check the quality of the results in Rome example. 

# In[21]:


error('Roma.csv',',')

