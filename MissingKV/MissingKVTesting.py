#!/usr/bin/env python
# coding: utf-8

# ### We import the libraries.

# In[1]:


import numpy as np
import pandas as pd


# ### We load the datasets of the tags. 

# In[2]:


buiBarcelona = pd.read_csv('buildingBarcelona.csv', delimiter=";", encoding='latin-1')
buiBarcelona = buiBarcelona.iloc[:,1:]
ameBarcelona = pd.read_csv('amenityBarcelona.csv', delimiter=";", encoding='latin-1')
ameBarcelona = ameBarcelona.iloc[:,1:]
# touBarcelona = pd.read_csv('tourismBarcelona.csv', delimiter=";", encoding='latin-1')
# touBarcelona = touBarcelona.iloc[:,1:]
higBarcelona = pd.read_csv('highwayBarcelona.csv', delimiter=";", encoding='latin-1')
higBarcelona = higBarcelona.iloc[:,1:]
Barcelona = buiBarcelona.append(ameBarcelona, ignore_index=True)
# Barcelona = Barcelona.append(touBarcelona, ignore_index = True)
Barcelona = Barcelona.append(higBarcelona, ignore_index = True)
Barcelona


# In[3]:


buiBoston = pd.read_csv('buildingBoston.csv', delimiter=";", encoding='latin-1')
buiBoston = buiBoston.iloc[:,1:]
ameBoston = pd.read_csv('amenityBoston.csv', delimiter=";", encoding='latin-1')
ameBoston = ameBoston.iloc[:,1:]
touBoston = pd.read_csv('tourismBoston.csv', delimiter=";", encoding='latin-1')
touBoston = touBoston.iloc[:,1:]
higBoston = pd.read_csv('highwayBoston.csv', delimiter=";", encoding='latin-1')
higBoston = higBoston.iloc[:,1:]
Boston = buiBoston.append(ameBoston, ignore_index=True)
Boston = Boston.append(touBoston, ignore_index = True)
Boston = Boston.append(higBoston, ignore_index = True)
Boston


# In[4]:


buiBrussel = pd.read_csv('buildingBrussel.csv', delimiter=";", encoding='latin-1')
buiBrussel = buiBrussel.iloc[:,1:]
ameBrussel = pd.read_csv('amenityBrussel.csv', delimiter=";", encoding='latin-1')
ameBrussel = ameBrussel.iloc[:,1:]
touBrussel = pd.read_csv('tourismBrussel.csv', delimiter=";", encoding='latin-1')
touBrussel = touBrussel.iloc[:,1:]
higBrussel = pd.read_csv('highwayBrussel.csv', delimiter=";", encoding='latin-1')
higBrussel = higBrussel.iloc[:,1:]
Brussel = buiBrussel.append(ameBrussel, ignore_index=True)
Brussel = Brussel.append(touBrussel, ignore_index = True)
Brussel = Brussel.append(higBrussel, ignore_index = True)
Brussel


# In[5]:


buiElCairo = pd.read_csv('buildingElCairo.csv', delimiter=";", encoding='latin-1')
buiElCairo = buiElCairo.iloc[:,1:]
ameElCairo = pd.read_csv('amenityElCairo.csv', delimiter=";", encoding='latin-1')
ameElCairo = ameElCairo.iloc[:,1:]
touElCairo = pd.read_csv('tourismElCairo.csv', delimiter=";", encoding='latin-1')
touElCairo = touElCairo.iloc[:,1:]
higElCairo = pd.read_csv('highwayElCairo.csv', delimiter=";", encoding='latin-1')
higElCairo = higElCairo.iloc[:,1:]
ElCairo = ameElCairo.append(higElCairo, ignore_index = True)
ElCairo = ElCairo.append(buiElCairo, ignore_index = True)
ElCairo = ElCairo.append(touElCairo, ignore_index = True)


# In[6]:


buiElCairo2 = pd.read_csv('buildingElCairo2.csv', delimiter=";", encoding='latin-1')
buiElCairo2 = buiElCairo2.iloc[:,1:]
ameElCairo2 = pd.read_csv('amenityElCairo2.csv', delimiter=";", encoding='latin-1')
ameElCairo2 = ameElCairo2.iloc[:,1:]
touElCairo2 = pd.read_csv('tourismElCairo2.csv', delimiter=";", encoding='latin-1')
touElCairo2 = touElCairo2.iloc[:,1:]
higElCairo2 = pd.read_csv('highwayElCairo2.csv', delimiter=";", encoding='latin-1')
higElCairo2 = higElCairo2.iloc[:,1:]
ElCairo2 = ameElCairo2.append(higElCairo2, ignore_index = True)
ElCairo2 = ElCairo2.append(buiElCairo2, ignore_index = True)
ElCairo2 = ElCairo2.append(touElCairo2, ignore_index = True)
ElCairo2


# In[7]:


buiLaHabana = pd.read_csv('buildingLaHabana.csv', delimiter=";", encoding='latin-1')
buiLaHabana = buiLaHabana.iloc[:,1:]
ameLaHabana = pd.read_csv('amenityLaHabana.csv', delimiter=";", encoding='latin-1')
ameLaHabana = ameLaHabana.iloc[:,1:]
touLaHabana = pd.read_csv('tourismLaHabana.csv', delimiter=";", encoding='latin-1')
touLaHabana = touLaHabana.iloc[:,1:]
higLaHabana = pd.read_csv('highwayLaHabana.csv', delimiter=";", encoding='latin-1')
higLaHabana = higLaHabana.iloc[:,1:]
LaHabana = buiLaHabana.append(ameLaHabana, ignore_index=True)
LaHabana = LaHabana.append(touLaHabana, ignore_index = True)
LaHabana = LaHabana.append(higLaHabana, ignore_index = True)
LaHabana


# In[8]:


buiLuanda = pd.read_csv('buildingLuanda.csv', delimiter=";", encoding='latin-1')
buiLuanda = buiLuanda.iloc[:,1:]
ameLuanda = pd.read_csv('amenityLuanda.csv', delimiter=";", encoding='latin-1')
ameLuanda = ameLuanda.iloc[:,1:]
touLuanda = pd.read_csv('tourismLuanda.csv', delimiter=";", encoding='latin-1')
touLuanda = touLuanda.iloc[:,1:]
higLuanda = pd.read_csv('highwayLuanda.csv', delimiter=";", encoding='latin-1')
higLuanda = higLuanda.iloc[:,1:]
Luanda = buiLuanda.append(ameLuanda, ignore_index=True)
Luanda = Luanda.append(touLuanda, ignore_index = True)
Luanda = Luanda.append(higLuanda, ignore_index = True)
Luanda


# In[9]:


buiOtawa = pd.read_csv('buildingOtawa.csv', delimiter=";", encoding='latin-1')
buiOtawa = buiOtawa.iloc[:,1:]
ameOtawa = pd.read_csv('amenityOtawa.csv', delimiter=";", encoding='latin-1')
ameOtawa = ameOtawa.iloc[:,1:]
# touOtawa = pd.read_csv('tourismOtawa.csv', delimiter=";", encoding='latin-1')
# touOtawa = touOtawa.iloc[:,1:]
higOtawa = pd.read_csv('highwayOtawa.csv', delimiter=";", encoding='latin-1')
higOtawa = higOtawa.iloc[:,1:]
Otawa = buiOtawa.append(ameOtawa, ignore_index=True)
# Otawa = Otawa.append(touOtawa, ignore_index = True)
Otawa = Otawa.append(higOtawa, ignore_index = True)
Otawa


# In[10]:


buiOtawa2 = pd.read_csv('buildingOtawa2.csv', delimiter=";", encoding='latin-1')
buiOtawa2 = buiOtawa2.iloc[:,1:]
ameOtawa2 = pd.read_csv('amenityOtawa2.csv', delimiter=";", encoding='latin-1')
ameOtawa2 = ameOtawa2.iloc[:,1:]
# touOtawa = pd.read_csv('tourismOtawa.csv', delimiter=";", encoding='latin-1')
# touOtawa = touOtawa.iloc[:,1:]
higOtawa2 = pd.read_csv('highwayOtawa2.csv', delimiter=";", encoding='latin-1')
higOtawa2 = higOtawa2.iloc[:,1:]
Otawa2 = buiOtawa2.append(ameOtawa2, ignore_index=True)
# Otawa = Otawa.append(touOtawa, ignore_index = True)
Otawa2 = Otawa2.append(higOtawa2, ignore_index = True)
Otawa2


# In[11]:


buiOtawa3 = pd.read_csv('buildingOtawa3.csv', delimiter=";", encoding='latin-1')
buiOtawa3 = buiOtawa3.iloc[:,1:]
ameOtawa3 = pd.read_csv('amenityOtawa3.csv', delimiter=";", encoding='latin-1')
ameOtawa3 = ameOtawa3.iloc[:,1:]
# touOtawa = pd.read_csv('tourismOtawa.csv', delimiter=";", encoding='latin-1')
# touOtawa = touOtawa.iloc[:,1:]
higOtawa3 = pd.read_csv('highwayOtawa3.csv', delimiter=";", encoding='latin-1')
higOtawa3 = higOtawa3.iloc[:,1:]
Otawa3 = buiOtawa3.append(ameOtawa3, ignore_index=True)
# Otawa = Otawa.append(touOtawa, ignore_index = True)
Otawa3 = Otawa3.append(higOtawa3, ignore_index = True)
Otawa3


# In[12]:


buiSantiagoChile = pd.read_csv('buildingSantiagoChile.csv', delimiter=";", encoding='latin-1')
buiSantiagoChile = buiSantiagoChile.iloc[:,1:]
ameSantiagoChile = pd.read_csv('amenitySantiagoChile.csv', delimiter=";", encoding='latin-1')
ameSantiagoChile = ameSantiagoChile.iloc[:,1:]
touSantiagoChile = pd.read_csv('tourismSantiagoChile.csv', delimiter=";", encoding='latin-1')
touSantiagoChile = touSantiagoChile.iloc[:,1:]
higSantiagoChile = pd.read_csv('highwaySantiagoChile.csv', delimiter=";", encoding='latin-1')
higSantiagoChile = higSantiagoChile.iloc[:,1:]
SantiagoChile = buiSantiagoChile.append(ameSantiagoChile, ignore_index=True)
SantiagoChile = SantiagoChile.append(touSantiagoChile, ignore_index = True)
SantiagoChile = SantiagoChile.append(higSantiagoChile, ignore_index = True)
SantiagoChile


# In[13]:


buiRome = pd.read_csv('buildingRome.csv', delimiter=";", encoding='latin-1')
buiRome = buiRome.iloc[:,1:]
ameRome = pd.read_csv('amenityRome.csv', delimiter=";", encoding='latin-1')
ameRome = ameRome.iloc[:,1:]
touRome = pd.read_csv('tourismRome.csv', delimiter=";", encoding='latin-1')
touRome = touRome.iloc[:,1:]
higRome = pd.read_csv('highwayRome.csv', delimiter=";", encoding='latin-1')
higRome = higRome.iloc[:,1:]
Rome = buiRome.append(ameRome, ignore_index=True)
Rome = Rome.append(touRome, ignore_index = True)
Rome = Rome.append(higRome, ignore_index = True)
Rome


# In[14]:


buiWuhan = pd.read_csv('buildingWuhan.csv', delimiter=";", encoding='latin-1')
buiWuhan = buiWuhan.iloc[:,1:]
ameWuhan = pd.read_csv('amenityWuhan.csv', delimiter=";", encoding='latin-1')
ameWuhan = ameWuhan.iloc[:,1:]
touWuhan = pd.read_csv('tourismWuhan.csv', delimiter=";", encoding='latin-1')
touWuhan = touWuhan.iloc[:,1:]
higWuhan = pd.read_csv('highwayWuhan.csv', delimiter=";", encoding='latin-1')
higWuhan = higWuhan.iloc[:,1:]
Wuhan = buiWuhan.append(ameWuhan, ignore_index=True)
Wuhan = Wuhan.append(touWuhan, ignore_index = True)
Wuhan = Wuhan.append(higWuhan, ignore_index = True)
Wuhan


# In[15]:


import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# In[16]:


ElCairo2.to_csv('ElCairo2.csv', index = False)  


# In[17]:


Barcelona


# ### We create the function.

# In[18]:


def error(dataset, deli):
    data = pd.read_csv(dataset, delimiter = deli)
#     data = pd.read_csv('ElCairo2.csv', delimiter = ',', encoding='latin-1')
    index = data.iloc[:,0]
    obj = data['Role']
    data = data[['alpha','alphap','beta']]
    data1 = list(data['alphap'])
    data2 = list(data['beta'])
    file = open("dict_all.obj",'rb')
    dict_all_loaded = pickle.load(file)
    file = open("target.obj",'rb')
    dict_obj_loaded = pickle.load(file)
    file.close()
    file.close()
    for col in data.columns[0:3]:
            data.replace(dict_all_loaded[col], inplace=True)
    data['aux1']=data1 
    data['aux2']=data2
    DF = pd.read_csv('DF.csv')
    ee = []
    for i in range(len(DF)):
        ee.append(str(DF["alphap"][i]) + " " + str(DF["beta"][i]))
    DF['conc'] = ee
    def get_key(val):
        for key, value in dict_obj_loaded.items():
             if val == value:
                 return key
        return "key doesn't exist"
    rf = joblib.load("./random_forest.joblib")
    lista = []
    for i in range(len(data)):        
            if str(data['alphap'][i]) + " " + str(data['beta'][i]) not in DF['conc'].values:
                lista.append(np.array([dict_obj_loaded['Useful Combination']]))
            else:
                pred = pd.DataFrame(data.iloc[i,0:3]).T
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
    print(df)
    print(accuracy_score(lista1,obj1))
    print(f1_score(lista1,obj1, average = 'weighted'))
    print(confusion_matrix(obj1, lista1))


# ### We check the quality of the results in Wuhan example. 

# In[19]:


Wuhan.to_csv('Wuhan.csv', index = False)  
error('Wuhan.csv', deli = ',')


# In[ ]:





# In[ ]:





# In[ ]:




