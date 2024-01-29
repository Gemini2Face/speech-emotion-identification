#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from glob import glob
import librosa
import librosa.display 
import IPython.display as ipd
from itertools import cycle
sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(color_pal)


# In[2]:


audio_files = glob(r'ravedss-audio\*\*.wav')


# In[3]:


audio_files[0]


# In[4]:


ipd.Audio(audio_files[0])


# In[5]:


y, sr = librosa.load(audio_files[0])
print(f"Signal Samples{y[:10]} and Shape of Signal {y.shape} ,Number of Samples {sr}")


# In[6]:


pd.Series(y).plot(figsize=(10, 5), lw=1, title = "RAW AUDIO SAMPLE", color = color_pal[0])


# In[7]:


y_trimmed_trial, _ = librosa.effects.trim(y)
pd.Series(y_trimmed_trial).plot(figsize=(10, 5),title="Trimmed Trial Data", color = color_pal[1])



# In[8]:


y_trimmed, _ = librosa.effects.trim(y, top_db=30)


# In[9]:


pd.Series(y_trimmed).plot(figsize=(10, 5),title="Trimmed Data", color = color_pal[2])


# In[10]:


pd.Series(y[30000:30200]).plot(figsize=(10, 5), lw=1, title="Raw Data Zoomed In", color=color_pal[3])
plt.show()


# In[11]:


D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref = np.max)
S_db.shape


# In[12]:


fig, ax = plt.subplots(figsize=(10, 5))
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)


# In[13]:


audio_file_path = audio_files[0]
y, sr = librosa.load(audio_file_path, sr=None)

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr*0.010), n_fft=int(sr*0.025))

# Display MFCCs
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCCs')
plt.show()
mfccs[0]


# In[14]:


audio_files[:10]


# In[15]:


lst = []
for file in audio_files:
    lst.append(file.split("-")[2])
   
   


# In[15]:





# In[16]:


Emotion = {"01":"neutral", "02":"calm", "03":"happy", "04":"sad", "05":"angry", "06":"fearful", "07":"disgust", "08":"surprised"}


# In[17]:


column_name = "Filename"
df=pd.DataFrame(audio_files, columns=[column_name])


# In[18]:


value_list = [Emotion[key] for key in lst]


# In[19]:


value_list


# In[20]:


df["feeling"] = lst


# In[21]:


df.head()


# In[22]:


df2 = pd.DataFrame()

result=np.array([])
for file in df["Filename"].tolist():
    y, sr = librosa.load(file)
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    stft=np.abs(librosa.stft(y_trimmed))
    mfccs=np.mean(librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=40).T, axis=0)
    mel=np.mean(librosa.feature.melspectrogram(y=y_trimmed, sr=sr).T,axis=0)
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    result = np.hstack(( mfccs, mel, chroma))
    if len(df2.columns)==0:
        print(len(df2.columns))
        print(len(result))
        column_names = [f'col{i+1}' for i in range(len(result))]
        df2 = pd.DataFrame(columns=column_names)
        print(column_names)    
    df2.loc[len(df2)] = result
print(df2)
    
    


# In[23]:


result_df = pd.concat([df2,df],axis=1)


# In[24]:


result_df.head()


# In[25]:


result_df.drop("Filename",axis=1,inplace=True)


# In[26]:


result_df.head()


# In[27]:


X = result_df.drop("feeling",axis=1)
y = result_df["feeling"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


get_ipython().system('pip install --upgrade tensorflow')


# In[29]:


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# In[30]:


from sklearn.neural_network import MLPClassifier


# In[31]:


model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


# In[32]:


model.fit(X_train,y_train)


# In[33]:


from sklearn.metrics import accuracy_score


# In[34]:


y_pred = model.predict(X_test)


# In[35]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# In[35]:




