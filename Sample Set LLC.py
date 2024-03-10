#!/usr/bin/env python
# coding: utf-8

# In[67]:


import pandas as pd 

# Since your file has a CSV extension, you should use read_csv
df = pd.read_csv(r'C:\Users\Asus\OneDrive\Desktop\Excler\LLC\BA - Assignment Part 1 data set.csv')

# Display the first few rows of the DataFrame
df.head()


# In[68]:


df.tail()


# In[69]:


df.shape


# In[70]:


df.info()


# In[71]:


df.isnull().sum()


# In[72]:


df.describe()


# In[73]:


data = pd.DataFrame(df)
replace_mean = data.mean()
for col in data.columns:
    data[col].fillna(replace_mean[col],inplace=True)
print(data)    
    


# In[74]:


import pandas as pd

# Assuming df is your DataFrame
data = pd.DataFrame(df)

# Calculating mean values
replace_mean = data.mean()

# Filling missing values with mean values
for col in data.columns:
    data[col].fillna(replace_mean[col], inplace=True)

print(data)


# In[ ]:





# In[75]:


import pandas as pd

# Assuming df is your DataFrame
data = pd.DataFrame(df)

# Calculating mean values
replace_mean = data.mean()

# Filling missing values with mean values
for col in data.columns:
    if col in replace_mean.index:  # Check if column exists in replace_mean
        data[col].fillna(replace_mean[col], inplace=True)
    else:
        print(f"Column '{col}' not found in mean values.")

print(data)


# In[76]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Assuming df is your DataFrame
# data = pd.DataFrame(df)  # No need to convert df to DataFrame again

# Fill missing values with mean for numeric columns and label encode object columns
replace_mean = df.mean()
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill missing values with mode for object columns
        df[col] = label_encoder.fit_transform(df[col])  # Label encode object columns
    else:
        df[col].fillna(replace_mean[col], inplace=True)  # Fill missing values with mean for numeric columns

print(df)


# In[77]:


data.head()


# In[78]:


data.isnull().sum()


# In[79]:


df.info()


# In[80]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(data)
SS_X = pd.DataFrame(SS_X)
SS_X.columns = list(data)
SS_X.head()


# In[81]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create boxplots for each column in SS_X
plt.figure(figsize=(10, 6))
sns.boxplot(data=SS_X)
plt.title('Boxplot of Standardized Data')
plt.xlabel('Features')
plt.ylabel('Standardized Values')
plt.xticks(rotation=45)
plt.show()


# In[82]:


data['Tenure'].max()


# In[83]:


data.hist()


# In[85]:


import numpy as np

# Calculate Z-Score
z_scores = (data - data.mean()) / data.std()

# Define Threshold
threshold = 3  # Example threshold, you can adjust it as needed

# Drop Outliers
data_no_outliers = data[(np.abs(z_scores) < threshold).all(axis=1)]
data_no_outliers


# In[86]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create boxplots for each column in SS_X
plt.figure(figsize=(10, 6))
sns.boxplot(data=SS_X)
plt.title('Boxplot of Standardized Data')
plt.xlabel('Features')
plt.ylabel('Standardized Values')
plt.xticks(rotation=45)
plt.show()


# In[87]:


data['Tenure'].max()


# In[88]:


Y = df["Churn"]
X = df.iloc[:, 3:]
X.head()


# In[90]:


#Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y, test_size=0.30)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# model fitting
logreg.fit(X_train,Y_train)
Y_pred_train = logreg.predict(X_train)
Y_pred_test = logreg.predict(X_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Training Accuracy score: ", ac1.round(2))
print("Testing Accuracy score: ", ac2.round(2))


# In[49]:


# cross validation method
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
from sklearn.model_selection import train_test_split

training_accuracy = []
test_accuracy = []
for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y, test_size=0.30,random_state=i)
    logreg.fit(X_train,Y_train)
    Y_pred_train = logreg.predict(X_train)
    Y_pred_test = logreg.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

print("Cross validation Training score: ", np.mean(training_accuracy).round(2))
print("Cross validation Test score: ", np.mean(test_accuracy).round(2))


# In[50]:


# k = 5, with cross validation

# cross validation method
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=9)
from sklearn.model_selection import train_test_split

training_accuracy = []
test_accuracy = []
for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y, test_size=0.30,random_state=i)
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test = KNN.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

print("Cross validation Training score: ", np.mean(training_accuracy).round(2))
print("Cross validation Test score: ", np.mean(test_accuracy).round(2))


# In[51]:


#Principal component Analysis
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(SS_X)
SS_X_pca = pca.transform(SS_X)


# In[52]:


SS_X_pca.shape


# In[54]:


pc_data = pd.DataFrame(SS_X_pca)


# In[55]:


pc_data.iloc[:,0].var()


# In[56]:


pca.explained_variance_ratio_


# In[57]:


pd.DataFrame(pca.explained_variance_ratio_)*100


# In[64]:


X_new = pc_data.iloc[:,0:7]


# In[65]:


# cross validation method
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=9)
from sklearn.model_selection import train_test_split

training_accuracy = []
test_accuracy = []
for i in range(1,101):
    X_train,X_test,Y_train,Y_test = train_test_split(X_new,Y, test_size=0.30,random_state=i)
    KNN.fit(X_train,Y_train)
    Y_pred_train = KNN.predict(X_train)
    Y_pred_test = KNN.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))

print("Cross validation Training score: ", np.mean(training_accuracy).round(2))
print("Cross validation Test score: ", np.mean(test_accuracy).round(2))


# In[63]:


KNN.predict(np.array([[2,2,2,2,2,2]]))


# In[7]:


import string
from collections import Counter

import matplotlib.pyplot as plt

# reading text file
text = open("BA - Assignment Part 2 data set.csv", encoding="utf-8").read()

# converting to lowercase
lower_case = text.lower()

# Removing punctuations
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# splitting text into words
tokenized_words = cleaned_text.split()

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# Removing stop words from the tokenized words list
final_words = []
for word in tokenized_words:
    if word not in stop_words:
        final_words.append(word)

# NLP Emotion Algorithm
# 1) Check if the word in the final word list is also present in emotion.txt
#  - open the emotion file
#  - Loop through each line and clear it
#  - Extract the word and emotion using split

# 2) If word is present -> Add the emotion to emotion_list
# 3) Finally count each emotion in the emotion list

emotion_list = []
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in final_words:
            emotion_list.append(emotion)

print(emotion_list)
w = Counter(emotion_list)
print(w)

# Plotting the emotions on the graph

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()


# In[ ]:





# In[ ]:




