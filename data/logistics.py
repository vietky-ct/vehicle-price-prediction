#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import underthesea as ud
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# In[2]:


df = pd.read_csv('./final1.csv', encoding = 'utf-8')


# In[3]:


#stopwords = pd.read_table('vietnamese-stopwords.txt', sep = '\\t', header = None, names = ['word'])
#stopwords


# In[4]:


df.head(2)


# In[5]:


df.info()


# In[6]:


#Chuyen gia tri null cua bien dung tich xi lanh = 6
df['mc_capacity'] = df['mc_capacity'].fillna(value = 6).reset_index(drop = True)
#Bo cac ad co lead tu 50 tro len
df = df[np.logical_and(df.duration < 169, df.duration > 1)].reset_index(drop = True)
df = df[df.lead > 1].reset_index(drop = True)
df = df[df.spending == 0].reset_index(drop = True)
df['mileage'] = df['mileage'].fillna(value = -1).reset_index(drop = True)


# In[7]:


df.info()


# In[8]:


#Bo cac thong tin khong can thiet
df_new = df.drop(['mc_brand_name', 'mc_model_name', 'ad_id', 'list_id', 'aa_date_time', 'sold_date', 'lead'], axis = 1)
#Bo cac ad co spending tu 40k tro len
#df_new = df_new[df_new.spending < 40000].reset_index(drop = True)
#Dua bien gia ve thang gia tri nho hon
df_new['price'] = df_new['price']/1000000
#Bo cac ad co gia tu 60tr tro len
df_new = df_new[df_new.price < 60].reset_index(drop = True)
#Sua lai bien brand lay 5 dong xe pho bien nhat, cac dong xe con lai gop chung thanh 1 nhom 6
df_new['brand'] = pd.Series([x if x <= 5 else 6 for x in df_new.brand])
#Bo het cac gia tri null
df_new = df_new.dropna().reset_index(drop = True)
#Tao dic cac model ban nhieu nhat, giu lai cac model co hon 100 ad, cac model con lai cho vao nhom 0
model_dict = dict(df_new.model.value_counts())
df_new['model'] = pd.Series([x if model_dict[x] > 500 else 0 for x in df_new.model])


# In[9]:


df_new.head()


# In[10]:


df_new.astype('object').describe()


# In[11]:


#x = df.duration
#IQR = x.quantile(0.75) - x.quantile(0.25)
#(x.quantile(0.25) - 1.5*IQR, 1.5*IQR + x.quantile(0.75))
df_new.price.count()/df.price.count()
#sns.boxplot(df.duration[np.logical_and(df.duration < 168, df.duration > 0)].reset_index(drop = True))


# In[12]:


#categorize duration
def cate_duration(x):
    if x >= 0 and x <= 48:
        value = '00_02_ngay'
    else:
        value = 'hon_2_ngay'
    return(value)

#categorize spending
def cate_spending(x):
    if x == 0:
        value = '00'
    elif x <= 20000:
        value = '20k'
    else:
        value = '40k'
    return(value)
#categorize mileage
def cate_mileage(x):
    if x >= 0 and x <= 5000 :
        value = '05k'
    elif x > 5000 and x <= 15000:
        value = '05k_15k'
    elif x > 15000 and x <= 30000:
        value = '15k_30k'
    elif x > 30000 and x <= 60000:
        value = '30k_60k'
    elif x > 60000:
        value = '60k'
    else:
        value = 'None'
    return value


# In[13]:


df_new['duration'] = df_new.duration.apply(lambda x: cate_duration(x))
#df_new['spending'] = df_new.spending.apply(lambda x: cate_spending(x))
df_new['mileage'] = df_new.mileage.apply(lambda x: cate_mileage(x))
df_new['regdate'] = df_new['regdate'].astype('str')


# In[14]:


df_new.head()


# In[15]:


#label ouput, 1: ban trong nua ngay, 0: ban tu 1 ngay tro len
df_new['duration'] = pd.Series([1 if x == 'hon_2_ngay' else 0 for x in df_new.duration])
#df_new = df_new[df_new.spending == '00'].reset_index(drop = True)
#df_new = df_new.drop('spending', axis = 1)
df_new.head()


# In[16]:


df_new.groupby('duration').count()
32/(32+13)


# In[17]:


#df_new.to_csv('final_preprocess.csv')


# In[18]:


#def remove(text):
 #   rm0 = str(text).lower()
  #  rm1 = re.sub('[0-9]', '', str(rm0))
   # rm2 = re.sub('[!@#$%\.\-\+//():]', '', str(rm1))
    #rm3 = ud.word_tokenize(str(rm2), format='text')
    #return rm3


# In[19]:


#df_new['subject'] = df_new.subject.apply(remove)
#df_new['body'] = df_new.body.apply(remove)
#stopwords['word'] = stopwords.word.apply(remove)
#stopwords = stopwords.word.values.tolist()
#df_1['mc_model_name'] = df_1.mc_model_name.apply(remove_space)


# In[20]:


#stopwords


# In[21]:


X = df_new.drop('duration', axis = 1)
y = df_new['duration']


# In[22]:


def prepare(X, y, test_size = 0.2, new_case = False, X_new = None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

    #tfidf_body = TfidfVectorizer(max_features=1000).fit(X_train.body)
    #tfidf_subject = TfidfVectorizer(max_features=5000).fit(X_train.subject)
    onehot_brand = OneHotEncoder().fit(X_train.brand.values.reshape(-1,1))
    onehot_model = OneHotEncoder().fit(X_train.model.values.reshape(-1,1))
    #onehot_spending = OneHotEncoder().fit(X_train.spending.values.reshape(-1,1))
    onehot_regdate = OneHotEncoder().fit(X_train.regdate.values.reshape(-1,1))
    onehot_mileage = OneHotEncoder().fit(X_train.mileage.values.reshape(-1,1))
    onehot_region =  OneHotEncoder().fit(X_train.region.values.reshape(-1,1))
    onehot_mctype =  OneHotEncoder().fit(X_train.mc_type.values.reshape(-1,1))
    onehot_mccapa =  OneHotEncoder().fit(X_train.mc_capacity.values.reshape(-1,1))


    #Convert
    if new_case == False:
        #X_train_body = tfidf_subject.transform(X_train.body).toarray()
        #X_test_body = tfidf_subject.transform(X_test.body).toarray()

        #X_train_subject = tfidf_subject.transform(X_train.subject).toarray()
        #X_test_subject = tfidf_subject.transform(X_test.subject).toarray()

        X_train_brand = onehot_brand.transform(X_train.brand.values.reshape(-1,1)).toarray()
        X_test_brand = onehot_brand.transform(X_test.brand.values.reshape(-1,1)).toarray()

        X_train_model = onehot_model.transform(X_train.model.values.reshape(-1,1)).toarray()
        X_test_model = onehot_model.transform(X_test.model.values.reshape(-1,1)).toarray()

        #X_train_spending = onehot_spending.transform(X_train.spending.values.reshape(-1,1)).toarray()
        #X_test_spending = onehot_spending.transform(X_test.spending.values.reshape(-1,1)).toarray()

        X_train_regdate = onehot_regdate.transform(X_train.regdate.values.reshape(-1,1)).toarray()
        X_test_regdate = onehot_regdate.transform(X_test.regdate.values.reshape(-1,1)).toarray()

        X_train_mileage = onehot_mileage.transform(X_train.mileage.values.reshape(-1,1)).toarray()
        X_test_mileage = onehot_mileage.transform(X_test.mileage.values.reshape(-1,1)).toarray()

        X_train_region = onehot_region.transform(X_train.region.values.reshape(-1,1)).toarray()
        X_test_region = onehot_region.transform(X_test.region.values.reshape(-1,1)).toarray()

        X_train_mctype = onehot_mctype.transform(X_train.mc_type.values.reshape(-1,1)).toarray()
        X_test_mctype = onehot_mctype.transform(X_test.mc_type.values.reshape(-1,1)).toarray()

        X_train_mccapa = onehot_mccapa.transform(X_train.mc_capacity.values.reshape(-1,1)).toarray()
        X_test_mccapa = onehot_mccapa.transform(X_test.mc_capacity.values.reshape(-1,1)).toarray()

        X_train_price = np.log10(X_train.price.values.reshape(-1, 1))
        X_test_price = np.log10(X_test.price.values.reshape(-1, 1))

        X_train_full = np.concatenate([X_train_price,X_train_mileage,X_train_brand, X_train_model, X_train_regdate, X_train_region, X_train_mctype, X_train_mccapa], axis = 1)
        X_test_full = np.concatenate([X_test_price,X_test_mileage,X_test_brand, X_test_model, X_test_regdate, X_test_region, X_test_mctype, X_test_mccapa], axis = 1)

        #X_train_full = np.concatenate([X_train_body,X_train_subject,X_train_brand, X_train_model, X_train_spending, X_train_regdate, X_train_region, X_train_mctype, X_train_mccapa], axis = 1)
        #X_test_full = np.concatenate([X_test_body,X_test_subject,X_test_brand, X_test_model, X_test_spending, X_test_regdate, X_test_region, X_test_mctype, X_test_mccapa], axis = 1)
        y_train_full = y_train
        y_test_full = y_test

        return (X_train_full, X_test_full, y_train_full, y_test_full)


# In[23]:


X_train, X_test, y_train, y_test = prepare(X,y)


# In[24]:


from sklearn.linear_model import LogisticRegression
lgmod = LogisticRegression()
lgmod.fit(X_train, y_train)
y_pred = lgmod.predict(X_test)
lgmod.score(X_test, y_test)


# In[25]:


y_pred = (1 - (lgmod.predict_proba(X_test)[:,0] > 0.645).astype('int64'))
y_pred


# In[26]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[27]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[28]:


from sklearn.model_selection import cross_val_score
cv_10 = cross_val_score(LogisticRegression(), X_train, y_train, cv = 10)


# In[29]:


np.mean(cv_10)


# In[ ]:





# In[30]:


from sklearn.ensemble import RandomForestClassifier
rfmod = RandomForestClassifier(n_estimators=200, max_depth=20)
rfmod.fit(X_train, y_train)
y_pred = rfmod.predict(X_test)
rfmod.score(X_test, y_test)


# In[31]:


y_pred = (1 - (rfmod.predict_proba(X_test)[:,0] > 0.642).astype('int64'))
y_pred


# In[32]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[33]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))


# In[44]:

filename = './finalized_model.sav'
s = pickle.dump(rfmod, open(filename, 'wb'))

