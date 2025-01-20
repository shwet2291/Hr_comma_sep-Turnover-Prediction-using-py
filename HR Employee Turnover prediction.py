#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Downoloading data
hr = pd.read_csv("E:/python/HR_comma_sep.csv")
hr.head()


# In[3]:


#Common dataset's information
hr.info()


# In[4]:


#Checking nulls within data
hr.isnull().any()


# In[5]:


#Creating df containing info about Department personnel 
personnel = {
    'Department':["sales","technical","support","IT","product_mng","marketing","RandD","accounting","hr","management"],
    'Number of Staff' :[4140,2720,2229,1227,902,858,787,767,739,630]
}
hr_staff = pd.DataFrame(personnel)
#Visualization of employee division by Department
ax = sns.barplot(data = hr_staff, x = "Number of Staff", y = "Department")
sns.set_style("white")
ax.set_title('Number of employees by Department', fontsize=12)
ax.bar_label(ax.containers[0])


# In[6]:


#Gathering info about quantity of left
hr.left.value_counts()
#Visualization of Left/Stayed proportion amongst employees
left_count = hr['left'].value_counts()

labels = ['Stayed', 'Left']
sizes = left_count.values
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%',startangle=140, 
       colors = sns.color_palette("Paired", 8))
plt.title('Left/Stayed Proportion')
plt.show()


# In[7]:


#Selecting data for department/left visualization
hr_left = hr[["Department", "left"]].copy()
#Barplot orders
plot_order_1 = hr_left.groupby('Department')['left'].mean().sort_values(ascending=False).index.values
plot_order_1

plot_order_2 = hr_left.groupby('Department')['left'].sum().sort_values(ascending=False).index.values
plot_order_2


# In[8]:


#Left ratio barplot
sns.set(rc={"figure.figsize":(7, 5)})
sns.set_style("white")
sns_hr_left = sns.barplot(data = hr_left, x = 'left', y = 'Department',errorbar=None, order =plot_order_1)
sns_hr_left.set_title('Left ratio by Department', fontsize=12)
sns_hr_left.set_xlabel('Left', fontsize=12)
sns_hr_left.set_ylabel('Department', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=6)
sns_hr_left.bar_label(sns_hr_left.containers[0])


# In[9]:


#Total number of left by Department barplot
sns.set(rc={"figure.figsize":(7, 5)})
sns.set_style("white")
sns_hr_left = sns.barplot(data = hr_left, x = 'left', y = 'Department',estimator=np.sum,errorbar=None,
                         order = plot_order_2)
sns_hr_left.set_title('Number of retired personnel by Department', fontsize=12)
sns_hr_left.set_xlabel('Left', fontsize=12)
sns_hr_left.set_ylabel('Department', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=6)
sns_hr_left.bar_label(sns_hr_left.containers[0])


# In[10]:


#Salary distribution by left
pd.crosstab(hr.salary,hr.left).plot(kind='bar')


# In[11]:


#"Recent" promotion by those who left or not
pd.crosstab(hr.promotion_last_5years,hr.left).plot(kind='bar')


# In[12]:


#Creating selected data for corr matrix
hr_corr = hr.drop(['Department', 'salary'], axis= 1 )
#Visualization of correlation matrix
matrix = np.triu(hr_corr.corr())
sns.heatmap(hr_corr.corr(), annot=True, mask=matrix)


# In[13]:


#Criteria for new column that displays overwork
def overworking(value):
    if value < 120:
        return "Below Normal"
    elif 120 <= value <= 160:
        return "Normal"
    elif 160 < value < 240:
        return "Above Normal"
    else:
        return "Extreme overworking"


# In[14]:


#Implementing new column based on original data
hr['overwork_status'] = hr['average_montly_hours'].map(overworking)
hr.head()
#Salary distribution within various overwork status
pd.crosstab(hr.salary,hr.overwork_status).plot(kind='bar')


# In[15]:


#Distribution of salaries by length of service of employees
pd.crosstab(hr.salary,hr.time_spend_company).plot(kind='bar')


# In[16]:


#Criteria for new column that evaluates fairness of wages
mask = (hr["time_spend_company"] >= 5) & \
       (hr["promotion_last_5years"] < 1) & \
       (hr["salary"] == "low")
hr.loc[ mask, 'fair_remuneration'] = 'non-fair'
hr.loc[~mask, 'fair_remuneration'] = 'fair'
hr.head()


# In[17]:


#Display of satisfaction level distribution by average working hours
sns.scatterplot(x="satisfaction_level",
                    y="average_montly_hours",hue = 'number_project',data=hr)


# In[18]:


#Encoding non-integer data to int
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()
encoders = {}
text_features = ['Department', 'salary', 'overwork_status', 'fair_remuneration']
for feature in text_features: 
    encoders[feature] = le
    hr[feature+"_encoded"] = encoders[feature].fit_transform(hr[feature])


# In[19]:


#Selecting features of model
features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 
            'time_spend_company','Work_accident','promotion_last_5years', 'Department_encoded',
           'salary_encoded', 'overwork_status_encoded', 'fair_remuneration_encoded']


# In[20]:


#Selecting target variable
X = hr[features]
y = hr['left']


# In[21]:


#Splitting data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[22]:


#In order to bring mean value to 0 using StandardScaler
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[23]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
ID3classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 100, max_depth=4) 
ID3classifier.fit(X_train, y_train)
#Confusion matrix
from sklearn import metrics
y_pred = ID3classifier.predict(X_test)
actual = y_test
predicted = y_pred
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title('Decision Tree Classifier')
plt.show()
#Main metrics for classifier models
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
ID3accuracy= accuracy_score(y_test,y_pred)
ID3recall = recall_score(y_test,y_pred)
ID3precision = precision_score(y_test,y_pred)
ID3f1 = f1_score(y_test, y_pred)
print('Accuracy:',ID3accuracy)
print('Recall:',ID3recall)
print('Precision:',ID3precision)
print('F1_Score:',ID3f1)
#Decision tree ROC-curve
y_pred_proba = ID3classifier. predict_proba (X_test)[::,1]
fpr, tpr, _ = metrics. roc_curve (y_test, y_pred_proba)
ID3auc = metrics. roc_auc_score (y_test, y_pred_proba)
plt.plot (fpr,tpr,label=" AUC= "+str(ID3auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Decision Tree ROC-AUC')
plt.legend(loc=4)
plt.show()


# In[24]:


#Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
RFclassifier = RandomForestClassifier(n_estimators = 400, max_depth = 4, criterion = 'entropy',random_state = 0)
RFclassifier.fit(X_train,y_train)
#Random Forest confusion matrix
y_pred = RFclassifier.predict(X_test)
actual = y_test
predicted = y_pred

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title('Random Forest Classifier')
plt.show()
#Random Forest metrics
RFaccuracy= accuracy_score(y_test,y_pred)
RFrecall = recall_score(y_test,y_pred)
RFprecision = precision_score(y_test,y_pred)
RFf1 = f1_score(y_test, y_pred)

print('Accuracy:',RFaccuracy)
print('Recall:',RFrecall)
print('Precision:',RFprecision)
print('F1_score:',RFf1)
#Random Forest ROC curve
y_pred_proba = RFclassifier. predict_proba (X_test)[::,1]
fpr, tpr, _ = metrics. roc_curve (y_test, y_pred_proba)
RFauc = metrics. roc_auc_score (y_test, y_pred_proba)


plt.plot (fpr,tpr,label=" AUC= "+str(RFauc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Random Forest ROC-AUC')
plt.legend(loc=4)
plt.show()


# In[25]:


#Logistic Regression classifier
from sklearn.linear_model import LogisticRegression 

LGclassifier = LogisticRegression(random_state = 0) 
LGclassifier.fit(X_train, y_train)
#Logistic Regression confusion matrix
y_pred = LGclassifier.predict(X_test)
actual = y_test
predicted = y_pred
confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title('Logistic Regression Classifier')
plt.show()
#Logistic Regression metrics
LGaccuracy= accuracy_score(y_test,y_pred)
LGrecall = recall_score(y_test,y_pred)
LGprecision = precision_score(y_test,y_pred)
LGf1 = f1_score(y_test,y_pred)

print('Accuracy:',LGaccuracy)
print('Recall:',LGrecall)
print('Precision:',LGprecision)
print('F1_Score:',LGf1)
#Logistic Regression ROC curve
y_pred_proba = LGclassifier. predict_proba (X_test)[::,1]
fpr, tpr, _ = metrics. roc_curve (y_test, y_pred_proba)
LGauc = metrics. roc_auc_score (y_test, y_pred_proba)


plt.plot (fpr,tpr,label=" AUC= "+str(LGauc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Logistic Regression ROC-AUC')
plt.legend(loc=4)
plt.show() 


# In[26]:


#SVM classifier
from sklearn.svm import SVC

SVMclassifier = SVC(kernel = 'linear', random_state = 0, probability = True)
SVMclassifier.fit(X_train, y_train)
#SVM confusion matrix
y_pred = SVMclassifier.predict(X_test)
actual = y_test
predicted = y_pred
confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.title('SVM Classifier')
plt.show()
#SVM metrics
SVMaccuracy= accuracy_score(y_test,y_pred)
SVMrecall = recall_score(y_test,y_pred)
SVMprecision = precision_score(y_test,y_pred)
SVMf1 = f1_score(y_test,y_pred)

print('Accuracy:',SVMaccuracy)
print('Recall:',SVMrecall)
print('Precision:',SVMprecision)
print('F1_Score:',SVMf1)
#SVM ROC curve
y_pred_proba = SVMclassifier. predict_proba (X_test)[::,1]
fpr, tpr, _ = metrics. roc_curve (y_test, y_pred_proba)
SVMauc = metrics. roc_auc_score (y_test, y_pred_proba)


plt.plot (fpr,tpr,label=" AUC= "+str(SVMauc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('SVM ROC-AUC')
plt.legend(loc=4)
plt.show()


# In[27]:


# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Initialize KNN Classifier
KNNclassifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)  # n_neighbors=5 by default
KNNclassifier.fit(X_train, y_train)

# KNN confusion matrix
y_pred = KNNclassifier.predict(X_test)
actual = y_test
predicted = y_pred
confusion_matrix = confusion_matrix(actual, predicted)

cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot()
plt.title('KNN Classifier')
plt.show()

# KNN metrics
KNNaccuracy = accuracy_score(y_test, y_pred)
KNNrecall = recall_score(y_test, y_pred)
KNNprecision = precision_score(y_test, y_pred)
KNNf1 = f1_score(y_test, y_pred)

print('Accuracy:', KNNaccuracy)
print('Recall:', KNNrecall)
print('Precision:', KNNprecision)
print('F1_Score:', KNNf1)

# KNN ROC curve
if hasattr(KNNclassifier, 'predict_proba'):
    y_pred_proba = KNNclassifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    KNNauc = roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr, tpr, label="AUC= " + str(KNNauc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('KNN ROC-AUC')
    plt.legend(loc=4)
    plt.show()
else:
    print("ROC-AUC cannot be calculated because `predict_proba` is not available for this classifier.")


# In[28]:


#Creating new df containing data about models and its metrics comparison
data = [['Decision Tree', ID3accuracy, ID3recall, ID3precision,  ID3f1 ],
        ['KNN Classifier', KNNaccuracy, KNNrecall, KNNprecision,  KNNf1 ],
        ['Random Forest', RFaccuracy, RFrecall, RFprecision,  RFf1],
        ['Logistic Regression', LGaccuracy, LGrecall, LGprecision,  LGf1],
        ['SVM', SVMaccuracy, SVMrecall, SVMprecision,  SVMf1]]

score_df = pd.DataFrame(data, columns=['Model', 'Accuracy', 'Recall', 'Precision', 'F1_Score'])
score_df


# In[29]:


#5 models ROC curve comparison

plt.figure(0).clf()


#Decision Tree
y_pred_proba = ID3classifier. predict_proba (X_test)[::,1]
fpr, tpr, _ = metrics. roc_curve (y_test, y_pred_proba)
ID3auc = round(metrics. roc_auc_score (y_test, y_pred_proba),4)
plt.plot (fpr,tpr,label="Decision Tree AUC= "+str(ID3auc))

#KNN Classifier
y_pred_proba = KNNclassifier. predict_proba (X_test)[::,1]
fpr, tpr, _ = metrics. roc_curve (y_test, y_pred_proba)
KNNauc = round(metrics. roc_auc_score (y_test, y_pred_proba),4)
plt.plot (fpr,tpr,label="KNN Classifier AUC= "+str(ID3auc))


#Random Forest
y_pred_proba = RFclassifier. predict_proba (X_test)[::,1]
fpr, tpr, _ = metrics. roc_curve (y_test, y_pred_proba)
RFauc = round(metrics. roc_auc_score (y_test, y_pred_proba),4)
plt.plot (fpr,tpr,label="Random Forest AUC= "+str(RFauc))

#Logistic Regression
y_pred_proba = LGclassifier. predict_proba (X_test)[::,1]
fpr, tpr, _ = metrics. roc_curve (y_test, y_pred_proba)
LGauc = round(metrics. roc_auc_score (y_test, y_pred_proba),4)
plt.plot (fpr,tpr,label="Logistic Regression AUC= "+str(LGauc))

#SVM
y_pred_proba = SVMclassifier. predict_proba (X_test)[::,1]
fpr, tpr, _ = metrics. roc_curve (y_test, y_pred_proba)
SVMauc = round(metrics. roc_auc_score (y_test, y_pred_proba),4)
plt.plot (fpr,tpr,label="SVM AUC= "+str(SVMauc))


plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC curves comparison')
plt.legend()


# In[ ]:





# In[ ]:




