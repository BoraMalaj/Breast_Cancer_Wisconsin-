#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
df.head()


# In[3]:


from sklearn.preprocessing import LabelEncoder

# Drop unnecessary columns
df = df.drop(['id', 'Unnamed: 32'], axis=1)

# Encode target M=1, B=0
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

print("Data prepared!")
print("Shape:", df.shape)
print("\nClass Distribution:")
print(df['diagnosis'].value_counts())
print("B (Benign)=0, M (Malignant)=1")


# In[5]:


from sklearn.model_selection import train_test_split

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Data split successfully!")
print(f"Train size : {X_train.shape}")
print(f"Test size  : {X_test.shape}")


# In[25]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500, random_state=60)
rf.fit(X_train, y_train)

print("Random Forest trained successfully!")
print("Accuracy:", round(rf.score(X_test, y_test), 2))


# In[27]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred_rf = rf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred_rf,
      target_names=['Benign', 'Malignant']))


# In[29]:


cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# In[13]:


importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
importances.head(15).plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Top 15 Feature Importances - Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("\nTop 5 Most Important Features:")
print(importances.head())


# In[15]:


from sklearn.metrics import roc_curve, roc_auc_score

y_prob_rf = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
auc = roc_auc_score(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='steelblue', linewidth=2,
         label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--',
         label='Random Classifier')
plt.title('ROC Curve - Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(f"AUC Score: {round(auc, 2)}")


# In[17]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')

print("Cross Validation Results (5-Fold):")
print(f"Scores : {scores.round(2)}")
print(f"Mean   : {scores.mean().round(2)}")
print(f"Std    : {scores.std().round(2)}")

