import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset=pd.read_csv('development_dataset.csv')
dataset2=pd.read_csv('leaderboard_dataset.csv')


sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')


for i in dataset2.columns:
     print(i,dataset2[i].isna().sum())

dataset=dataset.drop('VAR17',axis=1)

dataset=dataset.drop('VAR1',axis=1)
dataset2=dataset2.drop('VAR17',axis=1)

dataset2=dataset2.drop('VAR1',axis=1)


 
pd.value_counts(ytest).plot.bar()
plt.title('Click class histogram')
plt.xlabel('Class')
plt.ylabel('VAR21')
data['is_click'].value_counts()


dataset['VAR21'].replace({'Low': 0, 'Medium': 1, 'High': 2}, inplace=True)
y=dataset['VAR21'].values

dataset=dataset.drop('VAR21',axis=1)

dataset['VAR2'].fillna((dataset['VAR2'].mean()), inplace=True)
dataset['VAR3'].fillna((dataset['VAR3'].mean()), inplace=True)
dataset['VAR5'].fillna((dataset['VAR5'].mean()), inplace=True)
dataset['VAR6'].fillna((dataset['VAR6'].mean()), inplace=True)
dataset['VAR7'].fillna((dataset['VAR7'].mean()), inplace=True)
dataset['VAR8'].fillna((dataset['VAR8'].mean()), inplace=True)
dataset['VAR10'].fillna((dataset['VAR10'].mean()), inplace=True)
dataset['VAR11'].fillna((dataset['VAR11'].mean()), inplace=True)
dataset['VAR13'].fillna((dataset['VAR13'].mean()), inplace=True)
dataset['VAR15'].fillna((dataset['VAR15'].mean()), inplace=True)
dataset['VAR16'].fillna((dataset['VAR16'].mean()), inplace=True)
dataset['VAR4'].fillna((dataset['VAR4'].mean()), inplace=True)
dataset['VAR9'].fillna((dataset['VAR9'].mean()), inplace=True)
dataset['VAR12'].fillna((dataset['VAR12'].mean()), inplace=True)

dataset['VAR17'].fillna((dataset['VAR17'].mean()), inplace=True)
dataset2['VAR17'].fillna((dataset2['VAR17'].mean()), inplace=True)



dataset['VAR14']=dataset['VAR14'].replace(to_replace='.', value=1)

dataset2['VAR2'].fillna((dataset2['VAR2'].mean()), inplace=True)
dataset2['VAR3'].fillna((dataset2['VAR3'].mean()), inplace=True)
dataset2['VAR5'].fillna((dataset2['VAR5'].mean()), inplace=True)
dataset2['VAR6'].fillna((dataset2['VAR6'].mean()), inplace=True)
dataset2['VAR7'].fillna((dataset2['VAR7'].mean()), inplace=True)
dataset2['VAR8'].fillna((dataset2['VAR8'].mean()), inplace=True)
dataset2['VAR10'].fillna((dataset2['VAR10'].mean()), inplace=True)
dataset2['VAR11'].fillna((dataset2['VAR11'].mean()), inplace=True)
dataset2['VAR13'].fillna((dataset2['VAR13'].mean()), inplace=True)
dataset2['VAR15'].fillna((dataset2['VAR15'].mean()), inplace=True)
dataset2['VAR16'].fillna((dataset2['VAR16'].mean()), inplace=True)
dataset2['VAR4'].fillna((dataset2['VAR4'].mean()), inplace=True)
dataset2['VAR9'].fillna((dataset2['VAR9'].mean()), inplace=True)
dataset2['VAR12'].fillna((dataset2['VAR12'].mean()), inplace=True)
dataset2['VAR14']=dataset2['VAR14'].replace(to_replace='.', value=1)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
dataset = sc_X.fit_transform(dataset)
dataset2 = sc_X.transform(dataset2)


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
model=OneVsRestClassifier(RandomForestClassifier(n_estimators=100,random_state=0))
model.fit(dataset, y)
model.score(dataset,y)


from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import svc
model=OneVsOneClassifier(LinearSVC(random_state=0))
model.fit(dataset, y)
model.score(dataset,y)


    
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
clf=RandomForestClassifier(n_estimators=100,random_state=0)
clf.fit(dataset, y)
clf.score(dataset, y)
ytest=clf.predict(dataset2)



# Step 5: Fit a AdaBoost model, " compared to "Decision Tree model, accuracy go up by 10%
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(dataset, y)
ytest=clf.predict(dataset2)
accuracy_score(y_test, y_pred)
OUTPUT:0.833
# Step 6: Fit a Gradient Boosting model, " compared to "Decision Tree model, accuracy go up by 10%
clf = GradientBoostingClassifier(n_estimators=90)
clf.fit(dataset, y)
ytest=clf.predict(dataset2)
clf.score(dataset,y)



clf=GradientBoostingClassifier(n_estimators=5,learning_rate=0.1, min_samples_split=50,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=0)
clf.fit(dataset, y)
ytest=clf.predict(dataset2)
clf.score(dataset,y)



ytest=classifier.predict(dataset2)


from sklearn.model_selection import cross_val_score
ac=cross_val_score(estimator=clf,X=dataset,y=y,cv=10)
 

from sklearn.model_selection import GridSearchCV
params={
        'n_estimators':[100],
        'min_samples_split':range(1000,2000,200), 
        'min_samples_leaf':range(30,71,10),
        "max_depth":range(5,10,2), 
        'min_samples_split':range(200,1001,200),
        
        'max_features':range(7,20,2),
        'subsample':[0.6,0.7,0.8,0.9],
        
        
        
        }

clf2=GridSearchCV(estimator=clf,param_grid=params,scoring="accuracy",n_jobs=-1,cv=5)
clf2.fit(dataset,y)
ytest=clf2.predict(dataset2)
clf2.best_score_
clf2.best_params_
clf2.score(dataset,y)




from xgboost import XGBClassifier
classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=90, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)

classifier.fit(dataset, y)
classifier.score(dataset,y)
ytest=classifier.predict(dataset2)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(dataset, y)


param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(dataset,y)
gsearch1.predict(dataset2)


param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_






from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,dataset,y,cv=10)
score.mean()
ytest=classifier.predict(dataset2)


df = pd.DataFrame(ytest,index=range(1,10001))

df.replace({ 0:'Low',1: 'Medium',  2:'High'}, inplace=True)
df.to_csv('Rebellion_IITRoorkee_1.csv')

    
    
    
    
    
    
    
    
    
    
    
    