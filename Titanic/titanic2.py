#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : titanic2.py
# @Author: Jingjie Jin
# @Date  : 2018/1/21

# 数据处理
import numpy as np
import pandas as pd

#绘图
import seaborn as sns
import matplotlib.pyplot as plt

#各种模型、数据处理方法
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import warnings
warnings.filterwarnings('ignore')

'''数据读入'''
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine_df = pd.concat([train_df, test_df])
# print(combine_df.loc[combine_df['Fare'].isnull()])
# test.loc[test['Fare'].isnull(), 'Fare'] = \
# test[(test['Pclass'] == 3) & (test['Embarked'] == 'S') & (['Sex'] == 'male')].dropna().Fare.mean()


'''1.Name
将名字的长度也提取出来，会发现，名字的长度和是否获救相关:名字越长，获得概率越高？
'''
# train_df.groupby(train_df['Name'].apply(lambda x:len(x)))['Survived'].mean().plot()
# plt.show()

combine_df['NameSize'] = combine_df['Name'].apply(lambda x: len(x))
# cut将根据值本身来选择箱子均匀间隔，qcut是根据这些值的频率来选择箱子的均匀间隔
combine_df['NameSize'] = pd.qcut(combine_df['NameSize'],5) # 分箱
# combine_df.groupby(combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0]))['Survived'].mean().plot()
# plt.show()
titles = combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
combine_df['Title'] = titles
combine_df['Title'] = combine_df['Title'].replace(['Don','Dona', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir','Dr', 'Master'],'Mr')
combine_df['Title'] = combine_df['Title'].replace(['Mlle','Ms'], 'Miss')
combine_df['Title'] = combine_df['Title'].replace(['the Countess','Mme','Lady','Dr'], 'Mrs')
df = pd.get_dummies(combine_df['Title'],prefix='Title')
combine_df = pd.concat([combine_df,df],axis=1)

'''2.Family
有女性死亡的家庭 & 有男性存活的家庭
例如：有一个family有一个女性死亡，这个family其他的女性的死亡概率也比较高。
'''
combine_df['Fname'] = combine_df['Name'].apply(lambda x:x.split(',')[0])
combine_df['FamilySize'] = combine_df['SibSp']+combine_df['Parch']
dead_female_Fname = list(set(combine_df[(combine_df.Sex=='female') & (combine_df.Age>=12)
                              & (combine_df.Survived==0) & (combine_df.FamilySize>1)]['Fname'].values))
survive_male_Fname = list(set(combine_df[(combine_df.Sex=='male') & (combine_df.Age>=12)
                              & (combine_df.Survived==1) & (combine_df.FamilySize>1)]['Fname'].values))
combine_df['Dead_female_family'] = np.where(combine_df['Fname'].isin(dead_female_Fname),1,0)
combine_df['Survive_male_family'] = np.where(combine_df['Fname'].isin(survive_male_Fname),1,0)
combine_df = combine_df.drop(['Name','Fname'],axis=1)

'''3.Age
添加Child
对Age离散化
'''
group = combine_df.groupby(['Title', 'Pclass'])['Age']
combine_df['Age'] = group.transform(lambda x: x.fillna(x.median()))
combine_df = combine_df.drop('Title', axis = 1)
combine_df['IsChild'] = np.where(combine_df['Age'] <= 12,1,0) # 如果小于12岁则标记为1，否则0
combine_df['Age'] = pd.cut(combine_df['Age'],5)
combine_df = combine_df.drop('Age',axis=1)
# print(combine_df['Age'].value_counts())

'''4.FamilySize
离散化
'''
# 家庭成员没有其他人（==0）时标记为'solo'，小于等于3时标记为'normal'，大于3时标记为'big'
combine_df['FamilySize'] = np.where(combine_df['FamilySize'] == 0, 'solo',
                                    np.where(combine_df['FamilySize'] <= 3, 'normal','big'))
df = pd.get_dummies(combine_df['FamilySize'], prefix='FamilySize')
combine_df = pd.concat([combine_df, df], axis=1).drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
# print(combine_df.info())

'''5.Ticket

'''
combine_df['Ticket_Lett'] = combine_df['Ticket'].apply(lambda x: str(x)[0])
# combine_df['Ticket_Lett'] = combine_df['Ticket_Lett'].apply(lambda x: str(x))
combine_df['High_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['1', '2', 'P']),1,0)
combine_df['Low_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['A','W','3','7']),1,0)
combine_df = combine_df.drop(['Ticket','Ticket_Lett'],axis=1)
# print(combine_df.info())

'''6.Embarked
缺省值用S填充
'''
combine_df.Embarked = combine_df.Embarked.fillna('S')
df = pd.get_dummies(combine_df['Embarked'],prefix='Embarked')
combine_df = pd.concat([combine_df,df],axis=1).drop('Embarked',axis=1)
# print(combine_df.info())

'''7.Cabin
由于缺省值过多，直接分为【有，无】数据两类
'''
combine_df['Cabin_isNull'] = np.where(combine_df['Cabin'].isnull(), 0, 1)
combine_df = combine_df.drop('Cabin', axis=1)
# print(combine_df.info())

'''8.Pclass
'''
df = pd.get_dummies(combine_df['Pclass'], prefix='Pclass')
combine_df = pd.concat([combine_df, df], axis=1).drop('Pclass', axis=1)

'''9.Sex
'''
df = pd.get_dummies(combine_df['Sex'], prefix='Sex')
combine_df = pd.concat([combine_df, df], axis=1).drop('Sex',axis=1)

'''10.Fare
离散化，按频率来均匀分箱，分三份
缺省值用众数填充，之后进行离散化
'''
# mode()函数找出Fare的众数，填充缺省值
combine_df['Fare'].fillna(combine_df['Fare'].mode()[0], inplace=True)
# print(combine_df['Fare'].value_counts())
combine_df['Fare'] = pd.qcut(combine_df['Fare'], 3)
df = pd.get_dummies(combine_df['Fare'], prefix='Fare').drop('Fare_(-0.001, 8.662]',axis=1)
# print(combine_df['Fare'].value_counts())
combine_df = pd.concat([combine_df, df], axis=1).drop('Fare', axis=1)
# print(combine_df.info())

'''所有特征转化成数值型编码
'''
features = combine_df.drop(['PassengerId', 'Survived'], axis=1).columns
# print(features)
le = LabelEncoder()
for feature in features:
    le = le.fit(combine_df[feature])
    combine_df[feature] = le.transform(combine_df[feature])

'''得到训练/测试数据
'''
X_all = combine_df.iloc[:891,:].drop(['PassengerId', 'Survived'], axis=1)
Y_all = combine_df.iloc[:891,:]['Survived']
X_test = combine_df.iloc[891:,:].drop(['PassengerId', 'Survived'], axis=1)
# print(X_all.iloc[0,:])

'''模型调优
'''
# lr = LogisticRegression()
# rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight={0:0.745, 1:0.255})
# gbdt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=3)
# svc = SVC()
# dt = DecisionTreeClassifier()
# knn = KNeighborsClassifier(n_neighbors=3)

# clfs = [lr, svc, knn, dt, rf, gbdt]
# kf = 10
# cv_results = []
# for classifier in clfs:
#     cv_results.append(cross_val_score(classifier, X_all, Y_all, scoring='accuracy', cv = kf, n_jobs=4))
# #
# # print(np.mat(cv_results).shape)
# cv_means = []
# cv_std = []
#
# for cv_result in cv_results:
#     cv_means.append(cv_result.mean())
#     cv_std.append(cv_result.std())
# ag = ['lr', 'svc', 'knn', 'dt', 'rf', 'gbdt']
#
# for i in range(6):
#     print(ag[i], cv_means[i])
#
# cv_res = pd.DataFrame({'CrossValMeans': cv_means, 'CrossValerrors': cv_std,
#                        'Algorithm': ['LR','SVC','KNN','DT','RF','GBDT']})
# gr = sns.barplot('CrossValMeans', 'Algorithm', data=cv_res, palette='Set3',orient = "h",**{'xerr':cv_std})
# gr.set_xlabel('Mean Accuracy')
# gr = gr.set_title('Cross Validation Scores')
# plt.show()


class Ensemble(object):

    def __init__(self, estimators):
        self.estimator_names = []
        self.estimators = []
        for i in estimators:
            self.estimator_names.append(i[0])
            self.estimators.append(i[1])
        self.clf = LogisticRegression()

    def fit(self, x_train, y_train):
        for i in self.estimators:
            i.fit(x_train, y_train)
        x = np.array([i.predict(x_train) for i in self.estimators]).T
        y = y_train
        self.clf.fit(x, y)

    def predict(self, x):
        x = np.array([i.predict(x) for i in self.estimators]).T
        return self.clf.predict(x)

    def score(self, x, y):
        s = precision_score(y, self.predict(x))
        return s

lr = LogisticRegression()
svc = SVC()
rf = RandomForestClassifier(n_estimators=300,min_samples_leaf=4,class_weight={0:0.745,1:0.255})
gbdt = GradientBoostingClassifier(n_estimators=500,learning_rate=0.03,max_depth=3)
xgbt = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.03)

ensemble = Ensemble([('lr',lr),('rf',rf),('svc',svc),('gbdt',gbdt),('xgbt',xgbt)])
score = 0
for i in range(0, 10):
    num_test = 0.20
    x_train, x_cv, y_train, y_cv = train_test_split(X_all.values, Y_all.values, test_size=num_test)
    ensemble.fit(x_train, y_train)
#     # Y_test = ensemble.predict(X_test)
    acc = round(ensemble.score(x_cv, y_cv) * 100 , 2)
    score += acc
print(score/10)

ensemble.fit(X_all, Y_all)
Y_test = ensemble.predict(X_test).astype(int)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                           'Survived': Y_test})
submission.to_csv('submission.csv', index=False)