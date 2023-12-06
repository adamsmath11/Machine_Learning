# Import libraries and methods/functions
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


# Start your code here!
t_dem = pd.read_csv('telecom_demographics.csv')
t_use = pd.read_csv('telecom_usage.csv')
churn_df = t_dem.merge(t_use, how="left", left_on='customer_id', right_on='customer_id')

churn_rate = churn_df['churn'].value_counts()/len(churn_df)
print(churn_rate)
print(churn_df.info())

churn_df = pd.get_dummies(churn_df, \
                    columns=['telecom_partner','gender',\
                    'state','city','registration_event'])

scaler = StandardScaler()
features = churn_df.drop(['customer_id','churn'], axis=1)
features_scaled = scaler.fit_transform(features)
target = churn_df['churn']


train_X, test_X, train_Y, test_Y = train_test_split(features_scaled,target, \
                                                test_size=.2, random_state=42)

logreg = LogisticRegression(random_state=42)
logreg.fit(train_X, train_Y)
logreg_pred = logreg.predict(test_X)


rf=RandomForestClassifier(random_state=42)
rf.fit(train_X,train_Y)
rf_pred = rf.predict(test_X)



print(confusion_matrix(test_Y,logreg_pred))
print(classification_report(test_Y, logreg_pred))

print(confusion_matrix(test_Y, rf_pred))
print(classification_report(test_Y, rf_pred))

higher_accuracy='RandomForest'