
# coding: utf-8

# ## Wczytanie potrzebnych bibliotek

import pandas as pd
import numpy as np
import os, re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display

import pymysql
db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
cursor = db.cursor()

cursor.execute("USE S72t1V5idn") # select the database





discks_names =  re.findall(r"[A-Z]+:.*$",os.popen("mountvol /").read(),re.MULTILINE)

a = []
for i in range(len(discks_names)):
    b = discks_names[i].split(":")[0]
    a.append(b)

dics = a['C' in a]
dics = a[2]

dics
league_abb = 'N1'


pd.options.display.max_columns = None
loc = dics + ":/data_football/final_datasets/"
dataset = pd.read_csv(loc + 'training_dataset_'+league_abb+'.csv', index_col = 0)

df_final = dataset
warnings.simplefilter('ignore')
df_final['FTR'] = np.where(df_final['FTR']=='H', 1, 0)
df_final['FTR']=(df_final['FTR']==1).astype(int)



df_final = df_final[['FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP',
       'ATP', 'HT_Wins', 'AT_Wins', 'HT_Loss', 'AT_Loss', 'HT_Draws',
       'AT_Draws', 'MW', 'HomeTeamLP', 'AwayTeamLP', 'Sezon', 'HT_LP', 'AT_LP',
       'HTFormPts', 'ATFormPts', 'HTWinStreak5', 'HTWinStreak3',
       'HTLossStreak3', 'HTLossStreak5', 'HTDrawStreak3',
       'HTDrawStreak5', 'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3',
       'ATLossStreak5', 'ATDrawStreak3', 'ATDrawStreak5', 'HTGD', 'ATGD',
       'DiffPts', 'DiffFormPts', 'DiffLP', 'H2H_Home_pts', 'H2H_Away_pts',
        'Mean_home_goals', 'Mean_away_goals',
       'Elo_HT', 'Elo_AT',  'Age_H', 'Foreign_H', 'Total_value_H',
       'Market_value_H',  'Age_A', 'Foreign_A', 'Total_value_A',
       'Market_value_A']]



df_final.HTFormPts = df_final.HTFormPts.fillna(df_final.HTFormPts.median())
df_final.ATFormPts = df_final.ATFormPts.fillna(df_final.ATFormPts.median())
df_final.DiffFormPts = df_final.DiffFormPts.fillna(df_final.DiffFormPts.median())
df_final.Mean_home_goals = df_final.Mean_home_goals.fillna(df_final.Mean_home_goals.median())
df_final.Mean_away_goals = df_final.Mean_away_goals.fillna(df_final.Mean_away_goals.median())

# ### Stworzenie nowych zmiennych

df_final["Goals_mean_diff"] = df_final["Mean_home_goals"] - df_final["Mean_away_goals"]
df_final["H2H_Diff"] = df_final["H2H_Home_pts"] - df_final["H2H_Away_pts"]
df_final["Total_Diff"] = df_final["Total_value_H"] / df_final["Total_value_A"]
df_final["Age_diff"] = df_final["Age_H"] - df_final["Age_A"]
df_final["LP_Diff"] = df_final["HT_LP"] - df_final["AT_LP"]
df_final["ELO_diff"] = df_final["Elo_HT"] - df_final["Elo_AT"]

#df_final.columns



cols_cat = ['HTWinStreak5', 'HTWinStreak3', 'HTLossStreak3',
       'HTLossStreak5', 'HTDrawStreak3', 'HTDrawStreak5', 'ATWinStreak3',
       'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5', 'ATDrawStreak3',
       'ATDrawStreak5']
for col in cols_cat:
    df_final[col] = df_final[col].astype('category')


# ### Wybranie wybranych kolumn w zbiorze danych


df_final = df_final[[ 'HTGD', 'ATGD',
          'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3',
       'HTLossStreak5', 'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3',
       'ATLossStreak5',  'DiffPts', 'DiffFormPts', 'DiffLP',
        'Mean_home_goals', 'Mean_away_goals',  'H2H_Diff', "ELO_diff", 
        'Total_Diff', 'LP_Diff', 'FTR', 'Goals_mean_diff']]

#df_final.tail()


df_final = df_final.drop('Mean_home_goals',axis=1)
df_final = df_final.dropna()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_final.drop('FTR',axis=1).reset_index(drop = True), 
           df_final['FTR'].reset_index().drop('Date',axis=1), test_size=0.30, 
            random_state=101)

# https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = X_train #independent columns
y = y_train  #target column i.e price range
#apply SelectKBest class to extract top 10 best features
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#feat_importances.nlargest(10).plot(kind='barh')
#plt.show()




veriables = list(feat_importances.nlargest(10).index)




aic_lr = [None] * 4

import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train[veriables])
result=logit_model.fit()
#print(result.summary2())
aic_lr[0] = result.summary2().tables[0][3][1]
#result.summary2().tables[1]['P>|z|'][result.summary2().tables[1]['P>|z|'] > 0.05].index.tolist()


cols_to_df = result.summary2().tables[1]['P>|z|'][result.summary2().tables[1]['P>|z|'] <= 0.05].index.tolist()
import statsmodels.api as sm
logit_model2=sm.Logit(y_train,X_train[cols_to_df])
result2=logit_model2.fit()
#print(result2.summary2())

#AIC
aic_lr[1] = result2.summary2().tables[0][3][1]


# ## Cross walidacja

X = df_final.loc[:, df_final.columns != 'FTR']
y = df_final.loc[:, df_final.columns == 'FTR']


cols_to_norm = ['HTGD', 'ATGD', 'DiffPts' , 'DiffFormPts', 'DiffLP', 'Mean_away_goals',
                    'H2H_Diff', 'ELO_diff', 'Total_Diff', 'LP_Diff', 'Goals_mean_diff']

from sklearn import preprocessing
X_new = X
min_max_scaler = preprocessing.MinMaxScaler()
scaled = pd.DataFrame(min_max_scaler.fit_transform(X_new[cols_to_norm]))
scaled.columns = cols_to_norm
X_new = X_new.reset_index(drop = True)
X_new[cols_to_norm] = scaled


from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_new.reset_index(drop = True), 
          y.reset_index(drop = True), test_size=0.30, 
            random_state=101)



import statsmodels.api as sm
logit_model3=sm.Logit(y_train2,X_train2[veriables].astype(float))
result3=logit_model3.fit()
#print(result3.summary2())
aic_lr[2] = result3.summary2().tables[0][3][1]

import statsmodels.api as sm
logit_model4=sm.Logit(y_train2,X_train2[result3.summary2().tables[1]['P>|z|'][result3.summary2().tables[1]['P>|z|'] <= 0.05].index.tolist()].astype(float))
result4=logit_model3.fit()
#print(result4.summary2())
aic_lr[3] = result4.summary2().tables[0][3][1]


# ## Weryfikacja modelu 

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train[veriables], y_train)

if aic_lr.index(max(aic_lr)) == 0:
    logreg_final = logit_model
elif aic_lr.index(max(aic_lr)) == 1:
    logreg_final = logit_model2
elif aic_lr.index(max(aic_lr)) == 2:
    logreg_final = logit_model3
elif aic_lr.index(max(aic_lr)) == 3:
    logreg_final = logit_model4



y_pred = logreg.predict(X_test[veriables])
lr_acc = logreg.score(X_test[veriables], y_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test[veriables], y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


#from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test[veriables]))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test[veriables])[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
#plt.show()


from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred))


# # Nowe dane

predict = pd.read_csv(loc +"predict_"+league_abb+".csv", index_col = 0)
predict_mw = list(predict["MW"])[1]
predict_date = predict.index
predict_date = pd.to_datetime(predict_date).strftime("%Y-%m-%d")


cols_sel= ['HomeTeam', 'AwayTeam', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP',
       'ATP', 'HT_Wins', 'AT_Wins', 'HT_Loss', 'AT_Loss', 'HT_Draws',
       'AT_Draws', 'MW', 'HomeTeamLP', 'AwayTeamLP', 'Sezon', 'HT_LP', 'AT_LP',
       'HTFormPts', 'ATFormPts', 'HTWinStreak5', 'HTWinStreak3',
       'HTLossStreak3', 'HTLossStreak5', 'HTDrawStreak3',
       'HTDrawStreak5', 'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3',
       'ATLossStreak5', 'ATDrawStreak3', 'ATDrawStreak5', 'HTGD', 'ATGD',
       'DiffPts', 'DiffFormPts', 'DiffLP', 'H2H_Home_pts', 'H2H_Away_pts',
        'Mean_home_goals', 'Mean_away_goals',
       'Elo_HT', 'Elo_AT',  'Age_H', 'Foreign_H', 'Total_value_H',
       'Market_value_H',  'Age_A', 'Foreign_A', 'Total_value_A',
       'Market_value_A']

predict = predict[cols_sel]

predict.HTFormPts = predict.HTFormPts.fillna(predict.HTFormPts.median())
predict.ATFormPts = predict.ATFormPts.fillna(predict.ATFormPts.median())
predict.DiffFormPts = predict.DiffFormPts.fillna(predict.DiffFormPts.median())

predict["H2H_Diff"] = predict["H2H_Home_pts"] - predict["H2H_Away_pts"]
predict["Total_Diff"] = predict["Total_value_H"] / predict["Total_value_A"]
predict["Age_diff"] = predict["Age_H"] - predict["Age_A"]
predict["LP_Diff"] = predict["HT_LP"] - predict["AT_LP"]
predict["ELO_diff"] = predict["Elo_HT"] - predict["Elo_AT"]

predict = predict[['HomeTeam', 'AwayTeam', 'HTGD', 'ATGD',
          'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3',
       'HTLossStreak5', 'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3',
       'ATLossStreak5',  'DiffPts', 'DiffFormPts', 'DiffLP',
        'Mean_home_goals', 'Mean_away_goals',  'H2H_Diff', "ELO_diff", 
        'Total_Diff', 'LP_Diff', 'FTR']]

predict["Goals_mean_diff"] = predict["Mean_home_goals"] - predict["Mean_away_goals"]

predict = predict.drop('Mean_home_goals',axis=1)


#predict.tail()


predict = predict.reset_index().drop('Date', axis = 1)


y_pred_new = logreg.predict(predict[veriables])


#y_pred_new


final_logreg = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_logreg
final_logreg_to_db = pd.concat([pd.DataFrame(predict_date),predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_logreg_to_db["MW"] = predict_mw
final_logreg_to_db["Result"] = ""
final_logreg_to_db["model"] = "LR"
final_logreg_to_db["league"] = league_abb
final_logreg_to_db.columns = ["Date", "HomeTeam", "AwayTeam", "prediction", "MW", "Result", "model" , "league"]

#mySql_insert_query = """INSERT INTO predictons (date, league, HomeTeam, AwayTeam, model, prediction, Result) 
#                                VALUES (%s, %s, %s, %s) """


for i in range(0, len(final_logreg_to_db)):
    
    old_df = pd.read_sql("SELECT * FROM predictions WHERE MW='"+ str(predict_mw) +"' and model= '" + "LR" + "' and league = '"+ league_abb + "'", db)
    
    tuple_temp = (str(final_logreg_to_db["prediction"][i]),
              str(predict_mw),
              league_abb,
              final_logreg_to_db["HomeTeam"][i],
              final_logreg_to_db["AwayTeam"][i])
    
    if len(old_df[(old_df["HomeTeam"] == final_logreg_to_db["HomeTeam"][i]) & (old_df["AwayTeam"] == final_logreg_to_db["AwayTeam"][i]) ]) >0:
        cursor.execute("UPDATE predictions SET prediction = %s WHERE  MW=%s and model= 'LR' and league = %s  AND HomeTeam = %s and AwayTeam = %s", tuple_temp)
        db.commit()
    else:
        recordTuple = (final_logreg_to_db["Date"][i],
               final_logreg_to_db["league"][i], 
               float(final_logreg_to_db["MW"][i]),
               final_logreg_to_db["HomeTeam"][i], 
               final_logreg_to_db["AwayTeam"][i],
               final_logreg_to_db["model"][i],
               str(final_logreg_to_db["prediction"][i]),
               final_logreg_to_db["Result"][i]) 
        cursor.execute("INSERT INTO predictions (date, league,MW,  HomeTeam, AwayTeam, model, prediction, Result) VALUES (%s, %s,%s,  %s, %s, %s, %s, %s) ", recordTuple)
        db.commit() 
    



# # Decision Trees



# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation



# Create Decision Tree classifer object
clf =  DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(X_train[veriables], y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test[veriables])



# Model Accuracy, how often is the classifier correct?
dt_acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# ### Przeskalowane


# Create Decision Tree classifer object
clf_dt_scaled =  DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf_dt_scaled = clf_dt_scaled.fit(X_train2[veriables], y_train2)

#Predict the response for test dataset
y_pred = clf_dt_scaled.predict(X_test2[veriables])
# Model Accuracy, how often is the classifier correct?
dt_acc_scaled = metrics.accuracy_score(y_test2, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))




y_pred_new = clf_dt_scaled.predict(predict[veriables])
print(classification_report(y_test2, y_pred))



final_DT = pd.concat([predict[[ "HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_DT
final_DT_to_db = pd.concat([pd.DataFrame(predict_date),predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_DT_to_db["MW"] = predict_mw
final_DT_to_db["Result"] = ""
final_DT_to_db["model"] = "DT"
final_DT_to_db["league"] = league_abb
final_DT_to_db.columns = ["Date", "HomeTeam", "AwayTeam", "prediction", "MW", "Result", "model",  "league" ]

db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
cursor = db.cursor()

cursor.execute("USE S72t1V5idn") # select the database


for i in range(0, len(final_DT_to_db)):
    
    old_df = pd.read_sql("SELECT * FROM predictions WHERE MW='"+ str(predict_mw) +"' and model= '" + "DT" + "' and league = '"+ league_abb + "'", db)
    
    tuple_temp = (str(final_DT_to_db["prediction"][i]),
              str(predict_mw),
              league_abb,
              final_DT_to_db["HomeTeam"][i],
              final_DT_to_db["AwayTeam"][i])
    
    if len(old_df[(old_df["HomeTeam"] == final_DT_to_db["HomeTeam"][i]) & (old_df["AwayTeam"] == final_DT_to_db["AwayTeam"][i]) ]) >0:
        cursor.execute("UPDATE predictions SET prediction = %s WHERE  MW=%s and model= 'DT' and league = %s  AND HomeTeam = %s and AwayTeam = %s", tuple_temp)
        db.commit()
    else:
        recordTuple = (final_DT_to_db["Date"][i],
               final_DT_to_db["league"][i], 
               float(final_DT_to_db["MW"][i]),
               final_DT_to_db["HomeTeam"][i], 
               final_DT_to_db["AwayTeam"][i],
               final_DT_to_db["model"][i],
               str(final_DT_to_db["prediction"][i]),
               final_DT_to_db["Result"][i]) 
        cursor.execute("INSERT INTO predictions (date, league,MW,  HomeTeam, AwayTeam, model, prediction, Result) VALUES (%s, %s,%s,  %s, %s, %s, %s, %s) ", recordTuple)
        db.commit() 
    


#  Random Forest
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf_rf=RandomForestClassifier(n_estimators=250)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf_rf.fit(X_train,y_train)

y_pred_lr=clf_rf.predict(X_test)
y_pred = clf_rf.predict(predict[list(X_train.columns)])



from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
rf_acc = metrics.accuracy_score(y_test, y_pred_lr)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_lr))


# ### Przeskalowane



# Create Decision Tree classifer object
clf_rf_scaled =  RandomForestClassifier(n_estimators=250)

# Train Decision Tree Classifer
clf_rf_scaled = clf_rf_scaled.fit(X_train2[veriables], y_train2)

#Predict the response for test dataset
y_pred = clf_rf_scaled.predict(X_test2[veriables])
# Model Accuracy, how often is the classifier correct?
rf_acc_scaled = metrics.accuracy_score(y_test2, y_pred)
print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))


y_pred_new = clf_rf.predict(predict[list(X_train.columns)])




final_RF = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_RF_to_db = pd.concat([pd.DataFrame(predict_date),predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_RF_to_db["MW"] = predict_mw
final_RF_to_db["Result"] = ""
final_RF_to_db["model"] = "RF"
final_RF_to_db["league"] = league_abb

final_RF_to_db.columns = ["Date", "HomeTeam", "AwayTeam", "prediction", "MW", "Result", "model",  "league" ]

db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
cursor = db.cursor()

cursor.execute("USE S72t1V5idn") # select the database


for i in range(0, len(final_RF_to_db)):
    
    old_df = pd.read_sql("SELECT * FROM predictions WHERE MW='"+ str(predict_mw) +"' and model= '" + "RF" + "' and league = '"+ league_abb + "'", db)
    
    tuple_temp = (str(final_RF_to_db["prediction"][i]),
              str(predict_mw),
              league_abb,
              final_RF_to_db["HomeTeam"][i],
              final_RF_to_db["AwayTeam"][i])
    
    if len(old_df[(old_df["HomeTeam"] == final_RF_to_db["HomeTeam"][i]) & (old_df["AwayTeam"] == final_RF_to_db["AwayTeam"][i]) ]) >0:
        cursor.execute("UPDATE predictions SET prediction = %s WHERE  MW=%s and model= 'DT' and league = %s  AND HomeTeam = %s and AwayTeam = %s", tuple_temp)
        db.commit()
    else:
        recordTuple = (final_RF_to_db["Date"][i],
               final_RF_to_db["league"][i], 
               float(final_RF_to_db["MW"][i]),
               final_RF_to_db["HomeTeam"][i], 
               final_RF_to_db["AwayTeam"][i],
               final_RF_to_db["model"][i],
               str(final_RF_to_db["prediction"][i]),
               final_RF_to_db["Result"][i]) 
        cursor.execute("INSERT INTO predictions (date, league,MW,  HomeTeam, AwayTeam, model, prediction, Result) VALUES (%s, %s,%s,  %s, %s, %s, %s, %s) ", recordTuple)
        db.commit() 





# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into features and labels
X=df_final[['DiffLP', 'Total_Diff','ELO_diff', 'Goals_mean_diff', 'ATGD', 'Mean_away_goals', 
            'HTGD', 'DiffPts', 'LP_Diff', 'DiffFormPts']]  
y=df_final['FTR']                                       
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=5) #


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf_rf2=RandomForestClassifier(n_estimators=200)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf_rf2.fit(X_train,y_train)

# prediction on test set
y_pred_rf=clf_rf2.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rf))


y_pred_new = clf_rf2.predict(predict[list(X_train.columns)])
final_RF2 = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_RF2 = pd.concat([pd.DataFrame(predict_date),predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_RF2["MW"] = predict_mw
final_RF2["Result"] = ""
final_RF2["model"] = "RFs"
final_RF2["league"] = league_abb

final_RF2.columns = ["Date", "HomeTeam", "AwayTeam", "prediction", "MW", "Result", "model",  "league" ]

db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
cursor = db.cursor()

cursor.execute("USE S72t1V5idn") # select the database


for i in range(0, len(final_RF2)):
    
    old_df = pd.read_sql("SELECT * FROM predictions WHERE MW='"+ str(predict_mw) +"' and model= '" + "RFs" + "' and league = '"+ league_abb + "'", db)
    
    tuple_temp = (str(final_RF2["prediction"][i]),
              str(predict_mw),
              league_abb,
              final_RF2["HomeTeam"][i],
              final_RF2["AwayTeam"][i])
    
    if len(old_df[(old_df["HomeTeam"] == final_RF2["HomeTeam"][i]) & (old_df["AwayTeam"] == final_RF2["AwayTeam"][i]) ]) >0:
        cursor.execute("UPDATE predictions SET prediction = %s WHERE  MW=%s and model= 'RFs' and league = %s  AND HomeTeam = %s and AwayTeam = %s", tuple_temp)
        db.commit()
    else:
        recordTuple = (final_RF2["Date"][i],
               final_RF2["league"][i], 
               float(final_RF2["MW"][i]),
               final_RF2["HomeTeam"][i], 
               final_RF2["AwayTeam"][i],
               final_RF2["model"][i],
               str(final_RF2["prediction"][i]),
               final_RF2["Result"][i]) 
        cursor.execute("INSERT INTO predictions (date, league,MW,  HomeTeam, AwayTeam, model, prediction, Result) VALUES (%s, %s,%s,  %s, %s, %s, %s, %s) ", recordTuple)
        db.commit() 


# # SVM
from sklearn import svm
clf_svc = svm.SVC(gamma='scale')
clf_svc.fit(X_train, y_train)  

# prediction on test set
y_pred_svc=clf_svc.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
svc_acc = metrics.accuracy_score(y_test, y_pred_svc)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svc))


# ### Przeskalowane


# Create Decision Tree classifer object
clf_svc_scaled =  svm.SVC(gamma='scale')

# Train Decision Tree Classifer
clf_svc_scaled = clf_svc_scaled.fit(X_train2[veriables], y_train2)

#Predict the response for test dataset
y_pred = clf_svc_scaled.predict(X_test2[veriables])
# Model Accuracy, how often is the classifier correct?
svc_acc_scaled = metrics.accuracy_score(y_test2, y_pred)
print("Accuracy:",svc_acc_scaled)



y_pred_new = clf_svc_scaled.predict(predict[list(X_train.columns)])



from sklearn import svm
clf_svc2 = svm.SVC(kernel='linear')
clf_svc2.fit(X_train, y_train)  

# prediction on test set
y_pred_svc2=clf_svc2.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
svc_acc =metrics.accuracy_score(y_test, y_pred_svc2)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svc2))




y_pred_new = clf_svc2.predict(predict[list(X_train.columns)])
final_SVM2 = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_SVM = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_SVC = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_SVC = pd.concat([pd.DataFrame(predict_date),predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_SVC["MW"] = predict_mw
final_SVC["Result"] = ""
final_SVC["model"] = "SVC"
final_SVC["league"] = league_abb

final_SVC.columns = ["Date", "HomeTeam", "AwayTeam", "prediction", "MW", "Result", "model",  "league" ]

db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
cursor = db.cursor()

cursor.execute("USE S72t1V5idn") # select the database


for i in range(0, len(final_SVC)):
    
    old_df = pd.read_sql("SELECT * FROM predictions WHERE MW='"+ str(predict_mw) +"' and model= '" + "SVC" + "' and league = '"+ league_abb + "'", db)
    
    tuple_temp = (str(final_SVC["prediction"][i]),
              str(predict_mw),
              league_abb,
              final_SVC["HomeTeam"][i],
              final_SVC["AwayTeam"][i])
    
    if len(old_df[(old_df["HomeTeam"] == final_SVC["HomeTeam"][i]) & (old_df["AwayTeam"] == final_SVC["AwayTeam"][i]) ]) >0:
        cursor.execute("UPDATE predictions SET prediction = %s WHERE  MW=%s and model= 'SVC' and league = %s  AND HomeTeam = %s and AwayTeam = %s", tuple_temp)
        db.commit()
    else:
        recordTuple = (final_SVC["Date"][i],
               final_SVC["league"][i], 
               float(final_SVC["MW"][i]),
               final_SVC["HomeTeam"][i], 
               final_SVC["AwayTeam"][i],
               final_SVC["model"][i],
               str(final_SVC["prediction"][i]),
               final_SVC["Result"][i]) 
        cursor.execute("INSERT INTO predictions (date, league,MW,  HomeTeam, AwayTeam, model, prediction, Result) VALUES (%s, %s,%s,  %s, %s, %s, %s, %s) ", recordTuple)
        db.commit() 



# # Stochastic Gradient Descent (SGD)

# Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to discriminative learning of linear classifiers under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Even though SGD has been around in the machine learning community for a long time, it has received a considerable amount of attention just recently in the context of large-scale learning.
# 
# SGD has been successfully applied to large-scale and sparse machine learning problems often encountered in text classification and natural language processing. Given that the data is sparse, the classifiers in this module easily scale to problems with more than 10^5 training examples and more than 10^5 features.
# 
# The advantages of Stochastic Gradient Descent are:
# 
# Efficiency.
# Ease of implementation (lots of opportunities for code tuning).
# The disadvantages of Stochastic Gradient Descent include:
# 
# SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.
# SGD is sensitive to feature scaling.



from sklearn.linear_model import SGDClassifier
clf_sgd = SGDClassifier(loss="log", penalty="l2", max_iter=15)
clf_sgd.fit(X_train, y_train)


# prediction on test set
y_pred_sgd=clf_sgd.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_sgd))


# ### Przeskalowanie



# Create Decision Tree classifer object
clf_sgd_scaled =  SGDClassifier(loss="log", penalty="l2", max_iter=15)

# Train Decision Tree Classifer
clf_sgd_scaled = clf_sgd_scaled.fit(X_train2[veriables], y_train2)

#Predict the response for test dataset
y_pred = clf_sgd_scaled.predict(X_test2[veriables])
# Model Accuracy, how often is the classifier correct?
sgd_acc_scaled = metrics.accuracy_score(y_test2, y_pred)
print("Accuracy:",sgd_acc_scaled)




y_pred_new = clf.predict(predict[list(X_train.columns)])
final_SGD = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_SGD_to_db = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_SGD_to_db = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_SGD_to_db = pd.concat([pd.DataFrame(predict_date),predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_SGD_to_db["MW"] = predict_mw
final_SGD_to_db["Result"] = ""
final_SGD_to_db["model"] = "SGD"
final_SGD_to_db["league"] = league_abb

final_SGD_to_db.columns = ["Date", "HomeTeam", "AwayTeam", "prediction", "MW", "Result", "model",  "league" ]

db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
cursor = db.cursor()

cursor.execute("USE S72t1V5idn") # select the database


for i in range(0, len(final_SGD_to_db)):
    
    old_df = pd.read_sql("SELECT * FROM predictions WHERE MW='"+ str(predict_mw) +"' and model= '" + "SGD" + "' and league = '"+ league_abb + "'", db)
    
    tuple_temp = (str(final_SGD_to_db["prediction"][i]),
              str(predict_mw),
              league_abb,
              final_SGD_to_db["HomeTeam"][i],
              final_SGD_to_db["AwayTeam"][i])
    
    if len(old_df[(old_df["HomeTeam"] == final_SGD_to_db["HomeTeam"][i]) & (old_df["AwayTeam"] == final_SGD_to_db["AwayTeam"][i]) ]) >0:
        cursor.execute("UPDATE predictions SET prediction = %s WHERE  MW=%s and model= 'SGD' and league = %s  AND HomeTeam = %s and AwayTeam = %s", tuple_temp)
        db.commit()
    else:
        recordTuple = (final_SGD_to_db["Date"][i],
               final_SGD_to_db["league"][i], 
               float(final_SGD_to_db["MW"][i]),
               final_SGD_to_db["HomeTeam"][i], 
               final_SGD_to_db["AwayTeam"][i],
               final_SGD_to_db["model"][i],
               str(final_SGD_to_db["prediction"][i]),
               final_SGD_to_db["Result"][i]) 
        cursor.execute("INSERT INTO predictions (date, league,MW,  HomeTeam, AwayTeam, model, prediction, Result) VALUES (%s, %s,%s,  %s, %s, %s, %s, %s) ", recordTuple)
        db.commit() 


from sklearn.neighbors import KNeighborsClassifier



for i in range(3,20):
    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train) 
    print(knn.score(X_test, y_test))   


knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train, y_train) 
#print(knn.score(X_test, y_test))   




y_pred_new = knn.predict(predict[list(X_train.columns)])
final_KNN = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_KNN_to_db = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_KNN_to_db = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_KNN_to_db = pd.concat([pd.DataFrame(predict_date),predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
final_KNN_to_db["MW"] = predict_mw
final_KNN_to_db["Result"] = ""
final_KNN_to_db["model"] = "KNN"
final_KNN_to_db["league"] = league_abb

final_KNN_to_db.columns = ["Date", "HomeTeam", "AwayTeam", "prediction", "MW", "Result", "model",  "league" ]

db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
cursor = db.cursor()

cursor.execute("USE S72t1V5idn") # select the database


for i in range(0, len(final_KNN_to_db)):
    
    old_df = pd.read_sql("SELECT * FROM predictions WHERE MW='"+ str(predict_mw) +"' and model= '" + "KNN" + "' and league = '"+ league_abb + "'", db)
    
    tuple_temp = (str(final_KNN_to_db["prediction"][i]),
              str(predict_mw),
              league_abb,
              final_KNN_to_db["HomeTeam"][i],
              final_KNN_to_db["AwayTeam"][i])
    
    if len(old_df[(old_df["HomeTeam"] == final_KNN_to_db["HomeTeam"][i]) & (old_df["AwayTeam"] == final_KNN_to_db["AwayTeam"][i]) ]) >0:
        cursor.execute("UPDATE predictions SET prediction = %s WHERE  MW=%s and model= 'KNN' and league = %s  AND HomeTeam = %s and AwayTeam = %s", tuple_temp)
        db.commit()
    else:
        recordTuple = (final_KNN_to_db["Date"][i],
               final_KNN_to_db["league"][i], 
               float(final_KNN_to_db["MW"][i]),
               final_KNN_to_db["HomeTeam"][i], 
               final_KNN_to_db["AwayTeam"][i],
               final_KNN_to_db["model"][i],
               str(final_KNN_to_db["prediction"][i]),
               final_KNN_to_db["Result"][i]) 
        cursor.execute("INSERT INTO predictions (date, league,MW,  HomeTeam, AwayTeam, model, prediction, Result) VALUES (%s, %s,%s,  %s, %s, %s, %s, %s) ", recordTuple)
        db.commit() 





final = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(final_logreg, final_DT,  how='inner', left_on=['HomeTeam', 'AwayTeam'], 
                           right_on = ['HomeTeam', 'AwayTeam']), 
         final_RF,  how='inner', left_on=['HomeTeam', 'AwayTeam'], 
                           right_on = ['HomeTeam', 'AwayTeam']),
          final_SVM2,  how='inner', left_on=['HomeTeam', 'AwayTeam'], 
                           right_on = ['HomeTeam', 'AwayTeam']),
             final_SGD,  how='inner', left_on=['HomeTeam', 'AwayTeam'], 
                           right_on = ['HomeTeam', 'AwayTeam']),
             final_KNN,  how='inner', left_on=['HomeTeam', 'AwayTeam'], 
                           right_on = ['HomeTeam', 'AwayTeam'])

final.columns = [['HomeTeam', 'AwayTeam', 'LR', 'DT', 'RF', 'SVM', 'SGD', 'KNN']]
final



from sklearn.ensemble import VotingClassifier


# https://www.kaggle.com/den3b81/better-predictions-stacking-with-votingclassifier

# https://scikit-learn.org/stable/glossary.html#term-n-jobs




eclf  = VotingClassifier(estimators=[('lr', logreg),
                                     ('dt', clf_dt_scaled) ,
                                     ('rf', clf_rf), 
                                     ('svc', clf_svc_scaled),
                                     ('sgd', clf_sgd)
                                     ], voting='hard',
weights = [lr_acc,
           dt_acc_scaled, 
           rf_acc_scaled, 
           svc_acc_scaled,
           sgd_acc_scaled])





eclf.fit(X_train, y_train) 
print(eclf.score(X_test, y_test))   






y_val_pred = eclf.predict(predict[list(X_train.columns)])



final = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_val_pred)], axis=1)
#final_ECL_to_db = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_val_pred)], axis=1)
final_ECL_to_db = pd.concat([pd.DataFrame(predict_date),predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_val_pred)], axis=1)
final_ECL_to_db["MW"] = predict_mw
final_ECL_to_db["Result"] = ""
final_ECL_to_db["model"] = "ECL"
final_ECL_to_db["league"] = league_abb

final_ECL_to_db.columns = ["Date", "HomeTeam", "AwayTeam", "prediction", "MW", "Result", "model",  "league" ]

db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
cursor = db.cursor()

cursor.execute("USE S72t1V5idn") # select the database


for i in range(0, len(final_ECL_to_db)):
    
    old_df = pd.read_sql("SELECT * FROM predictions WHERE MW='"+ str(predict_mw) +"' and model= '" + "ECL" + "' and league = '"+ league_abb + "'", db)
    
    tuple_temp = (str(final_ECL_to_db["prediction"][i]),
              str(predict_mw),
              league_abb,
              final_ECL_to_db["HomeTeam"][i],
              final_ECL_to_db["AwayTeam"][i])
    
    if len(old_df[(old_df["HomeTeam"] == final_ECL_to_db["HomeTeam"][i]) & (old_df["AwayTeam"] == final_ECL_to_db["AwayTeam"][i]) ]) >0:
        cursor.execute("UPDATE predictions SET prediction = %s WHERE  MW=%s and model= 'ECL' and league = %s  AND HomeTeam = %s and AwayTeam = %s", tuple_temp)
        db.commit()
    else:
        recordTuple = (final_ECL_to_db["Date"][i],
               final_ECL_to_db["league"][i], 
               float(final_ECL_to_db["MW"][i]),
               final_ECL_to_db["HomeTeam"][i], 
               final_ECL_to_db["AwayTeam"][i],
               final_ECL_to_db["model"][i],
               str(final_ECL_to_db["prediction"][i]),
               final_ECL_to_db["Result"][i]) 
        cursor.execute("INSERT INTO predictions (date, league,MW,  HomeTeam, AwayTeam, model, prediction, Result) VALUES (%s, %s,%s,  %s, %s, %s, %s, %s) ", recordTuple)
        db.commit() 


final

