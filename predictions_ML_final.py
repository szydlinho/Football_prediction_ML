# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 17:22:51 2019

@author: base
"""




league_abb = 'D1'

def MLUpdate1_X2(league_abb):
    
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns

    from IPython.display import display
    import time
    import pymysql
    db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
    cursor = db.cursor()
    
    cursor.execute("USE S72t1V5idn") # select the database
    
    dics = 'G'
    loc_to_functions = dics + ":/data_football/Football_prediction_ML/functions/"
        
    os.chdir(loc_to_functions)
        
    import functions_for_dataset_preparation as f1
    loc = dics + ":/data_football/final_datasets/"
    dataset = pd.read_csv(loc + 'training_dataset_'+league_abb+'.csv', index_col = 0)
    #dataset = pd.read_csv('C:/Users/szydlikp/DT_opt/training_dataset_E0.csv', index_col = 0)
    df_final = dataset
    df_final['FTR'] = np.where(df_final['FTR']=='H', 1, 0)
    df_final['FTR']=(df_final['FTR']==1).astype(int)
    
    df_final = df_final.loc[~(df_final["MW"] == 1)]
    
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
    
    
    df_final = df_final.drop('Mean_away_goals',axis=1)
    df_final = df_final.dropna()
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_final.drop('FTR',axis=1).reset_index(drop = True), 
               df_final['FTR'].reset_index().drop('Date',axis=1), test_size=0.30, 
                random_state=101)
    
    # https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X = X_train #independent columns
    y = y_train  #target column i.e price range
    #apply SelectKBest class to extract top 10 best features
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier()
    model.fit(X,y.values.ravel())
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    #feat_importances.nlargest(10).plot(kind='barh')
    #plt.show()
    veriables = list(feat_importances.nlargest(10).index)
    
    
    aic_lr = [None] * 4
    
    #
    #logit_model = LogisticRegression(solver='lbfgs', max_iter = 300).fit(y = y_train.values.ravel(), X = X_train[veriables])
    #logit_model.score(X_test[veriables], y_test)

    import statsmodels.api as sm
    logit_model=sm.Logit(y_train,X_train[veriables])
    result=logit_model.fit(disp=0)
    #print(result.summary2())
    aic_lr[0] = result.summary2().tables[0][3][1]
    #result.summary2().tables[1]['P>|z|'][result.summary2().tables[1]['P>|z|'] > 0.05].index.tolist()
    
    
    cols_to_df = result.summary2().tables[1]['P>|z|'][result.summary2().tables[1]['P>|z|'] <= 0.05].index.tolist()
    import statsmodels.api as sm
    logit_model2=sm.Logit(y_train,X_train[cols_to_df])
    result2=logit_model2.fit(disp=0)
    #print(result2.summary2())
    
    #AIC
    aic_lr[1] = result2.summary2().tables[0][3][1]
    
    
    
    X = df_final.loc[:, df_final.columns != 'FTR']
    y = df_final.loc[:, df_final.columns == 'FTR']
    
    
    cols_to_norm = ['HTGD', 'ATGD', 'DiffPts' , 'DiffFormPts', 'DiffLP', 'Mean_home_goals',
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
    result3=logit_model3.fit(disp=0)
    #print(result3.summary2())
    aic_lr[2] = result3.summary2().tables[0][3][1]
    
    import statsmodels.api as sm
    logit_model4=sm.Logit(y_train2,X_train2[result3.summary2().tables[1]['P>|z|'][result3.summary2().tables[1]['P>|z|'] <= 0.05].index.tolist()].astype(float))
    result4=logit_model4.fit(disp=0)
    #print(result4.summary2())
    aic_lr[3] = result4.summary2().tables[0][3][1]
    
    
    # ## Weryfikacja modelu 
    
    from sklearn.linear_model import LogisticRegression
    
    if aic_lr.index(max(aic_lr)) == 0:
        logreg_final = LogisticRegression(solver='lbfgs', max_iter = 300).fit(y = y_train.values.ravel(), X = X_train[veriables])
    elif aic_lr.index(max(aic_lr)) == 1:
        logreg_final = LogisticRegression(solver='lbfgs', max_iter = 300).fit(y = y_train.values.ravel(), X = X_train[veriables])
    elif aic_lr.index(max(aic_lr)) == 2:
        logreg_final = LogisticRegression(solver='lbfgs', max_iter = 300).fit(y = y_train2.values.ravel(), X = X_train2[veriables].astype(float))
    elif aic_lr.index(max(aic_lr)) == 3:
        logreg_final = LogisticRegression(solver='lbfgs', max_iter = 300).fit(y = y_train2.values.ravel(), X = X_train2[result3.summary2().tables[1]['P>|z|'][result3.summary2().tables[1]['P>|z|'] <= 0.05].index.tolist()].astype(float))
    
    
    
    y_pred = logreg_final.predict(X_test[veriables])
    lr_acc = logreg_final.score(X_test[veriables], y_test)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg_final.score(X_test[veriables], y_test)))
    
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    #print(confusion_matrix)
    

    
    time.sleep(10)
    # # Nowe dane
    #predict = pd.read_csv('C:/Users/szydlikp/DT_opt/predict_E0.csv', index_col = 0)
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
    
    predict = predict.drop('Mean_away_goals',axis=1)
    
    
    #predict.tail()
    
    
    predict = predict.reset_index().drop('Date', axis = 1)
    
    
    y_pred_new = logreg_final.predict(predict[veriables])
    y_pred_new_proba =  pd.DataFrame(logreg_final.predict_proba(predict[veriables]))
    
    #y_pred_new
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'LR', league_abb)
    
    print("LR wrzucone")
    
    final_logreg = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_logreg_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
    
    time.sleep(10)

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
    #dt_acc = metrics.accuracy_score(y_test, y_pred)
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    
    # ### Przeskalowane
    
    
    # Create Decision Tree classifer object
    clf_dt_scaled =  DecisionTreeClassifier(criterion="entropy", max_depth=5)
    
    # Train Decision Tree Classifer
    clf_dt_scaled = clf_dt_scaled.fit(X_train2[veriables], y_train2)
    
    #Predict the response for test dataset
    y_pred = clf_dt_scaled.predict(X_test2[veriables])
    # Model Accuracy, how often is the classifier correct?
    dt_acc_scaled = metrics.accuracy_score(y_test2, y_pred)

    
    
    if metrics.accuracy_score(y_test, y_pred) > metrics.accuracy_score(y_test2, y_pred):
        y_pred_new = clf.predict(predict[veriables])
        y_pred_new_proba =  pd.DataFrame(clf.predict_proba(predict[veriables]))
        print("Accuracy DT:",metrics.accuracy_score(y_test, y_pred))
    else:
        y_pred_new = clf_dt_scaled.predict(predict[veriables])
        y_pred_new_proba =  pd.DataFrame(clf_dt_scaled.predict_proba(predict[veriables]))    
        print("Accuracy DT:", metrics.accuracy_score(y_test2, y_pred))
    
    
    #print(classification_report(y_test2, y_pred))
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'DT', league_abb)
    
    print("DT wrzucone")
    #final_DT = pd.concat([predict[[ "HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_DT = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_DT_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
    
    time.sleep(10)

    
    #  Random Forest
    #Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    
    #Create a Gaussian Classifier
    clf_rf=RandomForestClassifier(n_estimators=250)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf_rf.fit(X_train,    y_train.values.ravel())

    y_pred_lr = clf_rf.predict(X_test)
    #y_pred = clf_rf.predict(predict[list(X_train.columns)])

    
    # Create Decision Tree classifer object
    clf_rf_scaled =  RandomForestClassifier(n_estimators=250)
    
    # Train Decision Tree Classifer
    clf_rf_scaled = clf_rf_scaled.fit(X_train2[veriables], y_train2.values.ravel())
    
    #Predict the response for test dataset
    y_pred = clf_rf_scaled.predict(X_test2[veriables])
    # Model Accuracy, how often is the classifier correct?
    rf_acc_scaled = metrics.accuracy_score(y_test2, y_pred)
    #print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
    
    #Create a Gaussian Classifier
    clf_rf2=RandomForestClassifier(n_estimators=200)

    clf_rf2.fit(X_train2,y_train2.values.ravel())
    
    # prediction on test set
    y_pred_rf=clf_rf2.predict(X_test2)
        
    y_pred_new = clf_rf2.predict(predict[list(X_train2.columns)])
    y_pred_new_proba =  pd.DataFrame(clf_rf2.predict_proba(predict[list(X_train2.columns)]))
    
    
    
    if metrics.accuracy_score(y_test, y_pred_lr) > metrics.accuracy_score(y_test2, y_pred):
        rf_clf = clf_rf
        y_pred_new = clf_rf.predict(predict[list(X_train.columns)])
        y_pred_new_proba =  pd.DataFrame(clf_rf.predict_proba(predict[list(X_train.columns)]))
        print("Accuracy RF:", metrics.accuracy_score(y_test, y_pred_lr))
        rfs_metric = metrics.accuracy_score(y_test, y_pred_lr)
        
    elif metrics.accuracy_score(y_test2, y_pred) > metrics.accuracy_score(y_test2, y_pred_rf):
        rf_clf = clf_rf_scaled
        y_pred_new = clf_rf_scaled.predict(predict[veriables])
        y_pred_new_proba =  pd.DataFrame(clf_rf_scaled.predict_proba(predict[veriables]))    
        print("Accuracy RF:",metrics.accuracy_score(y_test2, y_pred))
        rfs_metric = metrics.accuracy_score(y_test2, y_pred)
    else:
        rf_clf = clf_rf2
        y_pred_new = clf_rf2.predict(predict[list(X_train2.columns)])
        y_pred_new_proba =  pd.DataFrame(clf_rf2.predict_proba(predict[list(X_train2.columns)]))    
        print("Accuracy RF:", metrics.accuracy_score(y_test2, y_pred_rf))       
        rfs_metric = metrics.accuracy_score(y_test2, y_pred_rf)
    
    
    
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'RF', league_abb)
    print("RF wrzucone")
    
    
    #final_RF = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_RF = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_RF_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
        
    time.sleep(5)

 
    
#    from sklearn.ensemble import RandomForestClassifier
#        
#    print("Accuracy RFs:",metrics.accuracy_score(y_test2, y_pred_rf))
#    
#    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'RFs', league_abb)
#    print("RFs wrzucone")
#    #
#    #final_RF2 = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
#    final_RF2 = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
#                              pd.DataFrame(y_pred_new)], axis=1)
#     #                         y_pred_new_proba.iloc[:,1]], axis=1)
#    final_RF2_to_db = pd.concat([pd.DataFrame(predict_date),
#                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
#                                    pd.DataFrame(y_pred_new),
#                                    y_pred_new_proba.iloc[:,1]], axis=1)
    
#    time.sleep(5)

    
    # # SVM
    from sklearn import svm
    clf_svc_scale = svm.SVC(gamma='scale', probability=True)
    clf_svc_scale.fit(X_train, y_train)  
    
    # prediction on test set
    y_pred_svc=clf_svc_scale.predict(X_test)
    
    svc_acc = metrics.accuracy_score(y_test, y_pred_svc)
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svc))
    
    
    # ### Przeskalowane
    
    clf_svc2 = svm.SVC(kernel='linear', probability=True)
    clf_svc2.fit(X_train2, y_train2.values.ravel())  
    
    # prediction on test set
    y_pred_svc2=clf_svc2.predict(X_test2)
    
    svc_acc2 = metrics.accuracy_score(y_test2, y_pred_svc2)
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svc2))
    
    if svc_acc > svc_acc2:
        y_pred_new = clf_svc_scale.predict(predict[list(X_train.columns)])
        clf_svc_proba = svm.SVC(kernel='linear', probability =  True)
        clf_svc_proba.fit(X_train, y_train)  
        y_pred_new_proba =  pd.DataFrame(clf_svc_proba.predict_proba(predict[veriables]))
        print("SVC accuracy:", svc_acc)
        
    else:
        y_pred_new = clf_svc2.predict(predict[list(X_train2.columns)])
        clf_svc2_proba = svm.SVC(kernel='linear', probability =  True)
        clf_svc2_proba.fit(X_train, y_train)  
        y_pred_new_proba =  pd.DataFrame(clf_svc2_proba.predict_proba(predict[veriables]))   
        print("SVC accuracy:", svc_acc2)
        
    
    
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'SVC', league_abb)
    print("SVC wrzucone")
    #final_SVM2 = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_SVM2 = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_SVM2_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
        
    time.sleep(10)

    
    
    from sklearn.linear_model import SGDClassifier
    clf_sgd = SGDClassifier(loss="log", penalty="l2")
    clf_sgd.fit(X_train, y_train)

    y_pred_sgd=clf_sgd.predict(X_test)
    

    clf_sgd_scaled =  SGDClassifier(loss="log", penalty="l2")
    
    # Train Decision Tree Classifer
    clf_sgd_scaled = clf_sgd_scaled.fit(X_train2[veriables], y_train2)
    
    #Predict the response for test dataset
    y_pred_sgd2 = clf_sgd_scaled.predict(X_test2[veriables])
    # Model Accuracy, how often is the classifier correct?
    
    #print("Accuracy:",sgd_acc_scaled)
    
    if metrics.accuracy_score(y_test, y_pred_sgd)> metrics.accuracy_score(y_test2, y_pred_sgd2):
        y_pred_new = clf_sgd.predict(predict[list(X_train.columns)])
        y_pred_new_proba =  pd.DataFrame(clf_sgd.predict_proba(predict[veriables]))
        print("SVC accuracy:", metrics.accuracy_score(y_test, y_pred_sgd))
        sgd_acc_scaled = metrics.accuracy_score(y_test, y_pred_sgd)
    else:
        y_pred_new = clf_sgd_scaled.predict(predict[list(X_train.columns)])
        y_pred_new_proba =  pd.DataFrame(clf_sgd_scaled.predict_proba(predict[veriables]))    
        print("SVC accuracy:",metrics.accuracy_score(y_test2, y_pred_sgd2))
        sgd_acc_scaled = metrics.accuracy_score(y_test2, y_pred_sgd2)
    
    
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'SGD', league_abb)
    print("SGD wrzucone")
    
    final_SGD = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_SGD = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_SGD_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
        
    time.sleep(10)

    from sklearn.neighbors import KNeighborsClassifier
    
    knn_acc = [None] * 17
    
    
    for i in range(3,20):
        knn = KNeighborsClassifier(n_neighbors=i)
    
        knn.fit(X_train2, y_train2.values.ravel()) 
        knn_acc[i-3] = knn.score(X_test2, y_test2)
    
    
    
    knn = KNeighborsClassifier(n_neighbors=knn_acc.index(max(knn_acc))+3)
    knn.fit(X_train2, y_train2) 
    print("KNN accuracy :",knn.score(X_test2, y_test2))   
    
    
    y_pred_new = knn.predict(predict[list(X_train2.columns)])
    y_pred_new_proba =  pd.DataFrame(knn.predict_proba(predict[list(X_train2.columns)]))
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'KNN', league_abb)
    print("KNN wrzucone")
    #
    #final_KNN = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_KNN = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_KNN_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
        
    time.sleep(10)

    
    
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
    
    
    
    
    time.sleep(10)
    
    from sklearn.ensemble import VotingClassifier
    
    
    # https://www.kaggle.com/den3b81/better-predictions-stacking-with-votingclassifier
    
    # https://scikit-learn.org/stable/glossary.html#term-n-jobs
    
    
    
    
    eclf  = VotingClassifier(estimators=[('lr', logreg_final),
                                         ('dt', clf_dt_scaled) ,
                                         ('rf', rf_clf), 
                                         ('svc', clf_svc_scale),
                                         ('sgd', clf_sgd)
                                         ], voting='soft',
                                        weights = [lr_acc,
                                                   dt_acc_scaled, 
                                                   rfs_metric, 
                                                   svc_acc,
                                                   sgd_acc_scaled])
    
    
    
    
    
    eclf.fit(X_train, y_train) 
    print(eclf.score(X_test, y_test))   
    
    
    
    
    
    y_pred_new = eclf.predict(predict[list(X_train.columns)])
    y_pred_new_proba =  pd.DataFrame(eclf.predict_proba(predict[veriables]))

    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'ECL', league_abb)
    
 
    #final_ECL_to_db = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_val_pred)], axis=1)
    final_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
    
    final_to_db




def MLUpdateOverUnder(league_abb):
    
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns

    from IPython.display import display
    import time
    import pymysql
    db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
    cursor = db.cursor()
    
    cursor.execute("USE S72t1V5idn") # select the database
    
    dics = 'G'
    loc_to_functions = dics + ":/data_football/Football_prediction_ML/functions/"
        
    os.chdir(loc_to_functions)
        
    import functions_for_dataset_preparation as f1
    loc = dics + ":/data_football/final_datasets/"
    dataset = pd.read_csv(loc + 'training_dataset_'+league_abb+'.csv', index_col = 0)
    #dataset = pd.read_csv('C:/Users/szydlikp/DT_opt/training_dataset_E0.csv', index_col = 0)
    df_final = dataset
    df_final['UnOV'] = np.where((df_final['FTHG']+df_final['FTAG'])> 2, 1, 0)
    
    df_final['UnOV'] = df_final['UnOV'].astype('category')
    
    df_final = df_final.loc[~(df_final["MW"] == 1)]
    
    df_final = df_final[['UnOV', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP',
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
    
    df_final["Goals_mean_2"] = df_final["Mean_home_goals"] + df_final["Mean_away_goals"]
    df_final["H2H_Diff"] = df_final["H2H_Home_pts"] - df_final["H2H_Away_pts"]
    df_final["Total_Diff"] = df_final["Total_value_H"] / df_final["Total_value_A"]
    df_final["Age_diff"] = df_final["Age_H"] - df_final["Age_A"]
    df_final["LP_Diff"] = df_final["HT_LP"] - df_final["AT_LP"]
    df_final["ELO_diff"] = df_final["Elo_HT"] - df_final["Elo_AT"]
    
    #df_final.columns
        
 
    
    df_final = df_final[[ 'HTGD', 'ATGD', 'HTGS', 'ATGS',
              'DiffPts', 'DiffFormPts', 'DiffLP',
            'Mean_home_goals', 'Mean_away_goals',  'H2H_Diff', "ELO_diff", 
            'Total_Diff', 'LP_Diff', 'UnOV', 'Goals_mean_2', 'Total_value_H', 
            'Total_value_A', 'H2H_Away_pts', 'H2H_Home_pts']]
    
    #df_final.tail()
    
    
    #df_final = df_final.drop('Mean_home_goals',axis=1)
    df_final = df_final.dropna()
    
    df_final.reset_index(inplace=True, drop=True) 
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_final.drop('UnOV',axis=1), 
               df_final['UnOV'], test_size=0.30, 
                random_state=101)
    
    # https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
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
    
      
    
    X = df_final.loc[:, df_final.columns != 'UnOV']
    y = df_final.loc[:, df_final.columns == 'UnOV']
    
    
    cols_to_norm = ['HTGD', 'ATGD', 'DiffPts' , 'DiffFormPts', 'DiffLP', 'Mean_away_goals', 'Mean_home_goals',
                        'H2H_Diff', 'ELO_diff', 'Total_Diff', 'LP_Diff', 'Goals_mean_2']
    
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
    
 
    
    # ## Weryfikacja modelu 
    
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train[veriables], y_train)
    logreg.coef_
    
   
    
    #y_pred = logreg_final.predict(X_test[veriables])
    #lr_acc = logreg_final.score(X_test[veriables], y_test)
    #print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg_final.score(X_test[veriables], y_test)))
    
    #from sklearn.metrics import confusion_matrix
    #confusion_matrix = confusion_matrix(y_test, y_pred)
    #print(confusion_matrix)
    

    
    time.sleep(10)
    # # Nowe dane
    #predict = pd.read_csv('C:/Users/szydlikp/DT_opt/predict_E0.csv', index_col = 0)
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
       
    predict['UnOV'] = np.nan
    
    predict.HTFormPts = predict.HTFormPts.fillna(predict.HTFormPts.median())
    predict.ATFormPts = predict.ATFormPts.fillna(predict.ATFormPts.median())
    predict.DiffFormPts = predict.DiffFormPts.fillna(predict.DiffFormPts.median())
    
    predict["H2H_Diff"] = predict["H2H_Home_pts"] - predict["H2H_Away_pts"]
    predict["Total_Diff"] = predict["Total_value_H"] / predict["Total_value_A"]
    predict["Age_diff"] = predict["Age_H"] - predict["Age_A"]
    predict["LP_Diff"] = predict["HT_LP"] - predict["AT_LP"]
    predict["ELO_diff"] = predict["Elo_HT"] - predict["Elo_AT"]
        
    predict["Goals_mean_2"] = predict["Mean_home_goals"] + predict["Mean_away_goals"]
    
    predict =predict[['HomeTeam', 'AwayTeam', 'HTGD', 'ATGD', 'HTGS', 'ATGS',
              'DiffPts', 'DiffFormPts', 'DiffLP',
            'Mean_home_goals', 'Mean_away_goals',  'H2H_Diff', "ELO_diff", 
            'Total_Diff', 'LP_Diff', 'UnOV', 'Goals_mean_2', 'Total_value_H', 
            'Total_value_A', 'H2H_Away_pts', 'H2H_Home_pts']]

    
    #predict = predict.drop('Mean_home_goals',axis=1)
    
    
    #predict.tail()
    
    
    predict = predict.reset_index().drop('Date', axis = 1)
    
    
    y_pred_new = logreg.predict(predict[veriables])
    y_pred_new_proba =  pd.DataFrame(logreg.predict_proba(predict[veriables]))
    
    #y_pred_new
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'LR', league_abb)
    
    
    final_logreg = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_logreg_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
    
    time.sleep(10)

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
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    
    # ### Przeskalowane
    
    
    # Create Decision Tree classifer object
    clf_dt_scaled =  DecisionTreeClassifier(criterion="entropy", max_depth=5)
    
    # Train Decision Tree Classifer
    clf_dt_scaled = clf_dt_scaled.fit(X_train2[veriables], y_train2)
    
    #Predict the response for test dataset
    y_pred = clf_dt_scaled.predict(X_test2[veriables])
    # Model Accuracy, how often is the classifier correct?
    dt_acc_scaled = metrics.accuracy_score(y_test2, y_pred)
    #print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
    
    
    if metrics.accuracy_score(y_test, y_pred) > metrics.accuracy_score(y_test2, y_pred):
        y_pred_new = clf.predict(predict[veriables])
        y_pred_new_proba =  pd.DataFrame(clf.predict_proba(predict[veriables]))
    else:
        y_pred_new = clf_dt_scaled.predict(predict[veriables])
        y_pred_new_proba =  pd.DataFrame(clf_dt_scaled.predict_proba(predict[veriables]))    
    
    
    #print(classification_report(y_test2, y_pred))
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'DT', league_abb)
    
    #final_DT = pd.concat([predict[[ "HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_DT = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_DT_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
    
    time.sleep(10)

    
    #  Random Forest
    #Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    
    #Create a Gaussian Classifier
    clf_rf=RandomForestClassifier(n_estimators=250)
    
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf_rf.fit(X_train,    y_train.values.ravel())

    y_pred_lr = clf_rf.predict(X_test)
    #y_pred = clf_rf.predict(predict[list(X_train.columns)])

    
    # Create Decision Tree classifer object
    clf_rf_scaled =  RandomForestClassifier(n_estimators=250)
    
    # Train Decision Tree Classifer
    clf_rf_scaled = clf_rf_scaled.fit(X_train2[veriables], y_train2.values.ravel())
    
    #Predict the response for test dataset
    y_pred = clf_rf_scaled.predict(X_test2[veriables])
    # Model Accuracy, how often is the classifier correct?
    rf_acc_scaled = metrics.accuracy_score(y_test2, y_pred)
    #print("Accuracy:",metrics.accuracy_score(y_test2, y_pred))
    
    if metrics.accuracy_score(y_test, y_pred_lr) > metrics.accuracy_score(y_test2, y_pred):
        y_pred_new = clf_rf.predict(predict[list(X_train.columns)])
        y_pred_new_proba =  pd.DataFrame(clf_rf.predict_proba(predict[list(X_train.columns)]))
    else:
        y_pred_new = clf_rf_scaled.predict(predict[veriables])
        y_pred_new_proba =  pd.DataFrame(clf_rf_scaled.predict_proba(predict[veriables]))    
    
    
    
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'RF', league_abb)
    
    
    
    #final_RF = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_RF = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_RF_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
        
    time.sleep(5)

    
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

    clf_rf2.fit(X_train,y_train)
    
    # prediction on test set
    y_pred_rf=clf_rf2.predict(X_test)
        
    y_pred_new = clf_rf2.predict(predict[list(X_train.columns)])
    y_pred_new_proba =  pd.DataFrame(clf_rf2.predict_proba(predict[veriables]))
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'RFs', league_abb)
    #
    #final_RF2 = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_RF2 = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_RF2_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
    
    time.sleep(5)

    
    # # SVM
    from sklearn import svm
    clf_svc_scale = svm.SVC(gamma='scale', probability=True)
    clf_svc_scale.fit(X_train, y_train)  
    
    # prediction on test set
    y_pred_svc=clf_svc_scale.predict(X_test)
    
    svc_acc = metrics.accuracy_score(y_test, y_pred_svc)
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svc))
    
    
    # ### Przeskalowane
    
    clf_svc2 = svm.SVC(kernel='linear', probability=True)
    clf_svc2.fit(X_train, y_train)  
    
    # prediction on test set
    y_pred_svc2=clf_svc2.predict(X_test)
    
    svc_acc2 =metrics.accuracy_score(y_test, y_pred_svc2)
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svc2))
    
    if svc_acc > svc_acc2:
        y_pred_new = clf_svc_scale.predict(predict[list(X_train.columns)])
        clf_svc_proba = svm.SVC(kernel='linear', probability =  True)
        clf_svc_proba.fit(X_train, y_train)  
        y_pred_new_proba =  pd.DataFrame(clf_svc_proba.predict_proba(predict[veriables]))
    else:
        y_pred_new = clf_svc2.predict(predict[list(X_train.columns)])
        clf_svc2_proba = svm.SVC(kernel='linear', probability =  True)
        clf_svc2_proba.fit(X_train, y_train)  
    
        y_pred_new_proba =  pd.DataFrame(clf_svc2_proba.predict_proba(predict[veriables]))    
    
    
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'SVC', league_abb)
    
    #final_SVM2 = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_SVM2 = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_SVM2_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
        
    time.sleep(10)

    
    
    from sklearn.linear_model import SGDClassifier
    clf_sgd = SGDClassifier(loss="log", penalty="l2")
    clf_sgd.fit(X_train, y_train)

    y_pred_sgd=clf_sgd.predict(X_test)
    

    clf_sgd_scaled =  SGDClassifier(loss="log", penalty="l2")
    
    # Train Decision Tree Classifer
    clf_sgd_scaled = clf_sgd_scaled.fit(X_train2[veriables], y_train2)
    
    #Predict the response for test dataset
    y_pred_sgd2 = clf_sgd_scaled.predict(X_test2[veriables])
    # Model Accuracy, how often is the classifier correct?
    sgd_acc_scaled = metrics.accuracy_score(y_test2, y_pred)
    #print("Accuracy:",sgd_acc_scaled)
    
    if metrics.accuracy_score(y_test, y_pred_sgd)> metrics.accuracy_score(y_test2, y_pred_sgd2):
        y_pred_new = clf_sgd.predict(predict[list(X_train.columns)])
        y_pred_new_proba =  pd.DataFrame(clf_sgd.predict_proba(predict[veriables]))
    else:
        y_pred_new = clf_sgd_scaled.predict(predict[list(X_train.columns)])
        y_pred_new_proba =  pd.DataFrame(clf_sgd_scaled.predict_proba(predict[veriables]))    
    
    
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'SGD', league_abb)
    
    final_SGD = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_SGD = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_SGD_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
        
    time.sleep(10)

    from sklearn.neighbors import KNeighborsClassifier
    
    knn_acc = [None] * 17
    
    
    for i in range(3,20):
        knn = KNeighborsClassifier(n_neighbors=i)
    
        knn.fit(X_train, y_train) 
        knn_acc[i-3] = knn.score(X_test, y_test)
    
    
    
    knn = KNeighborsClassifier(n_neighbors=knn_acc.index(max(knn_acc))+3)
    knn.fit(X_train, y_train) 
    #print(knn.score(X_test, y_test))   
    
    
    
    
    y_pred_new = knn.predict(predict[list(X_train.columns)])
    y_pred_new_proba =  pd.DataFrame(knn.predict_proba(predict[veriables]))
    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'KNN', league_abb)
    #
    #final_KNN = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_pred_new)], axis=1)
    final_KNN = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                              pd.DataFrame(y_pred_new)], axis=1)
     #                         y_pred_new_proba.iloc[:,1]], axis=1)
    final_KNN_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
        
    time.sleep(10)

    
    
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
    
    
    
    
    time.sleep(10)
    
    from sklearn.ensemble import VotingClassifier
    
    
    # https://www.kaggle.com/den3b81/better-predictions-stacking-with-votingclassifier
    
    # https://scikit-learn.org/stable/glossary.html#term-n-jobs
    
    
    
    
    eclf  = VotingClassifier(estimators=[('lr', logreg),
                                         ('dt', clf_dt_scaled) ,
                                         ('rf', clf_rf_scaled), 
                                         ('svc', clf_svc_scale),
                                         ('sgd', clf_sgd)
                                         ], voting='soft',
                                        weights = [lr_acc,
                                                   dt_acc_scaled, 
                                                   rf_acc_scaled, 
                                                   svc_acc,
                                                   sgd_acc_scaled])
    
    
    
    
    
    eclf.fit(X_train, y_train) 
    print(eclf.score(X_test, y_test))   
    
    
    
    
    
    y_pred_new = eclf.predict(predict[list(X_train.columns)])
    y_pred_new_proba =  pd.DataFrame(eclf.predict_proba(predict[veriables]))

    
    f1.DataBaseMLUpdateOrInsert(predict, predict_mw, predict_date, y_pred_new, y_pred_new_proba, 'ECL', league_abb)
    
 
    #final_ECL_to_db = pd.concat([predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), pd.DataFrame(y_val_pred)], axis=1)
    final_to_db = pd.concat([pd.DataFrame(predict_date),
                                   predict[["HomeTeam", "AwayTeam"]].reset_index(drop=True), 
                                    pd.DataFrame(y_pred_new),
                                    y_pred_new_proba.iloc[:,1]], axis=1)
    
    final_to_db

MLUpdate('D1')
MLUpdate('SP2')

MLUpdate1_X2('E0')
MLUpdate1_X2('E1')
MLUpdate('F2')



abb = ['E0', 'E1','I1', 'D1']
abb = ['D2', 'F1','F2', 'N1', 'P1', 'SP1', 'SP2']
abb = ['E0', 'E1', 'D1', 'D2', 'SP1','SP2','I1', 'N1', 'P1', 'F1', 'F2']

for league in abb:
    MLUpdate(league)
