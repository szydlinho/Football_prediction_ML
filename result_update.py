# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:03:33 2019

@author: base
"""

import pandas as pd
import pymysql
import numpy as np



#cursor.execute("UPDATE predictions SET prediction = %s WHERE  MW=%s and model= 'ECL' and league = %s  AND HomeTeam = %s and AwayTeam = %s", tuple_temp)
#db.commit()



abb = ['E0', 'E1', 'D1', 'D2', 'I1', 'F1', 'F2','SP1', 'SP2', 'N1', 'P1']

dataset = {}

for league in abb:
    dataset[league] = pd.read_csv('https://www.football-data.co.uk/mmz4281/19-20'+'/'+league+'.csv', encoding = 'unicode_escape')


db = pymysql.connect(host='remotemysql.com',user='S72t1V5idn',passwd='dEewXLcrbR')
cursor = db.cursor()

cursor.execute("USE S72t1V5idn") # select the database


old_df = pd.read_sql("SELECT * FROM predictions WHERE Result = '' OR result_binary is NULL", db)

for i in range(0, len(old_df)):
    #dataset[ old_df.iloc[i]["league"]]
    
    #old_df.iloc[i]["HomeTeam"]
    row = dataset[ old_df.iloc[i]["league"]][(dataset[ old_df.iloc[i]["league"]]["HomeTeam"]== old_df.iloc[i]["HomeTeam"]) & (dataset[ old_df.iloc[i]["league"]]["AwayTeam"]== old_df.iloc[i]["AwayTeam"])]
    if len(row) == 1:
        
        result_binary = int(np.where(int(row["FTHG"]) > int(row["FTAG"]) , 1, 0))
        tuple_temp = (str(int(row["FTHG"]))+"-"+str(int(row["FTAG"])),
               old_df.iloc[i]["HomeTeam"],
               old_df.iloc[i]["AwayTeam"],
               old_df.iloc[i]["league"]
          )
        
        
        tuple_temp2 = (result_binary,
               old_df.iloc[i]["HomeTeam"],
               old_df.iloc[i]["AwayTeam"],
               old_df.iloc[i]["league"]
          )
        
        
    
        cursor.execute("UPDATE predictions SET Result = %s WHERE  HomeTeam=%s and AwayTeam= %s and league = %s  ", tuple_temp)
        cursor.execute("UPDATE predictions SET result_binary = %s WHERE  HomeTeam=%s and AwayTeam= %s and league = %s  ", tuple_temp2)
       
        db.commit()



cursor.execute("UPDATE predictions SET corrected = 1 WHERE result_binary = prediction")
db.commit()

cursor.execute("UPDATE predictions SET corrected = 0 WHERE result_binary <> prediction")
db.commit()