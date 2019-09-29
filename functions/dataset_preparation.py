# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:58:30 2019

@author: pszydlik
"""


import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
#import itertools
#import progressbar
import datetime
import requests
pd.options.mode.chained_assignment = None
#pbar = progressbar.ProgressBar().start()
from datetime import timedelta  
import os, re
discks_names =  re.findall(r"[A-Z]+:.*$",os.popen("mountvol /").read(),re.MULTILINE)

a = []
for i in range(len(discks_names)):
    b = discks_names[i].split(":")[0]
    a.append(b)

dics = a['C' in a]
dics = a[2]
#dics = 'D'    


#Premier League: https://www.football-data.co.uk/mmz4281/1819/E0.csv
#Championship: https://www.football-data.co.uk/mmz4281/1819/E1.csv
#Bundesliga 1: https://www.football-data.co.uk/mmz4281/1819/D1.csv
#Bundesliga 2:  https://www.football-data.co.uk/mmz4281/1819/D2.csv
#Serie A: https://www.football-data.co.uk/mmz4281/1819/I1.csv
#Serie B: https://www.football-data.co.uk/mmz4281/1819/I2.csv
#Premierea Division: https://www.football-data.co.uk/mmz4281/1819/SP1.csv
#Segunda Division: https://www.football-data.o.uk/mmz4281/1819/SP2.csv
#ligue 1: https://www.football-data.co.uk/mmz4281/1819/F1.csv
#Ligue 2: https://www.football-data.co.uk/mmz4281/1819/F2.csv
#Netherland 1: https://www.football-data.co.uk/mmz4281/1819/N1.csv
#Portugal 1: https://www.football-data.co.uk/mmz4281/1819/P1.csv

abb = ['E0', 'E1', 'D1', 'D2', 'I1', 'I2', 'SP1','SP2','F1', 'F2', 'N1', 'P1']

league_abb = 'E1'
#


league = 'germany/2-bundesliga'
league_url = 'germany/2_bundesliga/'
league_url = 'england/premier_league/'
league_url = 'england/championship/'
league_url = 'germany/1_bundesliga/'
league_url = 'italy/serie_a/'
league_url = 'italy/serie_b_italy_football/'
league_url = 'france/ligue_1/'
league_url = 'france/ligue_2/'
league_url = 'netherlands/eredivisie/'
league_url = 'portugal/primeira_liga/'
def PrepareDataset(league_abb, league, data_uk = False):
    
    loc = dics + ":/data_football/final_datasets/"
    
    loc_to_functions = dics + ":/data_football/"
    
    os.chdir(loc_to_functions)
    
    import functions as f1
    
    dataset = {}
    # 10strings: `raw_data_10, .. ,raw_data_19`
    
    
    if league_abb == 'P1':
        dataset_names = ["raw_data_%d" % x for x in range(15, 21)]
    else:
        dataset_names = ["raw_data_%d" % x for x in range(10, 21)]
 
    if league_abb == 'P1':
        sezony = ["14-15", "15-16", "16-17", "17-18", "18-19", "19-20"]
    else:
        sezony = ["09-10", "10-11", "11-12", "12-13", "13-14","14-15", "15-16",
              "16-17", "17-18", "18-19", "19-20"]
    
    for (data_set, sezon) in zip(dataset_names, sezony):
        try:
            dataset[data_set] = pd.read_csv('https://www.football-data.co.uk/mmz4281/'+
                                   sezon+'/'+league_abb+'.csv', encoding = 'unicode_escape')
        except:
            dataset[data_set] = pd.read_csv('https://www.football-data.co.uk/mmz4281/'+
                                   sezon.replace("-", "") +'/'+league_abb + '.csv')
     
    
    a = pd.concat(dataset,  sort=False)
    a.HomeTeam.unique()
    pd.Series(a['HomeTeam'].unique()).sort_values()
    for data_set in dataset_names:
        dataset[data_set]= dataset[data_set].dropna(thresh=2) 
        dataset[data_set].Date = pd.to_datetime(dataset[data_set].Date)
    

    if(len(dataset["raw_data_15"])% 8 == 0):
        number_of_MW = 8
    elif (len(dataset["raw_data_15"])% 10 == 0):
        number_of_MW = 10
    elif(len(dataset["raw_data_15"])% 9 == 0):
        number_of_MW = 9
    elif(len(dataset["raw_data_15"])% 12 == 0):
        number_of_MW = 12
    elif(len(dataset["raw_data_15"])% 11 == 0):
        number_of_MW = 11
        
    for data_set in dataset_names:
        f1.fill_with_mw( dataset[data_set], number_of_MW)
        
    #Wybor tylko tych kolumn, ktore potrzebuje
    columns_req = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR', 'MW']
    if league_abb == 'P1':
        dataset_names_mew =  ["%d" % x for x in range(15, 21)]
    else:
        dataset_names_mew = ["%d" % x for x in range(10, 21)]
        
    playing_stat = {}
    
    for (data_set, datase_old) in zip(dataset_names_mew, dataset_names):
         playing_stat[data_set] = dataset[datase_old][columns_req]
         
    for data_set in dataset_names_mew:
         playing_stat[data_set] =  playing_stat[data_set].dropna()
    

    #playing_stat['10'] = playing_stat['10'].replace('Piacenza ', 'Piacenza')
    
    for data_set in dataset_names_mew:
        playing_stat[data_set]['HomeTeam'] = playing_stat[data_set]['HomeTeam'].str.strip()
        playing_stat[data_set]['AwayTeam'] = playing_stat[data_set]['AwayTeam'].str.strip()
    #Suma goli w sezonie dla kazdej z druzyn oraz numer kolejki
    #bramki
    for data_set in dataset_names_mew:
        playing_stat[data_set] = f1.get_gss(playing_stat[data_set], number_of_MW)
    
    #PUNKTACJA###############################################
    for data_set in dataset_names_mew:
        playing_stat[data_set] = f1.get_agg_points(playing_stat[data_set], number_of_MW)

    for data_set in dataset_names_mew:
        playing_stat[data_set] = f1.get_agg_wins(playing_stat[data_set], number_of_MW)
         
    for data_set in dataset_names_mew:
        playing_stat[data_set] = f1.get_agg_losses(playing_stat[data_set], number_of_MW)

    for data_set in dataset_names_mew:
        playing_stat[data_set] = f1.get_agg_draws(playing_stat[data_set], number_of_MW)

    
    #####FORMA DRUZYNY ##################################
    for data_set in dataset_names_mew:
        playing_stat[data_set]  = f1.add_form_df(playing_stat[data_set] , number_of_MW)
    
    # Rearranging columns
    #cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3',
    #    'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5','HT_Wins', 'AT_Wins', 'HT_Loss', 'AT_Loss', 'HT_Draws', 'AT_Draws', 'MW']

    
    #for data_set in dataset_names_mew:
    #     playing_stat[data_set] = playing_stat[data_set][cols]
    
    names = pd.read_csv(dics + ":/data_football/dictionary.csv", sep = ";")
    names = names.drop_duplicates()
    names.columns = ["old", "new"]
    names_dict = pd.Series(names.old.values,index=names.new).to_dict()
    names_dict2 = pd.Series(names.new.values,index=names.old).to_dict()
    

    Standings =  pd.read_csv(dics + ':/data_football/tables_positions/'+ names_dict[league_abb] +'.csv', header=0)
    Standings.set_index(['Team'], inplace=True)
    Standings = Standings.fillna(round(number_of_MW*2*0.75))
    
    years =  ["%d" % x for x in range(2010, 2020)]
    
    for (data_set, year) in zip(dataset_names_mew, years):
         playing_stat[data_set] = f1.get_last(playing_stat[data_set], Standings, year)
    
    
    for data_set in dataset_names_mew:
         playing_stat[data_set]["Sezon"] = int(data_set)
    
    for data_set in dataset_names_mew:
        f1.standings_live(playing_stat[data_set], number_of_MW)
    
    
    dataset_final = pd.concat(playing_stat, sort=False)
    
    dataset_final.to_csv(loc + league_abb + "_raw_final.csv", index=False, sep = ',')
    #dataset_final.filter(regex='HM')
    dataset_final['HM1'] = dataset_final['HM1'].fillna('M')
    dataset_final['HM2'] = dataset_final['HM2'].fillna('M')
    dataset_final['HM3'] = dataset_final['HM3'].fillna('M')
    dataset_final['HM4'] = dataset_final['HM4'].fillna('M')
    dataset_final['HM5'] = dataset_final['HM5'].fillna('M')
    
    dataset_final['AM1'] = dataset_final['AM1'].fillna('M')
    dataset_final['AM2'] = dataset_final['AM2'].fillna('M')
    dataset_final['AM3'] = dataset_final['AM3'].fillna('M')
    dataset_final['AM4'] = dataset_final['AM4'].fillna('M')
    dataset_final['AM5'] = dataset_final['AM5'].fillna('M')
    
    
    dataset_final['HTFormPtsStr'] = dataset_final['HM1'] + dataset_final['HM2'] + dataset_final['HM3'] + dataset_final['HM4'] + dataset_final['HM5']
    dataset_final['ATFormPtsStr'] = dataset_final['AM1'] + dataset_final['AM2'] + dataset_final['AM3'] + dataset_final['AM4'] + dataset_final['AM5']
    
    dataset_final['HTFormPts'] = dataset_final['HTFormPtsStr'].apply(f1.get_form_points, number_of_MW)
    dataset_final['ATFormPts'] = dataset_final['ATFormPtsStr'].apply(f1.get_form_points, number_of_MW)
    
    
    dataset_final['HTWinStreak3'] = dataset_final['HTFormPtsStr'].apply(f1.get_3game_ws)
    dataset_final['HTWinStreak5'] = dataset_final['HTFormPtsStr'].apply(f1.get_5game_ws)
    dataset_final['HTLossStreak3'] = dataset_final['HTFormPtsStr'].apply(f1.get_3game_ls)
    dataset_final['HTLossStreak5'] = dataset_final['HTFormPtsStr'].apply(f1.get_5game_ls)
    dataset_final['HTDrawStreak3'] = dataset_final['HTFormPtsStr'].apply(f1.get_3game_ds)
    dataset_final['HTDrawStreak5'] = dataset_final['HTFormPtsStr'].apply(f1.get_5game_ds)
    
    dataset_final['ATWinStreak3'] = dataset_final['ATFormPtsStr'].apply(f1.get_3game_ws)
    dataset_final['ATWinStreak5'] = dataset_final['ATFormPtsStr'].apply(f1.get_5game_ws)
    dataset_final['ATLossStreak3'] = dataset_final['ATFormPtsStr'].apply(f1.get_3game_ls)
    dataset_final['ATLossStreak5'] = dataset_final['ATFormPtsStr'].apply(f1.get_5game_ls)
    dataset_final['ATDrawStreak3'] = dataset_final['ATFormPtsStr'].apply(f1.get_3game_ds)
    dataset_final['ATDrawStreak5'] = dataset_final['ATFormPtsStr'].apply(f1.get_5game_ds)
    


    # Get Goal Difference
    dataset_final['HTGD'] = dataset_final['HTGS'] - dataset_final['HTGC']
    dataset_final['ATGD'] = dataset_final['ATGS'] - dataset_final['ATGC']
    
    # Diff in points
    dataset_final['DiffPts'] = dataset_final['HTP'] - dataset_final['ATP']
    dataset_final['DiffFormPts'] = dataset_final['HTFormPts'] - dataset_final['ATFormPts']
    
    # Diff in last year positions
    dataset_final['DiffLP'] = dataset_final['HomeTeamLP'] - dataset_final['AwayTeamLP']
    
    #New fixtures
    #league = 'england/premier-league'
    
    #url = 'https://us.soccerway.com/national/'+ league + '/20192020/regular-season/'
    #response = requests.get(url)
    #tables = pd.read_html(response.text)
    #table = tables[0]
    #table[table["Score/Time"]]
    if  data_uk == True: 
        fixtures =     pd.read_csv("https://www.football-data.co.uk/fixtures.csv")
        fixtures = fixtures[fixtures["Div"] == league_abb]
        fixtures = fixtures[["Date", "HomeTeam", "AwayTeam"]]
    else:
        url = 'http://www.tablesleague.com/' + league_url
        response = requests.get(url)
        tables = pd.read_html(response.text)
        table = tables[1]
        table.columns = table.iloc[0]
        table = table.drop(table.index[0])
        fixtures = pd.concat([table["Date"], table["Game"].str.split(" - ", expand= True)], axis=1)
        fixtures.Date = fixtures.Date.str.replace("Today", datetime.date.today().strftime("%d %b"))
        fixtures.Date = pd.to_datetime(fixtures.Date ,  format = "%d %b %H:%M")
        fixtures.columns = ['Date', 'HomeTeam', 'AwayTeam']
        
        for i in range(0, len(fixtures)):
            fixtures["HomeTeam"].iloc[i] = fixtures["HomeTeam"].iloc[i].replace(fixtures["HomeTeam"].iloc[i], names_dict2[fixtures["HomeTeam"].iloc[i]])
        
        for i in range(0, len(fixtures)):
            fixtures["AwayTeam"].iloc[i] = fixtures["AwayTeam"].iloc[i].replace(fixtures["AwayTeam"].iloc[i], names_dict2[fixtures["AwayTeam"].iloc[i]])
        
    #fixtures.columns = ['Date', 'HomeTeam', 'AwayTeam']
    #fixtures = fixtures.dropna(how='all', axis=0)
    #fixtures = fixtures[pd.notnull(fixtures['HomeTeam'])]
    #Tu trzeba jakos zrobic aby dzielił mecze na MW, bo na stronie jest wiecej niz z jednej kolejki

    fixtures["MW"] = 0
    if dataset_final.iloc[len(dataset_final)-1]["MW"] == ((number_of_MW * 4)-2):
        fixtures["MW"][0:number_of_MW] = 1
        fixtures["MW"][number_of_MW:] = 2
    else:
        fixtures["MW"][0:number_of_MW] = dataset_final.iloc[len(dataset_final)-1]["MW"] + 1
        fixtures["MW"][number_of_MW:] = dataset_final.iloc[len(dataset_final)-1]["MW"] + 2
    
    fixtures.Date = pd.to_datetime(fixtures.Date)
    for i in range(0, len(fixtures)):
        fixtures.Date.iloc[i] = fixtures.Date.iloc[i].replace(year = 2019)

    fixtures["Sezon"] = 20
    #fixtures["MW"] = int(max(playing_statistics_19['MW'])+1)
    if fixtures.iloc[len(fixtures)-1]["MW"] != 1:
        fixtures = playing_stat['20'].append(fixtures, sort=False)
        fixtures = f1.get_gss(fixtures, number_of_MW)
        fixtures = f1.get_agg_points(fixtures, number_of_MW)
        fixtures = f1.get_agg_wins(fixtures, number_of_MW)
        fixtures = f1.get_agg_losses(fixtures, number_of_MW)
        fixtures = f1.get_agg_draws(fixtures, number_of_MW)
        fixtures = f1.add_form_df(fixtures, number_of_MW)
        fixtures.reset_index(inplace=True, drop=True) 
        f1.standings_live(fixtures, number_of_MW)
        fixtures = f1.get_last(fixtures, Standings, "2019")
        fixtures['HTFormPtsStr'] = fixtures['HM1'] + fixtures['HM2'] #+ fixtures['HM3'] + fixtures['HM4'] + fixtures['HM5']
        fixtures['ATFormPtsStr'] = fixtures['AM1'] + fixtures['AM2'] #+ #fixtures['AM3'] + fixtures['AM4'] + fixtures['AM5']
        
        fixtures['HTFormPts'] = fixtures['HTFormPtsStr'].apply(f1.get_form_points)
        fixtures['ATFormPts'] = fixtures['ATFormPtsStr'].apply(f1.get_form_points)
        
        # Get Goal Difference
        fixtures['HTGD'] = fixtures['HTGS'] - fixtures['HTGC']
        fixtures['ATGD'] = fixtures['ATGS'] - fixtures['ATGC']
        
        # Diff in points
        fixtures['DiffPts'] = fixtures['HTP'] - fixtures['ATP']
        fixtures['DiffFormPts'] = fixtures['HTFormPts'] - fixtures['ATFormPts']
        
        # Diff in last year positions
        fixtures['DiffLP'] = fixtures['HomeTeamLP'] - fixtures['AwayTeamLP']
        
        fixtures['HTWinStreak3'] = fixtures['HTFormPtsStr'].apply(f1.get_3game_ws)
        fixtures['HTWinStreak5'] = fixtures['HTFormPtsStr'].apply(f1.get_5game_ws)
        fixtures['HTLossStreak3'] = fixtures['HTFormPtsStr'].apply(f1.get_3game_ls)
        fixtures['HTLossStreak5'] = fixtures['HTFormPtsStr'].apply(f1.get_5game_ls)
        fixtures['HTDrawStreak3'] = fixtures['HTFormPtsStr'].apply(f1.get_3game_ds)
        fixtures['HTDrawStreak5'] = fixtures['HTFormPtsStr'].apply(f1.get_5game_ds)
        
        fixtures['ATWinStreak3'] = fixtures['ATFormPtsStr'].apply(f1.get_3game_ws)
        fixtures['ATWinStreak5'] = fixtures['ATFormPtsStr'].apply(f1.get_5game_ws)
        fixtures['ATLossStreak3'] = fixtures['ATFormPtsStr'].apply(f1.get_3game_ls)
        fixtures['ATLossStreak5'] = fixtures['ATFormPtsStr'].apply(f1.get_5game_ls)
        fixtures['ATDrawStreak3'] = fixtures['ATFormPtsStr'].apply(f1.get_3game_ds)
        fixtures['ATDrawStreak5'] = fixtures['ATFormPtsStr'].apply(f1.get_5game_ds)
    elif fixtures.iloc[len(fixtures)-1]["MW"] == 1:
        new_cols = ['HTGS','ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 'HT_Loss',
                    'AT_Loss','HT_Wins', 'AT_Wins', 'HT_Draws', 'AT_Draws',
                    ]
        for col in new_cols:
            fixtures[col] = 0
        
        new_cols_2 = ['HM1', 'HM2', 'HM3', 'HM4', 'HM5', 
                      'AM5', 'AM4', 'AM3', 'AM2', 'AM1']
    
        for col in new_cols_2:
            fixtures[col] = 'M'
            
#        for i in range(0, len(fixtures)):
#            fixtures["HomeTeam"].iloc[i] = fixtures["HomeTeam"].iloc[i].replace(fixtures["HomeTeam"].iloc[i], names_dict2[fixtures["HomeTeam"].iloc[i]])
#        
#        for i in range(0, len(fixtures)):
#            fixtures["AwayTeam"].iloc[i] = fixtures["AwayTeam"].iloc[i].replace(fixtures["AwayTeam"].iloc[i], names_dict2[fixtures["AwayTeam"].iloc[i]])
#        
        fixtures.reset_index(inplace=True, drop=True) 
        fixtures = f1.standings_live(fixtures, number_of_MW)
        
        for i in range(0, len(fixtures)):
            if fixtures["AwayTeam"].iloc[i] not in Standings.index:
                 Standings = Standings.append(pd.Series(name=fixtures["AwayTeam"].iloc[i]))
                 Standings = Standings.fillna(round(number_of_MW*2*0.75))

                 
        for i in range(0, len(fixtures)):
            if fixtures["HomeTeam"].iloc[i] not in Standings.index:
                 Standings = Standings.append(pd.Series(name=fixtures["HomeTeam"].iloc[i]))
                 Standings = Standings.fillna(round(number_of_MW*2*0.75))
                 #Standings.to_csv(dics + ":/data_football/tables_positions/" + str(league.split("/")[1]) + ".csv",
                 #          index=True, sep = ',')    
                 
        fixtures = f1.get_last(fixtures, Standings, "2019")
        
        fixtures['HTFormPtsStr'] = fixtures['HM1'] + fixtures['HM2'] + fixtures['HM3'] + fixtures['HM4'] + fixtures['HM5']
        fixtures['ATFormPtsStr'] = fixtures['AM1'] + fixtures['AM2'] + fixtures['AM3'] + fixtures['AM4'] + fixtures['AM5']
        
        fixtures['HTFormPts'] = fixtures['HTFormPtsStr'].apply(f1.get_form_points)
        fixtures['ATFormPts'] = fixtures['ATFormPtsStr'].apply(f1.get_form_points)
        
        # Get Goal Difference
        fixtures['HTGD'] = fixtures['HTGS'] - fixtures['HTGC']
        fixtures['ATGD'] = fixtures['ATGS'] - fixtures['ATGC']
        
        # Diff in points
        fixtures['DiffPts'] = fixtures['HTP'] - fixtures['ATP']
        fixtures['DiffFormPts'] = fixtures['HTFormPts'] - fixtures['ATFormPts']
        
        # Diff in last year positions
        fixtures['DiffLP'] = fixtures['HomeTeamLP'] - fixtures['AwayTeamLP']
        
        fixtures['HTWinStreak3'] = fixtures['HTFormPtsStr'].apply(f1.get_3game_ws)
        fixtures['HTWinStreak5'] = fixtures['HTFormPtsStr'].apply(f1.get_5game_ws)
        fixtures['HTLossStreak3'] = fixtures['HTFormPtsStr'].apply(f1.get_3game_ls)
        fixtures['HTLossStreak5'] = fixtures['HTFormPtsStr'].apply(f1.get_5game_ls)
        fixtures['HTDrawStreak3'] = fixtures['HTFormPtsStr'].apply(f1.get_3game_ds)
        fixtures['HTDrawStreak5'] = fixtures['HTFormPtsStr'].apply(f1.get_5game_ds)
        
        fixtures['ATWinStreak3'] = fixtures['ATFormPtsStr'].apply(f1.get_3game_ws)
        fixtures['ATWinStreak5'] = fixtures['ATFormPtsStr'].apply(f1.get_5game_ws)
        fixtures['ATLossStreak3'] = fixtures['ATFormPtsStr'].apply(f1.get_3game_ls)
        fixtures['ATLossStreak5'] = fixtures['ATFormPtsStr'].apply(f1.get_5game_ls)
        fixtures['ATDrawStreak3'] = fixtures['ATFormPtsStr'].apply(f1.get_3game_ds)
        fixtures['ATDrawStreak5'] = fixtures['ATFormPtsStr'].apply(f1.get_5game_ds)
    
    final = pd.concat([ dataset_final.loc[dataset_final['Sezon']!=20], fixtures], ignore_index=True, sort = False) 
        
    f1.h2h(final)
    f1.get_points_h2h(final, "H2H_Home")  
    f1.get_points_h2h(final, "H2H_Away")
    f1.off_and_deff(final)
    #final.to_csv(loc + league_abb + "_log.csv")
    
    #Elo ranking
    names_elo = pd.read_csv(dics + ":/data_football/dictionary_elo.csv", sep = ";", header = None)
    names_elo = names_elo.drop_duplicates()
    names_elo.columns = ["old", "new"]
    names_dict_elo = pd.Series(names_elo.new.values,index=names_elo.old).to_dict()
    
    temp = final.loc[final['MW']==1]
    final["Elo_HT"] = 0
    final["Elo_AT"] = 0
     
    if len(temp) % number_of_MW == 0:
         for i in range(0, len(temp), number_of_MW):
             #.strftime("%Y-%m-%d")
             date = str(str(temp["Date"].iloc[i].year) + '-09-20')
             sezon = temp["Sezon"].iloc[i]
             elo = pd.read_csv('http://api.clubelo.com/' + date)[["Club", "Elo"]]
            
             for ind in final.loc[final["Sezon"] == sezon].index.tolist():
                if final["HomeTeam"].iloc[ind] in elo.Club.tolist():
                    final.iloc[ind, final.columns.get_loc("Elo_HT")] = int(elo.loc[elo["Club"] ==  final["HomeTeam"].iloc[ind]]["Elo"])
                elif final["HomeTeam"].iloc[ind] not in elo.Club.tolist():
                    final.iloc[ind, final.columns.get_loc("Elo_HT")] = int(elo.loc[elo["Club"] ==   names_dict_elo[final["HomeTeam"].iloc[ind]]]["Elo"])
                   
                if final["AwayTeam"].iloc[ind] in elo.Club.tolist():
                    final.iloc[ind, final.columns.get_loc("Elo_AT")] = int(elo.loc[elo["Club"] == final["AwayTeam"].iloc[ind]]["Elo"])
                elif final["AwayTeam"].iloc[ind] not in elo.Club.tolist():
                    final.iloc[ind, final.columns.get_loc("Elo_AT")] = int(elo.loc[elo["Club"] ==  names_dict_elo[final["AwayTeam"].iloc[ind]]]["Elo"])
     
      
    final.to_csv(loc + league_abb + "_log.csv")
    #to_predict = final.loc[final['Sezon']==20]
    
    #if (len(fixtures) % number_of_MW) == 0:
    #    to_predict = fixtures
    #else:
    #to_predict = to_predict.loc[to_predict["MW"]== int(max(playing_stat['20']['MW'])+1)]
    
    
    #to_predict.to_csv(loc + "predict_"+league_abb+".csv", index=False)
    #print(league_abb + " wrzucone")
     
        
     # dataset_final = pd.concat(playing_stat, sort=False)
#     for data_set in dataset_names_mew:
#         playing_stat[data_set]["Sezon"] = int(data_set)
#     
#     dataset_final = pd.concat(playing_stat, sort=False)
     #temp = dataset_final.loc[dataset_final['MW']==1]
   
     #elo_teams =[]
     #for i in range(0, len(temp)+1, number_of_MW):
     #        date = str(temp["Date"].iloc[i].strftime("%Y-%m-%d"))
     #        sezon = temp["Sezon"].iloc[i]
     #        elo = pd.read_csv('http://api.clubelo.com/' + date)
     #        elo_new_teams = elo.loc[elo["Country"] == "NED"]["Club"].unique().tolist()   
     #        elo_teams.extend(elo_new_teams)
     
     #dataset_final["HomeTeam"].unique().tolist()
     #elo_teams.unique().tolist()
     #sorted(list(set(elo_teams)))
     
     #elo = pd.read_csv('http://api.clubelo.com/2019-06-01')
     #elo = pd.read_csv('http://api.clubelo.com/Arsenal')
     #new_fix = pd.read_csv('http://api.clubelo.com/Fixtures')
     
    cols_final = ["Date", "HomeTeam", "AwayTeam", "FTR", "HTGS", "ATGS", "HTGC", 
                   "ATGC", "HTP", "ATP",  "HT_Wins", "AT_Wins", "HT_Loss", 
                   "AT_Loss", "HT_Draws", "AT_Draws", "MW", "HomeTeamLP", "AwayTeamLP", "Sezon",
                   "HT_LP", "AT_LP", "HTFormPts", "ATFormPts", "HTWinStreak5", 
                   'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3',
                   'HTLossStreak5', 'HTDrawStreak3', 'HTDrawStreak5', 'ATWinStreak3',
                   'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5', 'ATDrawStreak3',
                   'ATDrawStreak5', 'HTGD', 'ATGD','DiffPts', 'DiffFormPts', 'DiffLP',
                   'H2H_Home_pts', 'H2H_Away_pts', 'Off_A', 'Deff_H', 'Deff_A', 'Mean_home_goals', 'Mean_away_goals',
                   'Elo_HT', 'Elo_AT']
     
    final_df = final[cols_final]
     
    cols_to_diff = [ "HTGS", "ATGS", "HTGC", 
                   "ATGC", "HTP", "ATP",  "HT_Wins", "AT_Wins", "HT_Loss", 
                   "AT_Loss", "HT_Draws", "AT_Draws", 'HTGD', 'ATGD']
     
     
    for col in cols_to_diff:
        final_df[col] =  final_df[col] / final_df["MW"]
     
    marketv = pd.read_csv(loc + 'budget_'+ league_abb +'_final.csv', index_col = 0,  encoding='latin-1')
    
    marketv.Sezon = marketv.Sezon.astype(str).str[2:]
    marketv["Sezon"] = marketv["Sezon"].astype(int) + 1
    
    dataset = pd.merge(final_df, marketv,  how='inner', left_on=['HomeTeam','Sezon'], 
                           right_on = ['Club','Sezon'], sort=True)
    
    dataset = dataset.rename(columns={'Age': 'Age_H', 'Foreign': 'Foreign_H',
                                 'Total_value': 'Total_value_H', 'Market_value': 'Market_value_H'})
    
    dataset = pd.merge(dataset, marketv,  how='inner', left_on=['AwayTeam','Sezon'], 
                           right_on = ['Club','Sezon'], sort=False)
    
    dataset = dataset.rename(columns={'Age': 'Age_A', 'Foreign': 'Foreign_A',
                                    'Total_value': 'Total_value_A', 'Market_value': 'Market_value_A'})
        
    df_final = dataset
    df_final = df_final.drop_duplicates()
     
    #to_predict = df_final.loc[df_final['Sezon']==20]
    #training_dataset = df_final.loc[df_final['Sezon']!=20]
     
    to_predict = df_final.loc[(df_final['Sezon']==20) & (df_final['MW']==int(max(playing_stat['20']['MW']))+1)]
    training_dataset =  df_final.loc[~((df_final['Sezon']==20) & (df_final['MW']==int(max(playing_stat['20']['MW'])))+1)]
     
    

  
         
    to_predict.to_csv(loc + "predict_"+league_abb+".csv", index=False)
    training_dataset.to_csv(loc + "training_dataset_"+league_abb+".csv", index=False)
    
     


PrepareDataset('E1','england/championship/',  data_uk = True )
abb = ['E0', 'E1', 'D1', 'D2', 'I1', 'I2', 'SP1','SP2','F1', 'F2', 'N1', 'P1']

league_abb = 'E1'
#


league = 'germany/2-bundesliga'
league_url = 'germany/2_bundesliga/'
league_url = 'england/premier_league/'
league_url = 'england/championship/'
league_url = 'germany/1_bundesliga/'
league_url = 'italy/serie_a/'
league_url = 'italy/serie_b_italy_football/'
league_url = 'france/ligue_1/'
league_url = 'france/ligue_2/'
league_url = 'netherlands/eredivisie/'
league_url = 'portugal/primeira_liga/'

PrepareDataset('D1','germany/1_bundesliga/',  data_uk = True )
PrepareDataset('D2','germany/2-bundesliga',  data_uk = True )
PrepareDataset('I1','italy/serie_a/',  data_uk = True )
PrepareDataset('I2','italy/serie_b/',  data_uk = True )
PrepareDataset('F1','france/ligue_1/',  data_uk = True )
PrepareDataset('F2','france/ligue_2/',  data_uk = True )
PrepareDataset('N1','netherlands/eredivisie/',  data_uk = True )
PrepareDataset('P1','portugal/primeira_liga/',  data_uk = True )
