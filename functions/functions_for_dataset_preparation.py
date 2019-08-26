# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:45:43 2019

@author: pszydlik
"""
import numpy as np
import pandas as pd




def fill_with_mw(dataframe, number_of_MW):
    
    dataframe['MW'] = 0   
    j = 1
    for i in range(0, len(dataframe)):
        dataframe['MW'].iloc[i] = j
        
        if ((i + 1)% number_of_MW) == 0:
            j = j + 1


#Suma goli w sezonie dla kazdej z druzyn oraz numer kolejki
#bramki
def get_goals_scored(playing_stat, number_of_MW):
    # Dictonary z nazwami druzyn ktore beda kluczem
    teams = {}
    #groupby grupuje po druzynach, potem mean() albo cokolwiek np. sum()
    #daje to ramke danych z z druzyna jako pierwsza kolumna i srednia wszystkich pozostalych
    # T - transpozycja macierzy i wyciagniecie nazw kolumn czyli druzyn
    for i in list(set((playing_stat.HomeTeam.tolist() + playing_stat.AwayTeam.tolist()))):
        teams[i] = []
    
    # wartosc odnoszaca sie do klucza (druzyny) jest lista zawierajaca lokalziacje dom/wyjazd
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        # playing_stat.iloc[i].HomeTeam to klucz w slowniku - teams
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)
    
    # Tworzenie dataframe ze strzelonymi golami gdzie wiersze to druzyny a kolumny to numer kolejki
  
    if len(playing_stat)%number_of_MW == 0 and (not playing_stat.isnull().values.any()):
        GoalsScored = pd.DataFrame(data=teams, index = [i for i in range(1,max(playing_stat['MW'])+1)]).T
    else:
        GoalsScored = pd.DataFrame.from_dict(teams,orient='index')
    
    GoalsScored = GoalsScored.fillna(0)    
    GoalsScored[0] = 0
    GoalsScored = GoalsScored.reindex(sorted(GoalsScored.columns), axis=1)
    
    for i in range(2,max(playing_stat['MW'])):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
    return GoalsScored

def get_goals_conceded(playing_stat, number_of_MW):
    teams = {}
    
    for i in list(set((playing_stat.HomeTeam.tolist() + playing_stat.AwayTeam.tolist()))):
        teams[i] = []
        
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
        
    if len(playing_stat)%number_of_MW == 0 and (not playing_stat.isnull().values.any()):
        GoalsConceded = pd.DataFrame(data=teams, index = [i for i in range(1,max(playing_stat['MW'])+1)]).T
    else:
        GoalsConceded = pd.DataFrame.from_dict(teams,orient='index')
    
    GoalsConceded = GoalsConceded.fillna(0)
    GoalsConceded[0] = 0
    GoalsConceded = GoalsConceded.reindex(sorted(GoalsConceded.columns), axis=1)
    for i in range(2,max(playing_stat['MW'])):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
   

    return GoalsConceded


def get_gss(playing_stat, number_of_MW):
    GC = get_goals_conceded(playing_stat, number_of_MW)
    GS = get_goals_scored(playing_stat, number_of_MW)
   
    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []
    
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        #at i ht to nazwy druzyn , nazwy wierszy
        if pd.isna(GS.loc[ht][j]):
            HTGS.append(GS.loc[ht][j-1])
        else:
            HTGS.append(GS.loc[ht][j])
            
        if pd.isna(GS.loc[at][j]):
            ATGS.append(GS.loc[at][j-1])
        else:
            ATGS.append(GS.loc[at][j])
            
        if pd.isna(GC.loc[ht][j]):
            HTGC.append(GC.loc[ht][j-1])
        else:
            HTGC.append(GC.loc[ht][j])
            
        if pd.isna(GC.loc[at][j]):
            ATGC.append(GC.loc[at][j-1])
        else:
            ATGC.append(GC.loc[at][j])
            
        #jedna kolejka to 10 spotkan wiec co 10 kolejek iterator j przesuwa sie +1
        if  i+1 < len(playing_stat):
            if (playing_stat['MW'].iloc[i+1] > playing_stat['MW'].iloc[i]):
                j = j + 1
    
        
    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC
    
    return playing_stat

def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    elif result == 'L':
        return 0
    else:
        return np.nan    


def get_losses(result):
    if result == 'L':
        return 1
    elif result == 'D' or  result == 'W':
        return 0
    else:
        return np.nan     

def get_wins(result):
    if result == 'W':
        return 1
    elif result == 'L' or  result == 'D':
        return 0
    else:
        return np.nan 

def get_draws(result):
    if result == 'D':
        return 1
    elif result == 'L' or  result == 'W':
        return 0
    else:
        return np.nan 

def get_cuml_loss(matchres, number_of_MW):
    #Apply a function to a DataFrame that is intended to operate elementwise,
    #i.e. like doing map(func, series) for each series in the DataFrame    
    matchres_losses = matchres.applymap(get_losses)
    
    if matchres[int(len(matchres.T))-1].notnull().sum() == len(matchres):
        num = int(len(matchres.T)+1)
    else:
        num = int(len(matchres.T))

    for i in range(2,num):
        matchres_losses[i] = matchres_losses[i] + matchres_losses[i-1]

    #first_column = {0:[0*i for i in range(20)]}    
    #a = pd.concat([pd.DataFrame(data = first_column , index = matchres_points.index()), matchres_points], axis=1)
    matchres_losses.insert(column = 0, loc = 0, value = [0*i for i in range(number_of_MW*2)])
    return matchres_losses

def get_cuml_wins(matchres, number_of_MW):
    #Apply a function to a DataFrame that is intended to operate elementwise,
    #i.e. like doing map(func, series) for each series in the DataFrame    
    matchres_losses = matchres.applymap(get_wins)
    
    if matchres[int(len(matchres.T))-1].notnull().sum() == len(matchres):
        num = int(len(matchres.T)+1)
    else:
        num = int(len(matchres.T))

    for i in range(2,num):
        matchres_losses[i] = matchres_losses[i] + matchres_losses[i-1]

    #first_column = {0:[0*i for i in range(20)]}    
    #a = pd.concat([pd.DataFrame(data = first_column , index = matchres_points.index()), matchres_points], axis=1)
    matchres_losses.insert(column = 0, loc = 0, value = [0*i for i in range(number_of_MW*2)])
    return matchres_losses

def get_cuml_draws(matchres, number_of_MW):
    #Apply a function to a DataFrame that is intended to operate elementwise,
    #i.e. like doing map(func, series) for each series in the DataFrame    
    matchres_losses = matchres.applymap(get_draws)
    
    if matchres[int(len(matchres.T))-1].notnull().sum() == len(matchres):
        num = int(len(matchres.T)+1)
    else:
        num = int(len(matchres.T))

    for i in range(2,num):
        matchres_losses[i] = matchres_losses[i] + matchres_losses[i-1]

    #first_column = {0:[0*i for i in range(20)]}    
    #a = pd.concat([pd.DataFrame(data = first_column , index = matchres_points.index()), matchres_points], axis=1)
    matchres_losses.insert(column = 0, loc = 0, value = [0*i for i in range(number_of_MW*2)])
    return matchres_losses

def get_cuml_points(matchres, number_of_MW):
    #Apply a function to a DataFrame that is intended to operate elementwise,
    #i.e. like doing map(func, series) for each series in the DataFrame    
    matchres_points = matchres.applymap(get_points)
    
    if matchres[int(len(matchres.T))-1].notnull().sum() == len(matchres):
        num = int(len(matchres.T)+1)
    else:
        num = int(len(matchres.T))

    for i in range(2,num):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
    

    #first_column = {0:[0*i for i in range(20)]}    
    #a = pd.concat([pd.DataFrame(data = first_column , index = matchres_points.index()), matchres_points], axis=1)
    matchres_points.insert(column = 0, loc = 0, value = [0*i for i in range(number_of_MW*2)])
    return matchres_points


def get_matchres(playing_stat, number_of_MW):
    # Create a dictionary with team names as keys
    teams = {}
    for i in list(set((playing_stat.HomeTeam.tolist() + playing_stat.AwayTeam.tolist()))):
        teams[i] = []

    # the value corresponding to keys is a list containing the match result
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')
            
    if len(playing_stat)%number_of_MW == 0 and (not playing_stat.isnull().values.any()):
        return pd.DataFrame(data=teams, index = [i for i in range(1,max(playing_stat['MW'])+1)]).T
    else:
        df = pd.DataFrame(dict([(col_name,pd.Series(values)) for col_name,values in teams.items() ]), 
                             index = [i for i in range(0,max(playing_stat['MW']))]).T
                          
        df.columns =  df.columns.values +1            
        return  df
    
   

def get_agg_points(playing_stat, number_of_MW):
    matchres = get_matchres(playing_stat, number_of_MW)
    cum_pts = get_cuml_points(matchres, number_of_MW)
    HTP = []
    ATP = []
    j = 0
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        if pd.isna(cum_pts.loc[ht].iloc[j]):
            HTP.append(cum_pts.loc[ht].iloc[j-1])
        else:
            HTP.append(cum_pts.loc[ht].iloc[j])
            
        if pd.isna(cum_pts.loc[at].iloc[j]):
            ATP.append(cum_pts.loc[at].iloc[j-1])
        else:
            ATP.append(cum_pts.loc[at].iloc[j]) 

        if  i+1 < len(playing_stat):
            if (playing_stat['MW'].iloc[i+1] > playing_stat['MW'].iloc[i]):
                j = j + 1
            
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat    

def get_agg_losses(playing_stat, number_of_MW):
    matchres = get_matchres(playing_stat, number_of_MW)
    cum_pts = get_cuml_loss(matchres, number_of_MW)
    HT_Loss = []
    AT_Loss = []
    j = 0
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        if pd.isna(cum_pts.loc[ht].iloc[j]):
            HT_Loss.append(cum_pts.loc[ht].iloc[j-1])
        else:
            HT_Loss.append(cum_pts.loc[ht].iloc[j])
            
        if pd.isna(cum_pts.loc[at].iloc[j]):
            AT_Loss.append(cum_pts.loc[at].iloc[j-1])
        else:
            AT_Loss.append(cum_pts.loc[at].iloc[j]) 

        if  i+1 < len(playing_stat):
            if (playing_stat['MW'].iloc[i+1] > playing_stat['MW'].iloc[i]):
                j = j + 1
            
    playing_stat['HT_Loss'] = HT_Loss
    playing_stat['AT_Loss'] = AT_Loss
    return playing_stat    

def get_agg_wins(playing_stat, number_of_MW):
    matchres = get_matchres(playing_stat, number_of_MW)
    cum_pts = get_cuml_wins(matchres, number_of_MW)
    HT_Wins = []
    AT_Wins = []
    j = 0
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        if pd.isna(cum_pts.loc[ht].iloc[j]):
            HT_Wins.append(cum_pts.loc[ht].iloc[j-1])
        else:
            HT_Wins.append(cum_pts.loc[ht].iloc[j])
            
        if pd.isna(cum_pts.loc[at].iloc[j]):
            AT_Wins.append(cum_pts.loc[at].iloc[j-1])
        else:
            AT_Wins.append(cum_pts.loc[at].iloc[j]) 

        if  i+1 < len(playing_stat):
            if (playing_stat['MW'].iloc[i+1] > playing_stat['MW'].iloc[i]):
                j = j + 1
            
    playing_stat['HT_Wins'] = HT_Wins
    playing_stat['AT_Wins'] = AT_Wins
    return playing_stat    

def get_agg_draws(playing_stat, number_of_MW):
    matchres = get_matchres(playing_stat, number_of_MW)
    cum_pts = get_cuml_draws(matchres, number_of_MW)
    HT_Draws = []
    AT_Draws = []
    j = 0
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        if pd.isna(cum_pts.loc[ht].iloc[j]):
            HT_Draws.append(cum_pts.loc[ht].iloc[j-1])
        else:
            HT_Draws.append(cum_pts.loc[ht].iloc[j])
            
        if pd.isna(cum_pts.loc[at].iloc[j]):
            AT_Draws.append(cum_pts.loc[at].iloc[j-1])
        else:
            AT_Draws.append(cum_pts.loc[at].iloc[j]) 

        if  i+1 < len(playing_stat):
            if (playing_stat['MW'].iloc[i+1] > playing_stat['MW'].iloc[i]):
                j = j + 1
            
    playing_stat['HT_Draws'] = HT_Draws
    playing_stat['AT_Draws'] = AT_Draws
    return playing_stat    



def get_form(playing_stat,num, number_of_MW):
    form = get_matchres(playing_stat, number_of_MW)
    form_final = form.copy()
    for i in range(num,max(playing_stat['MW'])+1):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1           
    return form_final

def add_form(playing_stat,num, number_of_MW):
    form = get_form(playing_stat,num, number_of_MW)
    h = ['M' for i in range(num * number_of_MW)]  # since form is not available for n MW (n*10)
    a = ['M' for i in range(num * number_of_MW)]
    
    j = num
    for i in range((num*number_of_MW),len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        
        past = form.loc[ht][j]               # get past n results
        if pd.isna(past):
          h.append(form.loc[ht][j-1][num-1])
        else:
          h.append(past[num-1])                    # 0 index is most recent 
        
        past = form.loc[at][j]               # get past n results
        if pd.isna(past):
          a.append(form.loc[at][j-1][num-1])
        else:
          a.append(past[num-1]) 
 
        
        if  i+1 < len(playing_stat):
            if (playing_stat['MW'].iloc[i+1] > playing_stat['MW'].iloc[i]):
                j = j + 1

    playing_stat['HM' + str(num)] = h                 
    playing_stat['AM' + str(num)] = a

    
    return playing_stat


def add_form_df(playing_statistics, number_of_MW):
    for num_of_mw in range(1, int(np.where(max(playing_statistics["MW"])>=5, 6,
                                           max(playing_statistics["MW"])))):
        playing_statistics = add_form(playing_statistics,num_of_mw, number_of_MW)
    return playing_statistics 
    
    #playing_statistics = add_form(playing_statistics,1, number_of_MW)
    #playing_statistics = add_form(playing_statistics,2, number_of_MW)
    #playing_statistics = add_form(playing_statistics,3, number_of_MW)
    #playing_statistics = add_form(playing_statistics,4, number_of_MW)
    #playing_statistics = add_form(playing_statistics,5, number_of_MW)
       




def get_last(playing_stat, Standings, year):
    HomeTeamLP = []
    AwayTeamLP = []
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HomeTeamLP.append(Standings.loc[ht][year])
        AwayTeamLP.append(Standings.loc[at][year])
    playing_stat['HomeTeamLP'] = HomeTeamLP
    playing_stat['AwayTeamLP'] = AwayTeamLP
    return playing_stat


def standings_live(playing_stat, number_of_MW):
    
    playing_stat["HT_LP"] =  0
    playing_stat["AT_LP"] =  0
    

    for kolejka in range(1,max(playing_stat['MW'])+1):
        dfa = playing_stat.loc[playing_stat["MW"] == kolejka, ["HomeTeam",  "HTP", "HTGS", "HTGC"]]
        dfb = playing_stat.loc[playing_stat["MW"] == kolejka, ["AwayTeam",  "ATP", "ATGS", "ATGC"]]
        dfa.rename(columns={'HomeTeam': 'Team',  'HTP': 'TP', 'HTGS': 'GS', 'HTGC' : 'GC'}, inplace=True)
        dfb.rename(columns={'AwayTeam': 'Team', 'ATP': 'TP', 'ATGS': 'GS', 'ATGC' : 'GC'}, inplace=True)
        df = pd.concat([dfa, dfb])
        
        
        if (len(df) != number_of_MW * 2):
            if (len(df) < number_of_MW * 2):
                fill_dfa = playing_stat.loc[playing_stat["MW"] == kolejka-1, ["HomeTeam",  "HTP", "HTGS", "HTGC"]]
                fill_dfb = playing_stat.loc[playing_stat["MW"] == kolejka-1, ["AwayTeam",  "ATP", "ATGS", "ATGC"]]
                fill_dfa.rename(columns={'HomeTeam': 'Team',  'HTP': 'TP', 'HTGS': 'GS', 'HTGC' : 'GC'}, inplace=True)
                fill_dfb.rename(columns={'AwayTeam': 'Team', 'ATP': 'TP', 'ATGS': 'GS', 'ATGC' : 'GC'}, inplace=True)
                fill_df = pd.concat([fill_dfa, fill_dfb])
                teams = set(fill_df.Team.values) - set(df.Team.values)
                add = fill_df[fill_df.Team.isin(list(teams))].drop_duplicates(subset=['Team'], keep='last')
                df = pd.concat([df, add])
                                
            elif (len(df) > number_of_MW * 2):
                fill_dfa = playing_stat.loc[playing_stat["MW"] == kolejka, ["HomeTeam",  "HTP", "HTGS", "HTGC"]]
                fill_dfb = playing_stat.loc[playing_stat["MW"] == kolejka, ["AwayTeam",  "ATP", "ATGS", "ATGC"]]
                fill_dfa.rename(columns={'HomeTeam': 'Team',  'HTP': 'TP', 'HTGS': 'GS', 'HTGC' : 'GC'}, inplace=True)
                fill_dfb.rename(columns={'AwayTeam': 'Team', 'ATP': 'TP', 'ATGS': 'GS', 'ATGC' : 'GC'}, inplace=True)
                fill_df = pd.concat([fill_dfa, fill_dfb])
                df = fill_df = fill_df.drop_duplicates(subset=['Team'], keep='last')
        
        
        df["GD"] = df["GS"]  - df["GC"]
        df = df.sort_values(by=['TP', 'GD', 'GS'], ascending=False)  
        df = df.drop_duplicates(subset=None, keep='first', inplace=False)
        df["Position"] = list(range(1, len(df)+1))
        
        idx = playing_stat.index[playing_stat['MW'] == kolejka].tolist()
            
        for j in idx:
            playing_stat.iloc[j, playing_stat.columns.get_loc("HT_LP")] = int(df.loc[df["Team"] == playing_stat.loc[j, 'HomeTeam']].Position)
            playing_stat.iloc[j, playing_stat.columns.get_loc('AT_LP')]  = int(df.loc[df["Team"] == playing_stat.loc[j,'AwayTeam']].Position)
        
    if len(playing_stat)%number_of_MW != 0:
         idx = playing_stat.index[playing_stat['MW'] == int(max(playing_stat['MW'])) ].tolist()
        
         for j in idx:
            playing_stat.iloc[j, playing_stat.columns.get_loc("HT_LP")] = playing_stat.loc[playing_stat["HomeTeam"] == playing_stat.loc[j, 'HomeTeam']]["HT_LP"].iloc[-2]
            playing_stat.iloc[j, playing_stat.columns.get_loc('AT_LP')] = playing_stat.loc[playing_stat["AwayTeam"] == playing_stat.loc[j, 'AwayTeam']]["AT_LP"].iloc[-2]
    
        
    
    return playing_stat


#Offensive and defensive strength based on average goals in league
#def off_and_deff(playing_stat):
#    
#    playing_stat["Off_H"] =  0
#    playing_stat["Off_A"] =  0
#    playing_stat["Deff_H"] =  0
#    playing_stat["Deff_A"] =  0
#       
#    for sezon in range(int(min(playing_stat['Sezon'])), int(max(playing_stat['Sezon']))+1):
#        
#        for kolejka in range(1,int(max(playing_stat['MW']))+1):
#            
#            df_old = playing_stat.loc[(playing_stat["MW"] < kolejka) & (playing_stat["Sezon"] == sezon), ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
#            df_new = playing_stat.loc[(playing_stat["MW"] == kolejka) & (playing_stat["Sezon"] == sezon), ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
#            HG = df_old.FTHG.mean()
#            AG = df_old.FTAG.mean()
#            
#            if (len(df_old) == 0) & (sezon == int(min(playing_stat['Sezon']))):
#                playing_stat.loc[(playing_stat["MW"] == kolejka) & (playing_stat["Sezon"] == sezon), ["Off_H","Off_A" ,"Deff_H", "Deff_A" ]] = 0
#                
#            
#            elif (len(df_old) == 0) & (sezon != int(min(playing_stat['Sezon']))):
#                df_old = playing_stat.loc[playing_stat["Sezon"] == sezon, ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
#                df_new = playing_stat.loc[(playing_stat["MW"] == kolejka) & (playing_stat["Sezon"] == sezon), ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
#                HG = df_old.FTHG.mean()
#                AG = df_old.FTAG.mean()
#                
#                for i in range(0, len(df_new)):
#                
#                    playing_stat.loc[(playing_stat["Sezon"] == sezon) & 
#                               (playing_stat["MW"] == kolejka)  & 
#                               (playing_stat["HomeTeam"] == df_new.iloc[i].HomeTeam), ["Off_H"] ] = df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTHG.mean() / HG
#              
#                    playing_stat.loc[(playing_stat["Sezon"] == sezon) & 
#                               (playing_stat["MW"] == kolejka)  & 
#                               (playing_stat["AwayTeam"] == df_new.iloc[i].AwayTeam), ["Off_A"] ] =   df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTAG.mean() / AG
#                                     
#                    playing_stat.loc[(playing_stat["Sezon"] == sezon) & 
#                               (playing_stat["MW"] == kolejka)  & 
#                               (playing_stat["HomeTeam"] == df_new.iloc[i].HomeTeam), ["Deff_H"] ] = df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTAG.mean() / AG
#              
#                    playing_stat.loc[(playing_stat["Sezon"] == sezon) & 
#                               (playing_stat["MW"] == kolejka)  & 
#                               (playing_stat["AwayTeam"] == df_new.iloc[i].AwayTeam), ["Deff_A"] ] =   df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTHG.mean() / HG
#                    playing_stat.loc[(playing_stat["MW"] == kolejka)  & 
#                                       (playing_stat["HomeTeam"] == df_new.iloc[i].HomeTeam), ["Deff_H"] ] = float(np.where(pd.isna(df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTAG.mean() / AG), 
#                                             AG,df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTAG.mean() / AG))       
#                    playing_stat.loc[ (playing_stat["MW"] == kolejka)  & 
#                                       (playing_stat["AwayTeam"] == df_new.iloc[i].AwayTeam), ["Deff_A"] ] =  float(np.where(pd.isna(df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTHG.mean() / HG), 
#                                             HG,df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTHG.mean() / HG))  
#            else:        
#                for i in range(0, len(df_new)):
#                    playing_stat.loc[(playing_stat["Sezon"] == sezon) & 
#                               (playing_stat["MW"] == kolejka)  & 
#                               (playing_stat["HomeTeam"] == df_new.iloc[i].HomeTeam), ["Off_H"] ] =  df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTHG.mean() / HG 
#                    playing_stat.loc[(playing_stat["Sezon"] == sezon) & 
#                               (playing_stat["MW"] == kolejka)  & 
#                               (playing_stat["AwayTeam"] == df_new.iloc[i].AwayTeam), ["Off_A"] ] =   df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTAG.mean() / AG       
#                    playing_stat.loc[(playing_stat["Sezon"] == sezon) & 
#                               (playing_stat["MW"] == kolejka)  & 
#                               (playing_stat["HomeTeam"] == df_new.iloc[i].HomeTeam), ["Deff_H"] ] = df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTAG.mean() / AG
#                    playing_stat.loc[(playing_stat["Sezon"] == sezon) & 
#                               (playing_stat["MW"] == kolejka)  & 
#                               (playing_stat["AwayTeam"] == df_new.iloc[i].AwayTeam), ["Deff_A"] ] =   df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTHG.mean() / HG
#

##  mecze head to head ######

def h2h(playing_stat):
    
    playing_stat["H2H_Home"] = ""
    playing_stat["H2H_Away"] = ""

    for i in range(1,len(playing_stat)):
        df = playing_stat.iloc[0:i-1,]
        ht = playing_stat.loc[i,"HomeTeam"]
        at = playing_stat.loc[i,"AwayTeam"]
        dff = df[(((df.HomeTeam == ht) &  (df.AwayTeam == at)) | ((df.HomeTeam == at) &  (df.AwayTeam == ht)))].tail(3)
        if (len(dff) >= 1):
            for j in range(0,len(dff)):
                if ((dff.iloc[j].HomeTeam == ht)  and (dff.iloc[j]["FTR"] == "H")):
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Home')] += "W"
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Away')] += "L"
                elif ((dff.iloc[j].HomeTeam == at)  and (dff.iloc[j]["FTR"] == "H")):
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Home')] += "L"
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Away')] += "W"
                elif ((dff.iloc[j].HomeTeam == ht)  and (dff.iloc[j]["FTR"] == "A")):
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Home')] += "L"
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Away')] += "W"
                elif ((dff.iloc[j].HomeTeam == at)  and (dff.iloc[j]["FTR"] == "A")):
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Home')] += "W"
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Away')] += "L"
                elif ((dff.iloc[j].HomeTeam == ht)  and (dff.iloc[j]["FTR"] == "D")):
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Home')] += "D"
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Away')] += "D"
                elif ((dff.iloc[j].HomeTeam == at)  and (dff.iloc[j]["FTR"] == "D")):
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Home')] += "D"
                    playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Away')] += "D"

        else: 
            playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Home')] = "ND"
            playing_stat.iloc[i, playing_stat.columns.get_loc('H2H_Away')] = "ND"
    return playing_stat

   

def get_points_h2h(playing_stat, col):
    
    playing_stat[col+"_pts"] = 0
    
    for i in range(1,len(playing_stat)):
       if (len(playing_stat.iloc[i][col]) == 1):
           if playing_stat.iloc[i][col][0] =='L':
               playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] += 0
           elif playing_stat.iloc[i][col][0] =='W':
               playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] +=  3
           else:
               playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] +=  1
       elif (len(playing_stat.iloc[i][col]) == 2):
           for j in range(0,2):
               if playing_stat.iloc[i][col][j] =='L':
                   playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] += 0
               elif playing_stat.iloc[i][col][j] =='W':
                   playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] +=  3
               elif playing_stat.iloc[i][col][j] =='D':
                   playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] +=  1
               else:
                    playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] += 0
       elif (len(playing_stat.iloc[i][col]) == 3):
           for j in range(0,3):
              if playing_stat.iloc[i][col][j] =='L':
                  playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] += 0
              elif playing_stat.iloc[i][col][j] =='W':
                  playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] +=  3
              elif playing_stat.iloc[i][col][j] =='D':
                  playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] +=  1
       else:
            playing_stat.iloc[i, playing_stat.columns.get_loc(col+"_pts")] = 0
        
    return playing_stat
               
    # Gets the form points.
def get_form_points(string):
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum




# Identify Win/Loss Streaks if any.
def get_3game_ws(string):
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0
    
def get_5game_ws(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0
    
def get_3game_ls(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0
    
def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0
    
def get_3game_ds(string):
    if string[-3:] == 'DDD':
        return 1
    else:
        return 0
    
def get_5game_ds(string):
    if string == 'DDDDD':
        return 1
    else:
        return 0
    
    
#Offensive and defensive strength based on average goals in league
def off_and_deff(playing_stat):
    
       
    playing_stat["temp"] = 0
    
    for i in range(0, len(playing_stat)):    
        if playing_stat.iloc[i].Sezon == min(playing_stat.Sezon):
            playing_stat["temp"].iloc[i] = int(playing_stat.iloc[i].MW)
        else:
            s =  playing_stat["Sezon"].iloc[i]
            k =  playing_stat["MW"].iloc[i]
            s_1 = playing_stat["Sezon"].iloc[i-1]
            k_1 = playing_stat["MW"].iloc[i-1]
            
            if (s_1 < s) & (k != k_1):
                playing_stat["temp"].iloc[i] = int(playing_stat.iloc[i-1].temp)+1
            elif (s_1 == s) & (k == k_1):
                playing_stat["temp"].iloc[i] = int(playing_stat.iloc[i-1].temp)
            elif (s_1 == s) & (k != k_1):
                playing_stat["temp"].iloc[i] = int(playing_stat.iloc[i-1].temp)+1
    
    playing_stat["Off_H"] =  1
    playing_stat["Off_A"] =  1
    playing_stat["Deff_H"] =  1
    playing_stat["Deff_A"] =  1
    playing_stat["Mean_home_goals"] =  1
    playing_stat["Mean_away_goals"] =  1
    
    for kolejka in range(1,int(max(playing_stat['temp']))+1):
        
        if kolejka < 10:
            df_old = playing_stat.loc[playing_stat["temp"] < kolejka, ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
            df_new = playing_stat.loc[playing_stat["temp"] == kolejka, ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
            HG = df_old.FTHG.mean()
            AG = df_old.FTAG.mean()
            
            if len(df_old) == 0:
                playing_stat.loc[playing_stat["temp"] == kolejka, ["Off_H","Off_A" ,"Deff_H", "Deff_A" ]] = 1
            else:
                 for i in range(0, len(df_new)):
                            playing_stat.loc[(playing_stat["temp"] == kolejka)  & 
                                       (playing_stat["HomeTeam"] == df_new.iloc[i].HomeTeam), ["Off_H"] ] =  float(np.where(pd.isna(df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTHG.mean() / HG), 
                                             HG,df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTHG.mean() / HG) )
                            playing_stat.loc[(playing_stat["MW"] == kolejka)  & 
                                       (playing_stat["AwayTeam"] == df_new.iloc[i].AwayTeam), ["Off_A"] ] = float( np.where(pd.isna(df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTAG.mean() / AG ), 
                                             AG,df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTAG.mean() / AG )    )    
                            playing_stat.loc[(playing_stat["MW"] == kolejka)  & 
                                       (playing_stat["HomeTeam"] == df_new.iloc[i].HomeTeam), ["Deff_H"] ] = float(np.where(pd.isna(df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTAG.mean() / AG), 
                                             AG,df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTAG.mean() / AG))       
                            playing_stat.loc[ (playing_stat["MW"] == kolejka)  & 
                                       (playing_stat["AwayTeam"] == df_new.iloc[i].AwayTeam), ["Deff_A"] ] =  float(np.where(pd.isna(df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTHG.mean() / HG), 
                                             HG,df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTHG.mean() / HG)) 
                            
            
            indx = playing_stat.index[playing_stat['temp'] == kolejka].tolist()
                            
            for j in indx:
                playing_stat.iloc[j, playing_stat.columns.get_loc("Mean_home_goals")]  = playing_stat.iloc[j]["Off_H"] * playing_stat.iloc[j]["Deff_A"] * HG
                playing_stat.iloc[j, playing_stat.columns.get_loc("Mean_away_goals")]  = playing_stat.iloc[j]["Off_A"] * playing_stat.iloc[j]["Deff_H"] * AG
                            
        elif kolejka >= 10:
            df_old = playing_stat.loc[(playing_stat["temp"] < kolejka) & (playing_stat["temp"] >= kolejka-10), ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
            df_new = playing_stat.loc[playing_stat["temp"] == kolejka, ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
            HG = df_old.FTHG.mean()
            AG = df_old.FTAG.mean()
            
            if len(df_old) == 0:
                playing_stat.loc[playing_stat["temp"] == kolejka, ["Off_H","Off_A" ,"Deff_H", "Deff_A" ]] = 1
            else:
                 for i in range(0, len(df_new)):
                            playing_stat.loc[(playing_stat["temp"] == kolejka)  & 
                                       (playing_stat["HomeTeam"] == df_new.iloc[i].HomeTeam), ["Off_H"] ] =  float(np.where(pd.isna(df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTHG.mean() / HG), 
                                             HG,df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTHG.mean() / HG) )
                            playing_stat.loc[(playing_stat["MW"] == kolejka)  & 
                                       (playing_stat["AwayTeam"] == df_new.iloc[i].AwayTeam), ["Off_A"] ] = float( np.where(pd.isna(df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTAG.mean() / AG ), 
                                             AG,df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTAG.mean() / AG )    )    
                            playing_stat.loc[(playing_stat["MW"] == kolejka)  & 
                                       (playing_stat["HomeTeam"] == df_new.iloc[i].HomeTeam), ["Deff_H"] ] = float(np.where(pd.isna(df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTAG.mean() / AG), 
                                             AG,df_old.loc[df_old["HomeTeam"] == df_new.iloc[i].HomeTeam].FTAG.mean() / AG))       
                            playing_stat.loc[ (playing_stat["MW"] == kolejka)  & 
                                       (playing_stat["AwayTeam"] == df_new.iloc[i].AwayTeam), ["Deff_A"] ] =  float(np.where(pd.isna(df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTHG.mean() / HG), 
                                             HG,df_old.loc[df_old["AwayTeam"] == df_new.iloc[i].AwayTeam].FTHG.mean() / HG))                                         
                       
            indx = playing_stat.index[playing_stat['temp'] == kolejka].tolist()
                                
            for j in indx:
                playing_stat.iloc[j, playing_stat.columns.get_loc("Mean_home_goals")]  = playing_stat.iloc[j]["Off_H"] * playing_stat.iloc[j]["Deff_A"] * HG
                playing_stat.iloc[j, playing_stat.columns.get_loc("Mean_away_goals")]  = playing_stat.iloc[j]["Off_A"] * playing_stat.iloc[j]["Deff_H"] * AG
                                                               
        
    
            