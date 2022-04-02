# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:57:54 2022

@author: tomas
"""

import pandas as pd

dft = pd.read_csv('E0_2021.csv')
r30 = pd.read_excel('rodada31.xlsx')

def remove_games(dft):
    x1 = dft.iloc[165:176]
    x2 = dft.iloc[183:201]
    x3 = dft.iloc[224:234]
    x4 = dft.iloc[245:247]
    x5 = dft.iloc[267:270]
    x6 = dft.iloc[248:257]
    x7 = dft.iloc[282:283]
    
    x = x1.append(x2)
    x = x.append(x3)
    x = x.append(x4)
    x = x.append(x5)
    x = x.append(x6)
    x = x.append(x7)
    
    dff = dft.append(x)
    dff = dff.drop_duplicates(keep = False)
    return dff

dff = remove_games(dft)
#%%
r30 = r30[['TimeCasa','TimeFora']]
r30 = r30.rename(columns={"TimeCasa": "HomeTeam", "TimeFora": "AwayTeam"}, errors="raise")
col_list = dff.columns.tolist()
col_list_2 = dff.columns.tolist()
col_list.remove('HomeTeam')
col_list.remove('AwayTeam')

for i in col_list:
    r30[i] = pd.NaT 
    
r30 = r30[col_list_2]
r30['FTR'] = r30['FTR'].fillna('D')
r30 = r30.fillna(0)

dff = dff.append(r30)

#%%
l = dff['HomeTeam'].tolist()

ll = dff['AwayTeam'].tolist()

lll = l + ll


dfs = pd.DataFrame(lll, columns= ['Teams'])

print(dfs['Teams'].value_counts())
#%%

# Gols marcados agredados
def get_goals_scored(playing_stat):
    # Cria um dict contendo todos os times
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
    
    # Atribui uma lista com os gols que o time fez em cada rodada
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)
    
    #transforma em df e altera o range de 0 - 37 para 1 - 38
    GoalsScored = pd.DataFrame(data=teams, index = [i for i in range(1,25)]).T
    GoalsScored[0] = 0
    
    # Aggregate to get uptil that point
    for i in range(2,25):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
    return GoalsScored
#%%
###############################################################################
# Cria um dict contendo todos os times
teams = {}
for i in dff.groupby('HomeTeam').mean().T.columns:
    teams[i] = []
    
# Atribui uma lista com os gols que o time fez em cada rodada
for i in range(len(dff)):
    HTGS = dff.iloc[i]['FTHG']
    ATGS = dff.iloc[i]['FTAG']
    teams[dff.iloc[i].HomeTeam].append(HTGS)
    teams[dff.iloc[i].AwayTeam].append(ATGS)
    
    #transforma em df e altera o range de 0 - 37 para 1 - 38
GoalsScored = pd.DataFrame(data=teams, index = [i for i in range(1,25)]).T
GoalsScored[0] = 0
    
    # Aggregate to get uptil that point
for i in range(2,25):
    GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
    
    
##################################################################################
#%%
# Lógica muito parecida par os gols sofridos a cada rodada

def get_goals_conceded(playing_stat):
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)
    
    
    GoalsConceded = pd.DataFrame(data=teams, index = [i for i in range(1,25)]).T
    GoalsConceded[0] = 0

    for i in range(2,25):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
    return GoalsConceded

#%%
def get_sg(playing_stat):
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)
   
    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(240):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        
#Check de quando se alcança o 10 jogo, isso é divide o valor de i+1 por 10
#até ter um resto = 0 
        if ((i + 1)% 10) == 0:
            j = j + 1

        
    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC
    
    return playing_stat

df = get_sg(dff)

#%%
def get_points(result):
    #Esse é bem intuitivo galera
    
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0
#%%  

def get_cuml_points(matchres):
    matchres_points = matchres.applymap(get_points)
    for i in range(2,25):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
        
    matchres_points.insert(column =0, loc = 0, value = [0*i for i in range(20)])
    return matchres_points


#%%
def get_matchres(playing_stat):
    # Create a dictionary with team names as keys
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
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
            
    return pd.DataFrame(data=teams, index = [i for i in range(1,25)]).T

#%%
def get_agg_points(playing_stat):

    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = []
    ATP = []
    j = 0
    for i in range(240):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1)% 10) == 0:
            j = j + 1
            
    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat

df = get_agg_points(df)
#%%
df = df[['HomeTeam','AwayTeam','FTHG','FTAG','FTR', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP',]]

df.to_excel('final_dataset2022.xlsx', index = False)

