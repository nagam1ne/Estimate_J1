def load_data():
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    print 'loading data...'

    """pre-processing"""

    # read data
    cbp2014_df= pd.read_excel('data/2014_J1_CBP.xls')
    result2014_df= pd.read_excel('data/2014_J1_Result.xls')
    scorer2014_df= pd.read_excel('data/2014_J1_Scorer.xls')

    cbp2015_df= pd.read_excel('data/2015_J1_1st-2nd_CBP.xls')
    result2015_df= pd.read_excel('data/2015_J1_1st-2nd_Result.xls')
    scorer2015_df= pd.read_excel('data/2015_J1_1st-2nd_Scorer.xls')

    final_game_df = pd.read_excel('data/2015_J1_2nd_final_game.xls')

    pre_final_homeID = np.array([127,294,136,120,122,132,269,128,193])
    pre_final_awayID = np.array([150,126,30528,124,86,270,238,129,130])

    final_homeID = final_game_df[u'ホームチームID'].as_matrix()
    final_awayID = final_game_df[u'アウェイチームID'].as_matrix()

    remv_IDs = zip(np.r_[pre_final_homeID,final_homeID],np.r_[pre_final_awayID,final_awayID])

    # marge 2014's and 2015's
    cbp_df = pd.concat([cbp2014_df, cbp2015_df], ignore_index=True)
    result_df = pd.concat([result2014_df, result2015_df], ignore_index=True)

    # split to home_data and  away_data
    home_cbp_dfs = [cbp_df.loc[cbp_df[u'試合ID']==gameID] [(cbp_df['HA']=='H')] for gameID in result_df[u'試合ID']]
    away_cbp_dfs= [cbp_df.loc[cbp_df[u'試合ID']==gameID] [(cbp_df['HA']=='A')] for gameID in result_df[u'試合ID']]
    home_result = result_df [u'ホーム得点'].as_matrix()
    away_result = result_df [u'アウェイ得点'].as_matrix()

    # collect teamID
    teamID_2014 = np.r_[result2014_df[result2014_df[u'節']==1][u'ホームチームID'].as_matrix(),
                        result2014_df[result2014_df[u'節']==1][u'アウェイチームID'].as_matrix()]
    teamID_2015 = np.r_[result2015_df[result2015_df[u'節']==1][u'ホームチームID'][:9].as_matrix(),
                        result2015_df[result2015_df[u'節']==1][u'アウェイチームID'][:9].as_matrix()]

    # splite to homeID and awayID
    homeID_2014= result2014_df[u'ホームチームID'].as_matrix()
    awayID_2014= result2014_df[u'アウェイチームID'].as_matrix()
    homeID_2015= result2015_df[u'ホームチームID'].as_matrix()
    awayID_2015= result2015_df[u'アウェイチームID'].as_matrix()

    # search index each home and away
    home_2014_nb= [np.where(homeID_2014==ID)[0]  for ID in teamID_2014]
    away_2014_nb = [np.where(awayID_2014==ID)[0] for ID in teamID_2014]
    home_2015_nb= [np.where(homeID_2015==ID)[0] + np.array(home_2014_nb).size for ID in teamID_2015]
    away_2015_nb = [np.where(awayID_2015==ID)[0] + np.array(away_2014_nb).size for ID in teamID_2015]


    """
    create starting member's CBP dictionary

    cbp_2014_ave_starters_dict
    cbp_2015_ave_starters_dict
    cbp_2014_real_starters_dict
    cbp_2015_real_starters_dict
    """

    import multiprocessing as mp

    # rename ['CB','CH','OH','CF']
    def rename_pos(df):
        positions = ['CB','CH','OH','CF']
        for pos in positions:
            temp_df=df[df[u'先発\n選手\n位置']==pos]
            if temp_df.index.shape[0] == 1:
                df[u'先発\n選手\n位置'][temp_df.index[0]] = '%s1'%pos
            elif temp_df.index.shape[0] == 2:
                df[u'先発\n選手\n位置'][temp_df.index[0]] = '%s0'%pos
                df[u'先発\n選手\n位置'][temp_df.index[1]] = '%s2'%pos
            elif temp_df.index.shape[0] == 3:
                df[u'先発\n選手\n位置'][temp_df.index[0]] = '%s0'%pos
                df[u'先発\n選手\n位置'][temp_df.index[1]] = '%s1'%pos
                df[u'先発\n選手\n位置'][temp_df.index[2]] = '%s2'%pos
            elif temp_df.index.shape[0] >= 4:
                print 'home_cbp_df_pos.shape[0] >= 4:'
        return df

    #  multiprosessing
    pool = mp.Pool()

    home_cbp_dfs = pool.map(rename_pos, home_cbp_dfs)
    away_cbp_dfs = pool.map(rename_pos, away_cbp_dfs)

    #  create starting member list

    starters_left = []

    starters_left.append(['CF0','LSH','OH0','Non'])
    starters_left.append(['CF0','OH0','CF1','OH1'])
    starters_left.append(['CF1','OH1','CF2','OH2'])
    starters_left.append(['CF2','RSH','OH2','Non'])

    starters_left.append(['LSH','LWB','CH0','OH0'])
    starters_left.append(['CH0','OH0','CH1','OH1'])
    starters_left.append(['CH1','OH1','CH2','OH2'])
    starters_left.append(['RSH','RWB','CH2','OH2'])

    starters_left.append(['LSB','LWB','CH0','CB0'])
    starters_left.append(['CH0','CB0','CH1','CB1'])
    starters_left.append(['CH1','CB1','CH2','CB2'])
    starters_left.append(['RSB','RWB','CH2','CB2'])

    starters_right = []

    starters_right.append(['CF2','RSH','OH2','Non'])
    starters_right.append(['CF1','OH1','CF2','OH2'])
    starters_right.append(['CF0','OH0','CF1','OH1'])
    starters_right.append(['CF0','LSH','OH0','Non'])

    starters_right.append(['RSH','RWB','CH2','OH2'])
    starters_right.append(['CH1','OH1','CH2','OH2'])
    starters_right.append(['CH0','OH0','CH1','OH1'])
    starters_right.append(['LSH','LWB','CH0','OH0'])

    starters_right.append(['RSB','RWB','CH2','CB2'])
    starters_right.append(['CH1','CB1','CH2','CB2'])
    starters_right.append(['CH0','CB0','CH1','CB1'])
    starters_right.append(['LSB','LWB','CH0','CB0'])


    # sum pooling 
    home_starters_dfs_left = []
    for df in home_cbp_dfs:
        df = df.rename(columns={u'先発\n選手\n位置':'starters'})
        home_starters_dfs_left.append([df.query('starters in @starter').loc[:,[u'出場',u'実出場\n時間',u'シュート\nCBP',u'パス\nCBP',u'クロス\nCBP',u'ドリブル\nCBP',
                                                                                                                        u'攻撃CBP',u'守備CBP']].sum() for starter in starters_left])
    away_starters_dfs_left = []
    for df in away_cbp_dfs:
        df = df.rename(columns={u'先発\n選手\n位置':'starters'})
        away_starters_dfs_left.append([df.query('starters in @starter').loc[:,[u'出場',u'実出場\n時間',u'シュート\nCBP',u'パス\nCBP',u'クロス\nCBP',u'ドリブル\nCBP',
                                                                                                                        u'攻撃CBP',u'守備CBP']].sum() for starter in starters_left])
    home_starters_dfs_right = []
    for df in home_cbp_dfs:
        df = df.rename(columns={u'先発\n選手\n位置':'starters'})
        home_starters_dfs_right.append([df.query('starters in @starter').loc[:,[u'出場',u'実出場\n時間',u'シュート\nCBP',u'パス\nCBP',u'クロス\nCBP',u'ドリブル\nCBP',
                                                                                                                        u'攻撃CBP',u'守備CBP']].sum() for starter in starters_right])
    away_starters_dfs_right = []
    for df in away_cbp_dfs:
        df = df.rename(columns={u'先発\n選手\n位置':'starters'})
        away_starters_dfs_right.append([df.query('starters in @starter').loc[:,[u'出場',u'実出場\n時間',u'シュート\nCBP',u'パス\nCBP',u'クロス\nCBP',u'ドリブル\nCBP',
                                                                                                                        u'攻撃CBP',u'守備CBP']].sum() for starter in starters_right])
    #  change dataframe to vector and normaraize
    temp_dfs = []
    for dfs in home_starters_dfs_left:
        temp_df = np.array([df.as_matrix().reshape(df.size) for df in dfs]) # change matrix to vector
        temp_dfs.append(temp_df.reshape(temp_df.size))
    home_cbp_starters_left= np.array(temp_dfs)
    home_cbp_starters_left =np.nan_to_num(np.array([(sample - np.mean(sample))/np.std(sample) for sample in home_cbp_starters_left.T])).T

    temp_dfs = []
    for dfs in away_starters_dfs_left:
        temp_df = np.array([df.as_matrix().reshape(df.size) for df in dfs]) # change matrix to vector
        temp_dfs.append(temp_df.reshape(temp_df.size))
    away_cbp_starters_left= np.array(temp_dfs)
    away_cbp_starters_left =np.nan_to_num(np.array([(sample - np.mean(sample))/np.std(sample) for sample in away_cbp_starters_left.T])).T

    #  divide with each teams
    home_cbp_starters_team_2014_left =[home_cbp_starters_left[n] for n in home_2014_nb]
    away_cbp_starters_team_2014_left =[away_cbp_starters_left[n] for n in away_2014_nb]
    home_cbp_starters_team_2015_left =[home_cbp_starters_left[n] for n in home_2015_nb]
    away_cbp_starters_team_2015_left =[away_cbp_starters_left[n] for n in away_2015_nb]

    # create dictionary of cbp each team{teamID:cbp data}
    cbp_starters_team_2014_left = [np.r_[home,away].T for home, away in zip(home_cbp_starters_team_2014_left, away_cbp_starters_team_2014_left)]
    cbp_starters_team_2014_left = np.array([[np.average(row) for row in team] for team in cbp_starters_team_2014_left])
    cbp_2014_ave_starters_dict_left = dict(zip(teamID_2014, cbp_starters_team_2014_left))

    cbp_starters_team_2015_left = [np.r_[home,away].T for home, away in zip(home_cbp_starters_team_2015_left, away_cbp_starters_team_2015_left)]
    cbp_starters_team_2015_left = np.array([[np.average(row) for row in team] for team in cbp_starters_team_2015_left])
    cbp_2015_ave_starters_dict_left = dict(zip(teamID_2015, cbp_starters_team_2015_left))

    #  divide with each home and away
    home_cbp_starters_2014_left = home_cbp_starters_left[:np.array(home_2014_nb).size]
    away_cbp_starters_2014_left = away_cbp_starters_left[:np.array(away_2014_nb).size]

    home_cbp_starters_2015_left = home_cbp_starters_left[np.array(home_2014_nb).size:]
    away_cbp_starters_2015_left = away_cbp_starters_left[np.array(home_2014_nb).size:]


    # create dictionary of cbp each game {homeID, awayID : homecbp, awaycbp}
    cbp_2014_real_starters_dict_left = {(homeID,awayID):[homecbp,awaycbp] for homeID,awayID,homecbp,awaycbp in zip(result2014_df[u'ホームチームID'],
                                                                                            result2014_df[u'アウェイチームID'],home_cbp_starters_2014_left,away_cbp_starters_2014_left)}
    cbp_2015_real_starters_dict_left = {(homeID,awayID):[homecbp,awaycbp] for homeID,awayID,homecbp,awaycbp in zip(result2015_df[u'ホームチームID'],
                                                                                            result2015_df[u'アウェイチームID'],home_cbp_starters_2015_left,away_cbp_starters_2015_left)}

    #  change dataframe to vector and normaraize
    temp_dfs = []
    for dfs in home_starters_dfs_right:
        temp_df = np.array([df.as_matrix().reshape(df.size) for df in dfs]) # change matrix to vector
        temp_dfs.append(temp_df.reshape(temp_df.size))
    home_cbp_starters_right= np.array(temp_dfs)
    home_cbp_starters_right =np.nan_to_num(np.array([(sample - np.mean(sample))/np.std(sample) for sample in home_cbp_starters_right.T])).T

    temp_dfs = []
    for dfs in away_starters_dfs_right:
        temp_df = np.array([df.as_matrix().reshape(df.size) for df in dfs]) # change matrix to vector
        temp_dfs.append(temp_df.reshape(temp_df.size))
    away_cbp_starters_right= np.array(temp_dfs)
    away_cbp_starters_right =np.nan_to_num(np.array([(sample - np.mean(sample))/np.std(sample) for sample in away_cbp_starters_right.T])).T

    #  divide with each teams
    home_cbp_starters_team_2014_right =[home_cbp_starters_right[n] for n in home_2014_nb]
    away_cbp_starters_team_2014_right =[away_cbp_starters_right[n] for n in away_2014_nb]
    home_cbp_starters_team_2015_right =[home_cbp_starters_right[n] for n in home_2015_nb]
    away_cbp_starters_team_2015_right =[away_cbp_starters_right[n] for n in away_2015_nb]

    # create dictionary of cbp each team{teamID:cbp data}
    cbp_starters_team_2014_right = [np.r_[home,away].T for home, away in zip(home_cbp_starters_team_2014_right, away_cbp_starters_team_2014_right)]
    cbp_starters_team_2014_right = np.array([[np.average(row) for row in team] for team in cbp_starters_team_2014_right])
    cbp_2014_ave_starters_dict_right = dict(zip(teamID_2014, cbp_starters_team_2014_right))

    cbp_starters_team_2015_right = [np.r_[home,away].T for home, away in zip(home_cbp_starters_team_2015_right, away_cbp_starters_team_2015_right)]
    cbp_starters_team_2015_right = np.array([[np.average(row) for row in team] for team in cbp_starters_team_2015_right])
    cbp_2015_ave_starters_dict_right = dict(zip(teamID_2015, cbp_starters_team_2015_right))

    #  divide with each home and away
    home_cbp_starters_2014_right = home_cbp_starters_right[:np.array(home_2014_nb).size]
    away_cbp_starters_2014_right = away_cbp_starters_right[:np.array(away_2014_nb).size]

    home_cbp_starters_2015_right = home_cbp_starters_right[np.array(home_2014_nb).size:]
    away_cbp_starters_2015_right = away_cbp_starters_right[np.array(home_2014_nb).size:]


    # create dictionary of cbp each game {homeID, awayID : homecbp, awaycbp}
    cbp_2014_real_starters_dict_right = {(homeID,awayID):[homecbp,awaycbp] for homeID,awayID,homecbp,awaycbp in zip(result2014_df[u'ホームチームID'],
                                                                                            result2014_df[u'アウェイチームID'],home_cbp_starters_2014_right,away_cbp_starters_2014_right)}
    cbp_2015_real_starters_dict_right = {(homeID,awayID):[homecbp,awaycbp] for homeID,awayID,homecbp,awaycbp in zip(result2015_df[u'ホームチームID'],
                                                                                            result2015_df[u'アウェイチームID'],home_cbp_starters_2015_right,away_cbp_starters_2015_right)}


    """
    create sub member's CBP dictionary

    cbp_2014_ave_sub_dict
    cbp_2015_ave_sub_dict
    cbp_2014_real_sub_dict
    cbp_2015_real_sub_dict
    """

    # create temp_cbp for adjustment
    columns = cbp_df.columns
    temp_cbp = pd.DataFrame({'Pos.':['GK','DF','MF','FW'],u'先発\n選手\n位置':["NaN","NaN","NaN","NaN"]},columns=columns)
    temp_cbp = temp_cbp.fillna(0) # change nan to zero

    # divide each positions
    home_sub_dfs = []
    for df in home_cbp_dfs:
        df = pd.concat([df,temp_cbp], ignore_index=True)
        df = df.fillna('NaN') # np.nan → str(NaN)
        df = df.rename(columns={u'先発\n選手\n位置':'starters'})
        df = df.query('starters in ["GK","NaN"]').groupby(['Pos.']).sum().loc[:,[u'出場',u'実出場\n時間',u'シュート\nCBP',u'パス\nCBP',u'クロス\nCBP',u'ドリブル\nCBP',
                                                                                                                        u'攻撃CBP',u'守備CBP',u'セーブCBP']]
        home_sub_dfs.append(df)

    away_sub_dfs = []
    for df in away_cbp_dfs:
        df = pd.concat([df,temp_cbp], ignore_index=True)
        df = df.fillna('NaN') # np.nan → str(NaN)
        df = df.rename(columns={u'先発\n選手\n位置':'starters'})
        df = df.query('starters in ["GK","NaN"]').groupby(['Pos.']).sum().loc[:,[u'出場',u'実出場\n時間',u'シュート\nCBP',u'パス\nCBP',u'クロス\nCBP',u'ドリブル\nCBP',
                                                                                                                        u'攻撃CBP',u'守備CBP',u'セーブCBP']]
        away_sub_dfs.append(df)

    # nomarizing
    home_cbp_sub = np.array([df.as_matrix().reshape(df.size) for df in home_sub_dfs]) # change matrix to vector
    home_cbp_sub =np.nan_to_num(np.array([(sample - np.mean(sample))/np.std(sample) for sample in home_cbp_sub.T])).T

    away_cbp_sub = np.array([df.as_matrix().reshape(df.size) for df in away_sub_dfs]) # change matrix to vector
    away_cbp_sub = np.nan_to_num(np.array([(sample - np.mean(sample))/np.std(sample) for sample in away_cbp_sub.T])).T

    #  divide with each teams
    home_cbp_sub_team_2014 =[home_cbp_sub[n] for n in home_2014_nb]
    away_cbp_sub_team_2014 =[away_cbp_sub[n] for n in away_2014_nb]

    home_cbp_sub_team_2015 =[home_cbp_sub[n] for n in home_2015_nb]
    away_cbp_sub_team_2015 =[away_cbp_sub[n] for n in away_2015_nb]

    # create cbp dictionary [teamID, cbp data]
    cbp_sub_team_2014 = [np.r_[home,away].T for home, away in zip(home_cbp_sub_team_2014, away_cbp_sub_team_2014)]
    cbp_sub_team_2014 = np.array([[np.average(row) for row in team] for team in cbp_sub_team_2014])
    cbp_2014_ave_sub_dict = dict(zip(teamID_2014,cbp_sub_team_2014 ))

    cbp_sub_team_2015 = [np.r_[home,away].T for home, away in zip(home_cbp_sub_team_2015, away_cbp_sub_team_2015)]
    cbp_sub_team_2015 = np.array([[np.average(row) for row in team] for team in cbp_sub_team_2015])
    cbp_2015_ave_sub_dict = dict(zip(teamID_2015, cbp_sub_team_2015 ))

    #  divide with each home and away
    home_cbp_sub_2014 = home_cbp_sub[:np.array(home_2014_nb).size]
    away_cbp_sub_2014 = away_cbp_sub[:np.array(away_2014_nb).size]

    home_cbp_sub_2015 = home_cbp_sub[np.array(home_2014_nb).size:]
    away_cbp_sub_2015 = away_cbp_sub[np.array(away_2014_nb).size:]

    # create dictionary of cbp each game {homeID, awayID : homecbp, awaycbp}
    cbp_2014_real_sub_dict = {(homeID,awayID):[homecbp,awaycbp] for homeID,awayID,homecbp,awaycbp in zip(result2014_df[u'ホームチームID'],
                                                                                            result2014_df[u'アウェイチームID'],home_cbp_sub_2014,away_cbp_sub_2014)}
    cbp_2015_real_sub_dict = {(homeID,awayID):[homecbp,awaycbp] for homeID,awayID,homecbp,awaycbp in zip(result2015_df[u'ホームチームID'],
                                                                                            result2015_df[u'アウェイチームID'],home_cbp_sub_2015,away_cbp_sub_2015)}


    """
    create score dictionary

    real_score_2014_dict
    real_score_2015_dict
    ave_score_2014_dict
    ave_score_2015_dict
    """

    # create action's list
    action = [u'シュート', u'PK',u'直接FK',u'オウンゴール' ]

    # create actions's sample each games
    score_2014 = []
    for nb in range(result2014_df.index.shape[0]):
        game = scorer2014_df[scorer2014_df[u'試合ID']==result2014_df.ix[nb][u'試合ID']]
        home = game[game[u'チームID']==result2014_df.ix[nb][u'ホームチームID']]
        home_pt =[home[home[u'アクション名']==act].count()[u'試合ID'] for act in action]
        away = game[game[u'チームID']==result2014_df.ix[nb][u'アウェイチームID']]
        away_pt =[away[away[u'アクション名']==act].count()[u'試合ID'] for act in action]
        score_2014.append(np.array([home_pt, away_pt]).reshape(8))

    score_2015 = []
    for nb in range(result2015_df.index.shape[0]):
        game = scorer2015_df[scorer2015_df[u'試合ID']==result2015_df.ix[nb][u'試合ID']]
        home = game[game[u'チームID']==result2015_df.ix[nb][u'ホームチームID']]
        home_pt =[home[home[u'アクション名']==act].count()[u'試合ID'] for act in action]
        away = game[game[u'チームID']==result2015_df.ix[nb][u'アウェイチームID']]
        away_pt =[away[away[u'アクション名']==act].count()[u'試合ID'] for act in action]
        score_2015.append(np.array([home_pt, away_pt]).reshape(8))



    #  normarization
    score_2014_norm =  np.nan_to_num(np.array([(sample - sample.mean())/sample.std() for sample in np.array(score_2014).T])).T
    score_2015_norm =  np.nan_to_num(np.array([(sample - sample.mean())/sample.std() for sample in np.array(score_2015).T])).T

    # create dictionary of score each game  {homeID,awayID:score}
    real_score_2014_dict = {(homeID,awayID):score for homeID,awayID,score in zip(result2014_df[u'ホームチームID'],
                                                                                            result2014_df[u'アウェイチームID'],score_2014_norm)}

    real_score_2015_dict = {(homeID,awayID):score for homeID,awayID,score in zip(result2015_df[u'ホームチームID'],
                                                                                            result2015_df[u'アウェイチームID'],score_2015_norm)}


    # divide with home and away
    home_score_team_2014 =np.array([np.array(score_2014_norm)[n] for n in home_2014_nb])
    away_score_team_2014 =np.array([np.array(score_2014_norm)[n] for n in away_2014_nb])

    home_score_team_2015 =np.array([np.array(score_2015_norm)[n-np.array(home_2014_nb).size] for n in home_2015_nb])
    away_score_team_2015 =np.array([np.array(score_2015_norm)[n-np.array(away_2014_nb).size] for n in away_2015_nb])

    # create dictionary of score each team  {teamID : score}
    ave_score_2014 = np.array([np.r_[home,away].mean(axis=0) for home,away in zip(home_score_team_2014,away_score_team_2014)])
    ave_score_2014_dict ={teamID:score for teamID,score in zip(teamID_2014,ave_score_2014)}
    ave_score_2015 = np.array([np.r_[home,away].mean(axis=0) for home,away in zip(home_score_team_2015,away_score_team_2015)])
    ave_score_2015_dict ={teamID:score for teamID,score in zip(teamID_2015,ave_score_2015)}

    """
    create_train_data

    """

    cnt_nb = 0
    remv_index = []
    for homeID, awayID in zip(awayID_2015, homeID_2015):
        if (homeID,awayID) in remv_IDs:
            remv_index.append(cnt_nb)
        cnt_nb += 1

    home_ave_starters_2014_left = np.array([cbp_2014_ave_starters_dict_left[ID] for ID in homeID_2014])
    home_ave_starters_2015_left = np.array([cbp_2015_ave_starters_dict_left[ID] for ID in homeID_2015])
    home_ave_starters_2014_right = np.array([cbp_2014_ave_starters_dict_right[ID] for ID in homeID_2014])
    home_ave_starters_2015_right = np.array([cbp_2015_ave_starters_dict_right[ID] for ID in homeID_2015])

    home_ave_sub_2014 = np.array([cbp_2014_ave_sub_dict[ID] for ID in homeID_2014])
    home_ave_sub_2015 = np.array([cbp_2015_ave_sub_dict[ID] for ID in homeID_2015])

    home_ave_score_2014 = np.array([ave_score_2014_dict[ID] for ID in homeID_2014])
    home_ave_score_2015 = np.array([ave_score_2015_dict[ID] for ID in homeID_2015])

    away_ave_starters_2014_left = np.array([cbp_2014_ave_starters_dict_left[ID] for ID in awayID_2014])
    away_ave_starters_2015_left = np.array([cbp_2015_ave_starters_dict_left[ID] for ID in awayID_2015])
    away_ave_starters_2014_right = np.array([cbp_2014_ave_starters_dict_right[ID] for ID in awayID_2014])
    away_ave_starters_2015_right = np.array([cbp_2015_ave_starters_dict_right[ID] for ID in awayID_2015])

    away_ave_sub_2014 = np.array([cbp_2014_ave_sub_dict[ID] for ID in awayID_2014])
    away_ave_sub_2015 = np.array([cbp_2015_ave_sub_dict[ID] for ID in awayID_2015])

    away_ave_score_2014 = np.array([ave_score_2014_dict[ID] for ID in awayID_2014])
    away_ave_score_2015 = np.array([ave_score_2015_dict[ID] for ID in awayID_2015])

    for ix in remv_index:
        home_ave_starters_2015_left = np.delete(home_ave_starters_2015_left,ix,0)
        home_ave_starters_2015_right = np.delete(home_ave_starters_2015_right,ix,0)
        home_ave_sub_2015 = np.delete(home_ave_sub_2015,ix,0)
        home_ave_score_2015 = np.delete(home_ave_score_2015,ix,0)
        away_ave_starters_2015_left  = np.delete(away_ave_starters_2015_left,ix,0)
        away_ave_starters_2015_right  = np.delete(away_ave_starters_2015_right,ix,0)
        away_ave_sub_2015  = np.delete(away_ave_sub_2015,ix,0)
        away_ave_score_2015  = np.delete(away_ave_score_2015,ix,0)
        
    home_ave_starters_train = np.r_[home_ave_starters_2014_left,home_ave_starters_2015_left,home_ave_starters_2014_right,home_ave_starters_2015_right]
    home_ave_sub_train = np.r_[home_ave_sub_2014,home_ave_sub_2015,home_ave_sub_2014,home_ave_sub_2015]
    home_ave_score_train = np.r_[home_ave_score_2014,home_ave_score_2015,home_ave_score_2014,home_ave_score_2015]

    away_ave_starters_train = np.r_[away_ave_starters_2014_left,away_ave_starters_2015_left,away_ave_starters_2014_right,away_ave_starters_2015_right]
    away_ave_sub_train = np.r_[away_ave_sub_2014,away_ave_sub_2015,away_ave_sub_2014,away_ave_sub_2015]
    away_ave_score_train = np.r_[away_ave_score_2014,away_ave_score_2015,away_ave_score_2014,away_ave_score_2015]

    home_versus_starters_2014_left = np.array([cbp_2014_real_starters_dict_left[homeID,awayID][1] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    home_versus_starters_2015_left = np.array([cbp_2015_real_starters_dict_left[homeID,awayID][1] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])
    home_versus_starters_2014_right = np.array([cbp_2014_real_starters_dict_right[homeID,awayID][1] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    home_versus_starters_2015_right = np.array([cbp_2015_real_starters_dict_right[homeID,awayID][1] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    home_versus_sub_2014 = np.array([cbp_2014_real_sub_dict[homeID,awayID][1] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    home_versus_sub_2015 = np.array([cbp_2015_real_sub_dict[homeID,awayID][1] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    away_versus_starters_2014_left = np.array([cbp_2014_real_starters_dict_left[homeID,awayID][0] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    away_versus_starters_2015_left = np.array([cbp_2015_real_starters_dict_left[homeID,awayID][0] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])
    away_versus_starters_2014_right = np.array([cbp_2014_real_starters_dict_right[homeID,awayID][0] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    away_versus_starters_2015_right = np.array([cbp_2015_real_starters_dict_right[homeID,awayID][0] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    away_versus_sub_2014 = np.array([cbp_2014_real_sub_dict[homeID,awayID][0] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    away_versus_sub_2015 = np.array([cbp_2015_real_sub_dict[homeID,awayID][0] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    home_versus_score_2014 = np.array([real_score_2014_dict[homeID,awayID] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    home_versus_score_2015 = np.array([real_score_2015_dict[homeID,awayID] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    away_versus_score_2014 = np.array([real_score_2014_dict[homeID,awayID] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    away_versus_score_2015 = np.array([real_score_2015_dict[homeID,awayID] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    home_versus_starters_train = np.r_[home_versus_starters_2014_left,home_versus_starters_2015_left,home_versus_starters_2014_right,home_versus_starters_2015_right]
    home_versus_sub_train = np.r_[home_versus_sub_2014,home_versus_sub_2015,home_versus_sub_2014,home_versus_sub_2015]
    home_versus_score_train = np.r_[home_versus_score_2014,home_versus_score_2015,home_versus_score_2014,home_versus_score_2015]

    away_versus_starters_train = np.r_[away_versus_starters_2014_left,away_versus_starters_2015_left,away_versus_starters_2014_right,away_versus_starters_2015_right]
    away_versus_sub_train = np.r_[away_versus_sub_2014,away_versus_sub_2015,away_versus_sub_2014,away_versus_sub_2015]
    away_versus_score_train = np.r_[away_versus_score_2014,away_versus_score_2015,away_versus_score_2014,away_versus_score_2015]

    """averageをシーズン毎に計算"""

    n_games = homeID_2014.shape[0]/2
    homeID_2014_1st = homeID_2014[:n_games]
    awayID_2014_1st = awayID_2014[:n_games]
    homeID_2015_1st = homeID_2015[:n_games]
    awayID_2015_1st = awayID_2015[:n_games]
    homeID_2014_2nd = homeID_2014[n_games:]
    awayID_2014_2nd = awayID_2014[n_games:]
    homeID_2015_2nd = homeID_2015[n_games:]
    awayID_2015_2nd= awayID_2015[n_games:]


    home_2014_1st_nb= [np.where(homeID_2014_1st==ID)[0]  for ID in teamID_2014]
    away_2014_1st_nb = [np.where(awayID_2014_1st==ID)[0] for ID in teamID_2014]
    home_2015_1st_nb= [np.where(homeID_2015_1st==ID)[0] + homeID_2014.shape[0] for ID in teamID_2015]
    away_2015_1st_nb = [np.where(awayID_2015_1st==ID)[0] + homeID_2014.shape[0] for ID in teamID_2015]
    home_2014_2nd_nb= [np.where(homeID_2014_2nd==ID)[0] + np.array(homeID_2014_1st).size  for ID in teamID_2014]
    away_2014_2nd_nb = [np.where(awayID_2014_2nd==ID)[0] + np.array(homeID_2014_1st).size for ID in teamID_2014]
    home_2015_2nd_nb= [np.where(homeID_2015_2nd==ID)[0] + homeID_2014.shape[0] + np.array(homeID_2015_1st).size for ID in teamID_2015]
    away_2015_2nd_nb = [np.where(awayID_2015_2nd==ID)[0] + homeID_2014.shape[0] + np.array(homeID_2015_1st).size for ID in teamID_2015]

    #  divide with each teams
    home_cbp_starters_team_2014_1st_left =[home_cbp_starters_left[n] for n in home_2014_1st_nb]
    away_cbp_starters_team_2014_1st_left =[away_cbp_starters_left[n] for n in away_2014_1st_nb]
    home_cbp_starters_team_2015_1st_left =[home_cbp_starters_left[n] for n in home_2015_1st_nb]
    away_cbp_starters_team_2015_1st_left =[away_cbp_starters_left[n] for n in away_2015_1st_nb]
    home_cbp_starters_team_2014_2nd_left =[home_cbp_starters_left[n] for n in home_2014_2nd_nb]
    away_cbp_starters_team_2014_2nd_left =[away_cbp_starters_left[n] for n in away_2014_2nd_nb]
    home_cbp_starters_team_2015_2nd_left =[home_cbp_starters_left[n] for n in home_2015_2nd_nb]
    away_cbp_starters_team_2015_2nd_left =[away_cbp_starters_left[n] for n in away_2015_2nd_nb]


    # create dictionary of cbp each team{teamID:cbp data}
    home_cbp_starters_team_2014_1st_left = np.array([[np.average(row) for row in team.T] for team in home_cbp_starters_team_2014_1st_left])
    home_cbp_2014_1st_ave_starters_dict_left = dict(zip(teamID_2014, home_cbp_starters_team_2014_1st_left))

    away_cbp_starters_team_2014_1st_left = np.array([[np.average(row) for row in team.T] for team in away_cbp_starters_team_2014_1st_left])
    away_cbp_2014_1st_ave_starters_dict_left = dict(zip(teamID_2014, away_cbp_starters_team_2014_1st_left))

    home_cbp_starters_team_2015_1st_left = np.array([[np.average(row) for row in team.T] for team in home_cbp_starters_team_2015_1st_left])
    home_cbp_2015_1st_ave_starters_dict_left = dict(zip(teamID_2015, home_cbp_starters_team_2015_1st_left))

    away_cbp_starters_team_2015_1st_left = np.array([[np.average(row) for row in team.T] for team in away_cbp_starters_team_2015_1st_left])
    away_cbp_2015_1st_ave_starters_dict_left = dict(zip(teamID_2015, away_cbp_starters_team_2015_1st_left))

    home_cbp_starters_team_2014_2nd_left = np.array([[np.average(row) for row in team.T] for team in home_cbp_starters_team_2014_2nd_left])
    home_cbp_2014_2nd_ave_starters_dict_left = dict(zip(teamID_2014, home_cbp_starters_team_2014_2nd_left))

    away_cbp_starters_team_2014_2nd_left = np.array([[np.average(row) for row in team.T] for team in away_cbp_starters_team_2014_2nd_left])
    away_cbp_2014_2nd_ave_starters_dict_left = dict(zip(teamID_2014, away_cbp_starters_team_2014_2nd_left))

    home_cbp_starters_team_2015_2nd_left = np.array([[np.average(row) for row in team.T] for team in home_cbp_starters_team_2015_2nd_left])
    home_cbp_2015_2nd_ave_starters_dict_left = dict(zip(teamID_2015, home_cbp_starters_team_2015_2nd_left))

    away_cbp_starters_team_2015_2nd_left = np.array([[np.average(row) for row in team.T] for team in away_cbp_starters_team_2015_2nd_left])
    away_cbp_2015_2nd_ave_starters_dict_left = dict(zip(teamID_2015, away_cbp_starters_team_2015_2nd_left))

    # divide with each teams
    home_cbp_starters_team_2014_1st_right =[home_cbp_starters_right[n] for n in home_2014_1st_nb]
    away_cbp_starters_team_2014_1st_right =[away_cbp_starters_right[n] for n in away_2014_1st_nb]
    home_cbp_starters_team_2015_1st_right =[home_cbp_starters_right[n] for n in home_2015_1st_nb]
    away_cbp_starters_team_2015_1st_right =[away_cbp_starters_right[n] for n in away_2015_1st_nb]
    home_cbp_starters_team_2014_2nd_right =[home_cbp_starters_right[n] for n in home_2014_2nd_nb]
    away_cbp_starters_team_2014_2nd_right =[away_cbp_starters_right[n] for n in away_2014_2nd_nb]
    home_cbp_starters_team_2015_2nd_right =[home_cbp_starters_right[n] for n in home_2015_2nd_nb]
    away_cbp_starters_team_2015_2nd_right =[away_cbp_starters_right[n] for n in away_2015_2nd_nb]


    # create dictionary of cbp each team{teamID:cbp data}
    home_cbp_starters_team_2014_1st_right = np.array([[np.average(row) for row in team.T] for team in home_cbp_starters_team_2014_1st_right])
    home_cbp_2014_1st_ave_starters_dict_right = dict(zip(teamID_2014, home_cbp_starters_team_2014_1st_right))

    away_cbp_starters_team_2014_1st_right = np.array([[np.average(row) for row in team.T] for team in away_cbp_starters_team_2014_1st_right])
    away_cbp_2014_1st_ave_starters_dict_right = dict(zip(teamID_2014, away_cbp_starters_team_2014_1st_right))

    home_cbp_starters_team_2015_1st_right = np.array([[np.average(row) for row in team.T] for team in home_cbp_starters_team_2015_1st_right])
    home_cbp_2015_1st_ave_starters_dict_right = dict(zip(teamID_2015, home_cbp_starters_team_2015_1st_right))

    away_cbp_starters_team_2015_1st_right = np.array([[np.average(row) for row in team.T] for team in away_cbp_starters_team_2015_1st_right])
    away_cbp_2015_1st_ave_starters_dict_right = dict(zip(teamID_2015, away_cbp_starters_team_2015_1st_right))

    home_cbp_starters_team_2014_2nd_right = np.array([[np.average(row) for row in team.T] for team in home_cbp_starters_team_2014_2nd_right])
    home_cbp_2014_2nd_ave_starters_dict_right = dict(zip(teamID_2014, home_cbp_starters_team_2014_2nd_right))

    away_cbp_starters_team_2014_2nd_right = np.array([[np.average(row) for row in team.T] for team in away_cbp_starters_team_2014_2nd_right])
    away_cbp_2014_2nd_ave_starters_dict_right = dict(zip(teamID_2014, away_cbp_starters_team_2014_2nd_right))

    home_cbp_starters_team_2015_2nd_right = np.array([[np.average(row) for row in team.T] for team in home_cbp_starters_team_2015_2nd_right])
    home_cbp_2015_2nd_ave_starters_dict_right = dict(zip(teamID_2015, home_cbp_starters_team_2015_2nd_right))

    away_cbp_starters_team_2015_2nd_right = np.array([[np.average(row) for row in team.T] for team in away_cbp_starters_team_2015_2nd_right])
    away_cbp_2015_2nd_ave_starters_dict_right = dict(zip(teamID_2015, away_cbp_starters_team_2015_2nd_right))


    #  divide with each teams
    home_cbp_sub_team_2014_1st =[home_cbp_sub[n] for n in home_2014_1st_nb]
    away_cbp_sub_team_2014_1st =[away_cbp_sub[n] for n in away_2014_1st_nb]
    home_cbp_sub_team_2015_1st =[home_cbp_sub[n] for n in home_2015_1st_nb]
    away_cbp_sub_team_2015_1st =[away_cbp_sub[n] for n in away_2015_1st_nb]
    home_cbp_sub_team_2014_2nd =[home_cbp_sub[n] for n in home_2014_2nd_nb]
    away_cbp_sub_team_2014_2nd =[away_cbp_sub[n] for n in away_2014_2nd_nb]
    home_cbp_sub_team_2015_2nd =[home_cbp_sub[n] for n in home_2015_2nd_nb]
    away_cbp_sub_team_2015_2nd =[away_cbp_sub[n] for n in away_2015_2nd_nb]

    # create dictionary of cbp each team{teamID:cbp data}
    home_cbp_sub_team_2014_1st = np.array([[np.average(row) for row in team.T] for team in home_cbp_sub_team_2014_1st])
    home_cbp_2014_1st_ave_sub_dict = dict(zip(teamID_2014, home_cbp_sub_team_2014_1st))

    away_cbp_sub_team_2014_1st = np.array([[np.average(row) for row in team.T] for team in away_cbp_sub_team_2014_1st])
    away_cbp_2014_1st_ave_sub_dict = dict(zip(teamID_2014, away_cbp_sub_team_2014_1st))

    home_cbp_sub_team_2015_1st = np.array([[np.average(row) for row in team.T] for team in home_cbp_sub_team_2015_1st])
    home_cbp_2015_1st_ave_sub_dict = dict(zip(teamID_2015, home_cbp_sub_team_2015_1st))

    away_cbp_sub_team_2015_1st = np.array([[np.average(row) for row in team.T] for team in away_cbp_sub_team_2015_1st])
    away_cbp_2015_1st_ave_sub_dict = dict(zip(teamID_2015, away_cbp_sub_team_2015_1st))

    home_cbp_sub_team_2014_2nd = np.array([[np.average(row) for row in team.T] for team in home_cbp_sub_team_2014_2nd])
    home_cbp_2014_2nd_ave_sub_dict = dict(zip(teamID_2014, home_cbp_sub_team_2014_2nd))

    away_cbp_sub_team_2014_2nd = np.array([[np.average(row) for row in team.T] for team in away_cbp_sub_team_2014_2nd])
    away_cbp_2014_2nd_ave_sub_dict = dict(zip(teamID_2014, away_cbp_sub_team_2014_2nd))

    home_cbp_sub_team_2015_2nd = np.array([[np.average(row) for row in team.T] for team in home_cbp_sub_team_2015_2nd])
    home_cbp_2015_2nd_ave_sub_dict = dict(zip(teamID_2015, home_cbp_sub_team_2015_2nd))

    away_cbp_sub_team_2015_2nd = np.array([[np.average(row) for row in team.T] for team in away_cbp_sub_team_2015_2nd])
    away_cbp_2015_2nd_ave_sub_dict = dict(zip(teamID_2015, away_cbp_sub_team_2015_2nd))


    # divide with home and away
    home_score_team_2014_1st =np.array([np.array(score_2014_norm)[n] for n in home_2014_1st_nb])
    away_score_team_2014_1st =np.array([np.array(score_2014_norm)[n]for n in away_2014_1st_nb])
    home_score_team_2015_1st =np.array([np.array(score_2015_norm)[n-np.array(home_2014_nb).size] for n in home_2015_1st_nb])
    away_score_team_2015_1st =np.array([np.array(score_2015_norm)[n-np.array(away_2014_nb).size]  for n in away_2015_1st_nb])
    home_score_team_2014_2nd =np.array([np.array(score_2014_norm)[n] for n in home_2014_2nd_nb])
    away_score_team_2014_2nd =np.array([np.array(score_2014_norm)[n]for n in away_2014_2nd_nb])
    home_score_team_2015_2nd =np.array([np.array(score_2015_norm)[n-np.array(home_2014_nb).size] for n in home_2015_2nd_nb])
    away_score_team_2015_2nd =np.array([np.array(score_2015_norm)[n-np.array(away_2014_nb).size]  for n in away_2015_2nd_nb])

    home_ave_score_team_2014_1st = np.array([score.mean(axis=0) for score in home_score_team_2014_1st])
    home_ave_score_2014_1st_dict ={teamID:score for teamID,score in zip(teamID_2014,home_ave_score_team_2014_1st)}

    away_ave_score_team_2014_1st = np.array([score.mean(axis=0) for score in away_score_team_2014_1st])
    away_ave_score_2014_1st_dict ={teamID:score for teamID,score in zip(teamID_2014,away_ave_score_team_2014_1st)}

    home_ave_score_team_2015_1st = np.array([score.mean(axis=0) for score in home_score_team_2015_1st])
    home_ave_score_2015_1st_dict ={teamID:score for teamID,score in zip(teamID_2015,home_ave_score_team_2015_1st)}

    away_ave_score_team_2015_1st = np.array([score.mean(axis=0) for score in away_score_team_2015_1st])
    away_ave_score_2015_1st_dict ={teamID:score for teamID,score in zip(teamID_2015,away_ave_score_team_2015_1st)}

    home_ave_score_team_2014_2nd = np.array([score.mean(axis=0) for score in home_score_team_2014_2nd])
    home_ave_score_2014_2nd_dict ={teamID:score for teamID,score in zip(teamID_2014,home_ave_score_team_2014_2nd)}

    away_ave_score_team_2014_2nd = np.array([score.mean(axis=0) for score in away_score_team_2014_2nd])
    away_ave_score_2014_2nd_dict ={teamID:score for teamID,score in zip(teamID_2014,away_ave_score_team_2014_2nd)}

    home_ave_score_team_2015_2nd = np.array([score.mean(axis=0) for score in home_score_team_2015_2nd])
    home_ave_score_2015_2nd_dict ={teamID:score for teamID,score in zip(teamID_2015,home_ave_score_team_2015_2nd)}

    away_ave_score_team_2015_2nd = np.array([score.mean(axis=0) for score in away_score_team_2015_2nd])
    away_ave_score_2015_2nd_dict ={teamID:score for teamID,score in zip(teamID_2015,away_ave_score_team_2015_2nd)}

    """
    create_train_data

    """

    cnt_nb = 0
    remv_index = []
    for homeID, awayID in zip(awayID_2015, homeID_2015):
        if (homeID,awayID) in remv_IDs:
            remv_index.append(cnt_nb)
        cnt_nb += 1


    home_ave_starters_2014_1st_left = np.array([home_cbp_2014_1st_ave_starters_dict_left[ID] for ID in homeID_2014_1st])
    home_ave_starters_2015_1st_left = np.array([home_cbp_2015_1st_ave_starters_dict_left[ID] for ID in homeID_2015_1st])
    home_ave_starters_2014_1st_right = np.array([home_cbp_2014_1st_ave_starters_dict_right[ID] for ID in homeID_2014_1st])
    home_ave_starters_2015_1st_right = np.array([home_cbp_2015_1st_ave_starters_dict_right[ID] for ID in homeID_2015_1st])

    home_ave_sub_2014_1st = np.array([home_cbp_2014_1st_ave_sub_dict[ID] for ID in homeID_2014_1st])
    home_ave_sub_2015_1st = np.array([home_cbp_2015_1st_ave_sub_dict[ID] for ID in homeID_2015_1st])
    home_ave_score_2014_1st = np.array([home_ave_score_2014_1st_dict[ID] for ID in homeID_2014_1st])
    home_ave_score_2015_1st = np.array([home_ave_score_2015_1st_dict[ID] for ID in homeID_2015_1st])

    away_ave_starters_2014_1st_left = np.array([away_cbp_2014_1st_ave_starters_dict_left[ID] for ID in awayID_2014_1st])
    away_ave_starters_2015_1st_left = np.array([away_cbp_2015_1st_ave_starters_dict_left[ID] for ID in awayID_2015_1st])
    away_ave_starters_2014_1st_right = np.array([away_cbp_2014_1st_ave_starters_dict_right[ID] for ID in awayID_2014_1st])
    away_ave_starters_2015_1st_right = np.array([away_cbp_2015_1st_ave_starters_dict_right[ID] for ID in awayID_2015_1st])

    away_ave_sub_2014_1st = np.array([away_cbp_2014_1st_ave_sub_dict[ID] for ID in awayID_2014_1st])
    away_ave_sub_2015_1st = np.array([away_cbp_2015_1st_ave_sub_dict[ID] for ID in awayID_2015_1st])

    away_ave_score_2014_1st = np.array([away_ave_score_2014_1st_dict[ID] for ID in awayID_2014_1st])
    away_ave_score_2015_1st = np.array([away_ave_score_2015_1st_dict[ID] for ID in awayID_2015_1st])



    home_ave_starters_2014_2nd_left = np.array([home_cbp_2014_2nd_ave_starters_dict_left[ID] for ID in homeID_2014_2nd])
    home_ave_starters_2015_2nd_left = np.array([home_cbp_2015_2nd_ave_starters_dict_left[ID] for ID in homeID_2015_2nd])
    home_ave_starters_2014_2nd_right = np.array([home_cbp_2014_2nd_ave_starters_dict_right[ID] for ID in homeID_2014_2nd])
    home_ave_starters_2015_2nd_right = np.array([home_cbp_2015_2nd_ave_starters_dict_right[ID] for ID in homeID_2015_2nd])

    home_ave_sub_2014_2nd = np.array([home_cbp_2014_2nd_ave_sub_dict[ID] for ID in homeID_2014_2nd])
    home_ave_sub_2015_2nd = np.array([home_cbp_2015_2nd_ave_sub_dict[ID] for ID in homeID_2015_2nd])

    home_ave_score_2014_2nd = np.array([home_ave_score_2014_2nd_dict[ID] for ID in homeID_2014_2nd])
    home_ave_score_2015_2nd = np.array([home_ave_score_2015_2nd_dict[ID] for ID in homeID_2015_2nd])

    away_ave_starters_2014_2nd_left = np.array([away_cbp_2014_2nd_ave_starters_dict_left[ID] for ID in awayID_2014_2nd])
    away_ave_starters_2015_2nd_left = np.array([away_cbp_2015_2nd_ave_starters_dict_left[ID] for ID in awayID_2015_2nd])
    away_ave_starters_2014_2nd_right = np.array([away_cbp_2014_2nd_ave_starters_dict_right[ID] for ID in awayID_2014_2nd])
    away_ave_starters_2015_2nd_right = np.array([away_cbp_2015_2nd_ave_starters_dict_right[ID] for ID in awayID_2015_2nd])

    away_ave_sub_2014_2nd = np.array([away_cbp_2014_2nd_ave_sub_dict[ID] for ID in awayID_2014_2nd])
    away_ave_sub_2015_2nd = np.array([away_cbp_2015_2nd_ave_sub_dict[ID] for ID in awayID_2015_2nd])

    away_ave_score_2014_2nd = np.array([away_ave_score_2014_2nd_dict[ID] for ID in awayID_2014_2nd])
    away_ave_score_2015_2nd = np.array([away_ave_score_2015_2nd_dict[ID] for ID in awayID_2015_2nd])



    for ix in reversed(remv_index):
        home_ave_starters_2015_1st_left = np.delete(home_ave_starters_2015_1st_left,ix,0)
        home_ave_starters_2015_1st_right = np.delete(home_ave_starters_2015_1st_right,ix,0)
        home_ave_sub_2015_1st = np.delete(home_ave_sub_2015_1st,ix,0)
        home_ave_score_2015_1st = np.delete(home_ave_score_2015_1st,ix,0)
        away_ave_starters_2015_1st_left  = np.delete(away_ave_starters_2015_1st_left,ix,0)
        away_ave_starters_2015_1st_right  = np.delete(away_ave_starters_2015_1st_right,ix,0)
        away_ave_sub_2015_1st  = np.delete(away_ave_sub_2015_1st,ix,0)
        away_ave_score_2015_1st  = np.delete(away_ave_score_2015_1st,ix,0)

    home_ave_starters_train = np.r_[home_ave_starters_2014_1st_left,home_ave_starters_2014_2nd_left,home_ave_starters_2015_1st_left,home_ave_starters_2015_2nd_left,
                                    home_ave_starters_2014_1st_right,home_ave_starters_2014_2nd_right,home_ave_starters_2015_1st_right,home_ave_starters_2015_2nd_right]
    home_ave_sub_train = np.r_[home_ave_sub_2014_1st,home_ave_sub_2014_2nd,home_ave_sub_2015_1st,home_ave_sub_2015_2nd,
                                    home_ave_sub_2014_1st,home_ave_sub_2014_2nd,home_ave_sub_2015_1st,home_ave_sub_2015_2nd]
    home_ave_score_train = np.r_[home_ave_score_2014_1st,home_ave_score_2014_2nd,home_ave_score_2015_1st,home_ave_score_2015_2nd,
                                    home_ave_score_2014_1st,home_ave_score_2014_2nd,home_ave_score_2015_1st,home_ave_score_2015_2nd]

    away_ave_starters_train = np.r_[away_ave_starters_2014_1st_left,away_ave_starters_2014_2nd_left,away_ave_starters_2015_1st_left,away_ave_starters_2015_2nd_left,
                                    away_ave_starters_2014_1st_right,away_ave_starters_2014_2nd_right,away_ave_starters_2015_1st_right,away_ave_starters_2015_2nd_right]
    away_ave_sub_train = np.r_[away_ave_sub_2014_1st,away_ave_sub_2014_2nd,away_ave_sub_2015_1st,away_ave_sub_2015_2nd,
                                    away_ave_sub_2014_1st,away_ave_sub_2014_2nd,away_ave_sub_2015_1st,away_ave_sub_2015_2nd]
    away_ave_score_train = np.r_[away_ave_score_2014_1st,away_ave_score_2014_2nd,away_ave_score_2015_1st,away_ave_score_2015_2nd,
                                    away_ave_score_2014_1st,away_ave_score_2014_2nd,away_ave_score_2015_1st,away_ave_score_2015_2nd]

    home_versus_starters_2014_left = np.array([cbp_2014_real_starters_dict_left[homeID,awayID][1] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    home_versus_starters_2015_left = np.array([cbp_2015_real_starters_dict_left[homeID,awayID][1] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])
    home_versus_starters_2014_right = np.array([cbp_2014_real_starters_dict_right[homeID,awayID][1] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    home_versus_starters_2015_right = np.array([cbp_2015_real_starters_dict_right[homeID,awayID][1] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    home_versus_sub_2014 = np.array([cbp_2014_real_sub_dict[homeID,awayID][1] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    home_versus_sub_2015 = np.array([cbp_2015_real_sub_dict[homeID,awayID][1] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    away_versus_starters_2014_left = np.array([cbp_2014_real_starters_dict_left[homeID,awayID][0] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    away_versus_starters_2015_left = np.array([cbp_2015_real_starters_dict_left[homeID,awayID][0] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])
    away_versus_starters_2014_right = np.array([cbp_2014_real_starters_dict_right[homeID,awayID][0] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    away_versus_starters_2015_right = np.array([cbp_2015_real_starters_dict_right[homeID,awayID][0] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    away_versus_sub_2014 = np.array([cbp_2014_real_sub_dict[homeID,awayID][0] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    away_versus_sub_2015 = np.array([cbp_2015_real_sub_dict[homeID,awayID][0] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    home_versus_score_2014 = np.array([real_score_2014_dict[homeID,awayID] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    home_versus_score_2015 = np.array([real_score_2015_dict[homeID,awayID] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    away_versus_score_2014 = np.array([real_score_2014_dict[homeID,awayID] for homeID, awayID in zip(awayID_2014, homeID_2014)])
    away_versus_score_2015 = np.array([real_score_2015_dict[homeID,awayID] for homeID, awayID in zip(awayID_2015, homeID_2015) if not (homeID,awayID) in remv_IDs])

    home_versus_starters_train = np.r_[home_versus_starters_2014_left,home_versus_starters_2015_left,home_versus_starters_2014_right,home_versus_starters_2015_right]
    home_versus_sub_train = np.r_[home_versus_sub_2014,home_versus_sub_2015,home_versus_sub_2014,home_versus_sub_2015]
    home_versus_score_train = np.r_[home_versus_score_2014,home_versus_score_2015,home_versus_score_2014,home_versus_score_2015]

    away_versus_starters_train = np.r_[away_versus_starters_2014_left,away_versus_starters_2015_left,away_versus_starters_2014_right,away_versus_starters_2015_right]
    away_versus_sub_train = np.r_[away_versus_sub_2014,away_versus_sub_2015,away_versus_sub_2014,away_versus_sub_2015]
    away_versus_score_train = np.r_[away_versus_score_2014,away_versus_score_2015,away_versus_score_2014,away_versus_score_2015]

    # create target data

    home_result_2014 = home_result[:homeID_2014.shape[0]]
    home_result_2015 = home_result[homeID_2014.shape[0]:]

    away_result_2014 = away_result[:awayID_2014.shape[0]]
    away_result_2015 = away_result[awayID_2014.shape[0]:]

    for ix in remv_index:
        home_result_2015 = np.delete(home_result_2015,ix,0)
        away_result_2015 = np.delete(away_result_2015,ix,0)
        
    # create real dict and data

    home_real_starters_2014_left = np.array([cbp_2014_real_starters_dict_left[homeID,awayID][0] for homeID, awayID in zip( homeID_2014,awayID_2014)])
    home_real_starters_2015_left = np.array([cbp_2015_real_starters_dict_left[homeID,awayID][0] for homeID, awayID in zip(homeID_2015, awayID_2015)])
    home_real_starters_2014_right = np.array([cbp_2014_real_starters_dict_right[homeID,awayID][0] for homeID, awayID in zip(homeID_2014, awayID_2014)])
    home_real_starters_2015_right = np.array([cbp_2015_real_starters_dict_right[homeID,awayID][0] for homeID, awayID in zip(homeID_2015, awayID_2015)])

    away_real_starters_2014_left = np.array([cbp_2014_real_starters_dict_left[homeID,awayID][1] for homeID, awayID in zip( homeID_2014,awayID_2014)])
    away_real_starters_2015_left = np.array([cbp_2015_real_starters_dict_left[homeID,awayID][1] for homeID, awayID in zip(homeID_2015, awayID_2015)])
    away_real_starters_2014_right = np.array([cbp_2014_real_starters_dict_right[homeID,awayID][1] for homeID, awayID in zip(homeID_2014, awayID_2014)])
    away_real_starters_2015_right = np.array([cbp_2015_real_starters_dict_right[homeID,awayID][1] for homeID, awayID in zip(homeID_2015, awayID_2015)])

    for ix in remv_index:
        home_real_starters_2015_left = np.delete(home_real_starters_2015_left,ix,0)
        home_real_starters_2015_right = np.delete(home_real_starters_2015_right,ix,0)
        away_real_starters_2015_left = np.delete(away_real_starters_2015_left,ix,0)
        away_real_starters_2015_right = np.delete(away_real_starters_2015_right,ix,0)


    home_real_starters = np.r_[
        home_real_starters_2014_left,home_real_starters_2015_left,
        home_real_starters_2014_right,home_real_starters_2015_right
    ]
    away_real_starters = np.r_[
        away_real_starters_2014_left,away_real_starters_2015_left,
        away_real_starters_2014_right,away_real_starters_2015_right
    ]

    home_y = np.r_[
        home_result_2014,home_result_2015,
        home_result_2014,home_result_2015,
    ]
    away_y = np.r_[
        away_result_2014,away_result_2015,
        away_result_2014,away_result_2015,
    ]

    train_x = np.c_[
        home_versus_starters_train,away_versus_starters_train,
        home_ave_starters_train, away_ave_starters_train,
        home_versus_score_train # home_ave_score_train,away_ave_score_train,
    ]

    train_y = np.c_[home_y, away_y]

    train_real = np.c_[home_real_starters, away_real_starters]
return train_x, train_y, train_real