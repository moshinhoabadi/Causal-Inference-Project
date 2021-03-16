import pandas as pd
import numpy as np
import copy
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams['figure.figsize'] = (12.8, 9.6)

poss_name = 'possession_until_halftime'
Y_column = 'shots_on_second_half'


# ATE using IPW algotithm
def ATE_IPW(df):
    # Compute ATT using IPW equation
    df_copy = copy.deepcopy(df)
    df1 = df_copy[df_copy['T'] == 1] # treated group
    df0 = df_copy[df_copy['T'] == 0] # control group

    left_term = ((df1[Y_column].divide(df1['propensity'])).sum())/len(df_copy) # the left term in the equation
    right_term = ((df0[Y_column].divide(1-df0['propensity'])).sum())/len(df_copy) # the left term in the equation

    return (left_term - right_term) # ATE

# ATE using S learner algotithm
def ATE_S_learner(df):
    try:
        # perform S-learner (with possibilty to 2d+1 with the comments)

        df_copy = copy.deepcopy(df)
        Y = df_copy[Y_column]
        df_copy = df_copy.drop([Y_column], axis='columns')

        #reg = LinearRegression().fit(df_copy, Y) # fit regression model on the whole data
        reg = DecisionTreeRegressor(random_state=0).fit(df_copy,Y)

        df0 = copy.deepcopy(df_copy) # treated group if they have had T=0
        df1 = copy.deepcopy(df_copy) # treated group if they have had T=1
        df0['T'] = 0
        df1['T'] = 1

        # predict potential outcomes for treated group using the model
        pred1 = reg.predict(df1) # Y1
        pred0 = reg.predict(df0) # Y0
        return np.sum(pred1-pred0)/(len(pred1)) # ATE
    except:
        return None
    
# ATE using T learner algotithm
def ATE_T_learner(df):
    try:
        # perform T-learner 
        df_copy = copy.deepcopy(df)
        df0 = copy.deepcopy(df_copy)
        df1 = copy.deepcopy(df_copy)
        df0 = df[df['T'] == 0] # control group
        df1 = df[df['T'] == 1] # treated group

        Y0 = df0[Y_column]
        Y1 = df1[Y_column]
        df0 = df0.drop(['T',Y_column], axis='columns')
        df1 = df1.drop(['T',Y_column], axis='columns')

        #reg0 = LinearRegression().fit(df0, Y0) # model for T=0
        reg0 = DecisionTreeRegressor(random_state=0).fit(df0,Y0)

        #reg1 = LinearRegression().fit(df1, Y1) # model for T=1
        reg1 = DecisionTreeRegressor(random_state=0).fit(df1,Y1)

        # compute potential outcomes for all observations  
        df_copy = df_copy.drop(['T',Y_column], axis='columns')
        pred0 = reg0.predict(df_copy) # Y0
        pred1 = reg1.predict(df_copy) # Y1
        return np.sum(pred1-pred0)/(len(pred1)) # ATE
    except:
        return None

# ATE using matching algotithm
def ATE_matching(df):
    try:
        # calcualtes ATT using matching
        df_copy = copy.deepcopy(df)
        sum_ate = 0
        df0 = copy.deepcopy(df_copy)
        df1 = copy.deepcopy(df_copy)

        df0 = df[df['T'] == 0] # control group
        df1 = df[df['T'] == 1] # treated group

        Y0 = df0[Y_column]
        Y1 = df1[Y_column]
        T0 = df0['T']
        T1 = df1['T']

        df0 = df0.drop(['T'], axis='columns')
        df1 = df1.drop(['T'], axis='columns')
        neigh0 = KNeighborsClassifier(n_neighbors=1).fit(df0.drop([Y_column], axis='columns'), Y0.apply(lambda x: str(x))) #classifier with control group as training set. The label is Y
        for i,row in df1.iterrows():
            match = neigh0.predict([row.drop(labels=[Y_column])]) # find match
            sum_ate += (row[Y_column] - float(match)) # computes TT

        neigh1 = KNeighborsClassifier(n_neighbors=1).fit(df1.drop([Y_column], axis='columns'), Y1.apply(lambda x: str(x))) #classifier with treatment group as training set. The label is Y
        for i,row in df0.iterrows():
            match = neigh1.predict([row.drop(labels=[Y_column])]) # find match
            sum_ate += (float(match)-row[Y_column]) # computes TT

        return (sum_ate/len(df_copy)) # return ATE
    except:
        return None

def clac_prop(df, trim=True, get_a=False):
    prop_df = copy.deepcopy(df).drop(['T',Y_column], axis='columns')

    clf = LogisticRegression(random_state=0).fit(prop_df, T)

#     print('prop score', clf.score(prop_df, T))

    prob = clf.predict_proba(prop_df)
    propensity_scores = [p[1] for p in prob] 
    df['propensity'] = propensity_scores

    prob_T = [[p,t] for p,t in zip(propensity_scores, T)]
    prob_0 = []
    prob_1 = []
    for pt in prob_T:
        if pt[1] == 1:
            prob_1.append(pt[0])
        else:
            prob_0.append(pt[0])

    if trim:
        min_max  = np.min([np.max(prob_0),np.max(prob_1)])
        max_min = np.max([np.min(prob_0),np.min(prob_1)])
        df = df[df['propensity'] >= max_min]
        df = df[df['propensity'] <= min_max]

    if get_a:
        return prob_1, prob_0
    return df


disc_cov = ['country_name', 'season', 'month', 'stage', 'is_home_team', 
            'curr_buildUpPlayDribblingClass', 'rival_buildUpPlayDribblingClass',
            'curr_buildUpPlayPositioningClass', 'rival_buildUpPlayPositioningClass',
            'curr_chanceCreationPositioningClass', 'rival_chanceCreationPositioningClass',
            'curr_defenceDefenderLineClass', 'rival_defenceDefenderLineClass']

cont_cov = ['shots_on_first_half', 'goals_first_half', 
            'curr_buildUpPlaySpeed', 'rival_buildUpPlaySpeed', 
            'curr_buildUpPlayPassing', 'rival_buildUpPlayPassing', 
            'curr_chanceCreationPassing', 'rival_chanceCreationPassing', 
            'curr_chanceCreationCrossing', 'rival_chanceCreationCrossing', 
            'curr_chanceCreationShooting', 'rival_chanceCreationShooting', 
            'curr_defencePressure', 'rival_defencePressure', 
            'curr_defenceAggression', 'rival_defenceAggression', 
            'curr_defenceTeamWidth', 'rival_defenceTeamWidth',
            'curr_field players rating', 'rival_field players rating', 
            'curr_gk rating', 'rival_gk rating', 
            'curr_shots_1', 'rival_shots_1', 
            'curr_bets', 'draw_bets', 'rival_bets']




# default values
threshold_over_50 = 10
diff = (0, 11)

# for threshold_over_50 in [1, 5, 10, 15]:  # different thresholds
# for diff in [(0, 3), (3, 7), (7, 15)]:  # different difference ranges
for ha in [True]:
    print(threshold_over_50)
    df = pd.read_csv(f"data/Mathces_with_full_team_details_{threshold_over_50}.csv")
    df = df[~pd.isnull(df).any(axis=1)]

#     print('len before filter', len(df))
    df = df[( ( df['country_name'].isin(['Netherlands', 'Germany'])) & (df['stage'] < 33)) | 
            ( (~df['country_name'].isin(['Netherlands', 'Germany'])) & (df['stage'] < 37)) ]
    df = df[(df['curr_field players rating'] - df['rival_field players rating']).abs().between(*diff)]
    df = df[df['is_home_team'] != df['T']]
#     print('len after filter', len(df))


    T = df['T']
    T = T.astype(bool)
    Y = df[Y_column]
    df = df[disc_cov+cont_cov+['match_id', 'T', Y_column]]
    df = pd.get_dummies(df, columns=disc_cov, drop_first=False)
    df = df.fillna(0)
    
    res = []
    for i in range(500):
        df_i = df.groupby("match_id").apply(pd.DataFrame.sample, n=1)
        T = df_i['T']

        df_i = clac_prop(df_i, trim=(False))

        res.append({
                "IPW": ATE_IPW(df_i),
                "S_learner": ATE_S_learner(df_i),
                "T_learner": ATE_T_learner(df_i),
                "Matching_propensity": ATE_matching(df_i[['propensity', Y_column,'T']]),
                "Matching_all": ATE_matching(df_i)
                   })
        if i % 50 == 0:
            print(i, res[-1])

    print('finished!')

    res_df = pd.DataFrame(res)
    print('saving', len(res_df), 'results')
    
    name = '_'+threshold_over_50
    res_df.to_csv(f'results/res_{name}.csv')


