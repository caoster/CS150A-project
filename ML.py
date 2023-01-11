import lightgbm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

preprocess = StandardScaler()
BIGGER_DATA = False

train_file = 'data/train.csv'
test_file = 'data/test.csv'


def train():
    cols = ['CFA_Personal', 'CFA_problem', 'Anon Student Id', 'Problem Unit', 'Problem Section',
            'Step Name', 'Problem Name', 'Problem View', 'KC_num', 'CFA_Step', 'CFA_KC']
    train_df = pd.read_csv('train_pre.csv', sep='\t')
    test_df = pd.read_csv('test_pre.csv', sep='\t')

    X = train_df.dropna()
    y = np.array(X['Correct First Attempt']).astype(int).ravel()
    del X['Correct First Attempt']
    XX = test_df.dropna()
    yy = np.array(XX['Correct First Attempt']).astype(int).ravel()
    del XX['Correct First Attempt']

    # Normalization
    """
    for item in  cols:
        X[item] = preprocess.fit_transform(np.array(X[item]).reshape(-1, 1))
        XX[item] = preprocess.fit_transform(np.array(XX[item]).reshape(-1, 1)) 
    """

    clf = VotingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=.01,
                                                                     n_jobs=4, random_state=None, verbose=0)),
                                       ('lgbm', lightgbm.LGBMClassifier(boosting_type='gbdt', objective='binary', max_depth=4, num_leaves=31,
                                                                        learning_rate=0.1, n_estimators=2150, n_jobs=-1, min_child_weight=1, seed=33, subsample=0.85, subsample_freq=1,
                                                                        boost_from_average=False, reg_lambda=0.12,
                                                                        verbose=-1))], voting='soft', weights=[1, 1.5])
    clf.fit(X, y)
    y_pred = clf.predict_proba(XX)[:, 1]
    print('VoteClassifier', np.sqrt(mean_squared_error(y_pred, yy)))

    XX_withnan = test_df
    yy_withnan = np.array(XX_withnan['Correct First Attempt']).astype(float).ravel()
    del XX_withnan['Correct First Attempt']

    y_pred = clf.predict_proba(XX_withnan)[:, 1]

    for index, val in enumerate(yy_withnan):
        if np.isnan(val):
            yy_withnan[index] = y_pred[index]
    test_res = pd.read_csv(test_file, sep='\t')
    test_res['Correct First Attempt'] = yy_withnan

    test_res.to_csv('test.csv', sep='\t', index=False)


train()
