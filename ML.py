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


def prepare():
    cols = ['CFA_Personal', 'CFA_problem', 'Anon Student Id', 'Problem Unit', 'Problem Section',
            'Step Name', 'Problem Name', 'Problem View', 'Correct First Attempt', 'KC_num', 'CFA_Step', 'CFA_KC']
    traindata = pd.read_csv(train_file, sep='\t')
    testdata = pd.read_csv(test_file, sep='\t')

    def function_CFA_Personal(x):
        if x in pcaf_student.keys():
            return pcaf_student[x][0]
        else:
            return mean_SCFAR

    def function_CFA_problem(x):
        if x in pcaf_problem.keys():
            return pcaf_problem[x]
        else:
            return mean_PCFAR

    def function_CFA_Step(x):
        if x in pcaf_step.keys():
            return pcaf_step[x]
        else:
            return mean_STCFAR

    def function_KC_num(x):
        if pd.isnull(x):
            return 0
        else:
            return x.count('~~') + 1

    def func_kc(x):
        kc_list = []
        if not pd.isnull(x):
            kc_list = x.split("~~")
        P_kc = 1
        if kc_list:
            for kc in kc_list:
                if kc in KC:
                    P_kc *= KC_dic[kc][0] / KC_dic[kc][1]
                else:
                    P_kc *= KC_avg
        else:
            P_kc = KC_avg
        return P_kc

    def function_Oppo_Mean(x):
        return np.mean(list(map(int, x.replace('nan', '0').split('~~'))))

    def encoding_Id(x):
        return sid_dict[x]

    def encoding_Problem_Name(x):
        return names_dict[x]

    def encoding_Problem_Unit(x):
        return units_dict[x]

    def encoding_Problem_Section(x):
        return sections_dict[x]

    def encoding_Step_Name(x):
        return sname_dict[x]

    pcaf_student = {}
    pcaf_problem = {}
    pcaf_step = {}

    # CFAR Calculation
    for student, group in traindata.groupby(['Anon Student Id']):
        pcaf_student[student] = (len(group[group['Correct First Attempt'] == 1]), len(group['Correct First Attempt']))

    for problem, group in traindata.groupby(['Problem Name']):
        pcaf_problem[problem] = 1.0 * len(group[group['Correct First Attempt'] == 1]) / len(group['Correct First Attempt'])

    for step, group in traindata.groupby(['Step Name']):
        pcaf_step[step] = 1.0 * len(group[group['Correct First Attempt'] == 1]) / len(group['Correct First Attempt'])

    KC = []
    for _, row in traindata.iterrows():
        if pd.isnull(row['KC(Default)']):
            continue
        KC.extend(row['KC(Default)'].split("~~"))
    KC = np.unique(KC)

    KC_dic = {}
    for kc in KC:
        KC_dic[kc] = [0, 0]
    for _, row in traindata.iterrows():
        kc_list = []
        if not pd.isnull(row['KC(Default)']):
            kc_list = row['KC(Default)'].split("~~")

        if kc_list:
            for kc in kc_list:
                KC_dic[kc][1] += 1

        if row['Correct First Attempt'] == 1:
            if kc_list:
                for kc in kc_list:
                    KC_dic[kc][0] += 1

    correct = 0
    alll = 0
    for value in KC_dic.values():
        correct += value[0]
        alll += value[1]

    mean_SCFAR = np.mean(list(map(lambda x: x[0], list(pcaf_student.values()))))
    mean_PCFAR = np.mean(list(pcaf_problem.values()))
    mean_STCFAR = np.mean(list(pcaf_step.values()))
    KC_avg = correct / alll

    traindata['CFA_Personal'] = traindata['Anon Student Id'].apply(function_CFA_Personal)
    traindata['CFA_problem'] = traindata['Problem Name'].apply(function_CFA_problem)
    traindata['CFA_Step'] = traindata['Step Name'].apply(function_CFA_Step)
    traindata['KC_num'] = traindata['KC(Default)'].astype("str").apply(function_KC_num)
    traindata['Problem Unit'] = traindata['Problem Hierarchy'].str.split(',', 1).str[0]
    traindata['Problem Section'] = traindata['Problem Hierarchy'].str.split(',', 1).str[1]
    traindata['CFA_KC'] = traindata['KC(Default)'].apply(func_kc)
    train_x = traindata[cols].copy()
    train_x['Oppo_Mean'] = traindata['Opportunity(Default)'].astype("str").apply(function_Oppo_Mean)

    testdata['Problem Unit'] = testdata['Problem Hierarchy'].str.split(',', 1).str[0]
    testdata['Problem Section'] = testdata['Problem Hierarchy'].str.split(',', 1).str[1]
    testdata['CFA_Personal'] = testdata['Anon Student Id'].apply(function_CFA_Personal)
    testdata['CFA_problem'] = testdata['Problem Name'].apply(function_CFA_problem)
    testdata['CFA_Step'] = testdata['Step Name'].apply(function_CFA_Step)
    testdata['KC_num'] = testdata['KC(Default)'].astype("str").apply(function_KC_num)
    testdata['CFA_KC'] = testdata['KC(Default)'].apply(func_kc)
    test_x = testdata[cols].copy()
    test_x['Oppo_Mean'] = testdata['Opportunity(Default)'].astype("str").apply(function_Oppo_Mean)

    sid_dict = {}
    names_dict = {}
    units_dict = {}
    sections_dict = {}
    sname_dict = {}

    sids = list(set(train_x['Anon Student Id']).union(set(test_x['Anon Student Id'])))
    for index, sid in enumerate(sids):
        sid_dict[sid] = index

    names = list(set(train_x['Problem Name']).union(set(test_x['Problem Name'])))
    for index, name in enumerate(names):
        names_dict[name] = index

    units = list(set(train_x['Problem Unit']).union(set(test_x['Problem Unit'])))
    for index, hierarchy in enumerate(units):
        units_dict[hierarchy] = index

    sections = list(set(train_x['Problem Section']).union(set(test_x['Problem Section'])))
    for index, hierarchy in enumerate(sections):
        sections_dict[hierarchy] = index

    sname = list(set(train_x['Step Name']).union(set(test_x['Step Name'])))
    for index, name in enumerate(sname):
        sname_dict[name] = index

    train_x['Anon Student Id'] = train_x['Anon Student Id'].apply(encoding_Id)
    test_x['Anon Student Id'] = test_x['Anon Student Id'].apply(encoding_Id)
    train_x['Problem Name'] = train_x['Problem Name'].apply(encoding_Problem_Name)
    test_x['Problem Name'] = test_x['Problem Name'].apply(encoding_Problem_Name)
    train_x['Problem Unit'] = train_x['Problem Unit'].apply(encoding_Problem_Unit)
    test_x['Problem Unit'] = test_x['Problem Unit'].apply(encoding_Problem_Unit)
    train_x['Problem Section'] = train_x['Problem Section'].apply(encoding_Problem_Section)
    test_x['Problem Section'] = test_x['Problem Section'].apply(encoding_Problem_Section)
    train_x['Step Name'] = train_x['Step Name'].apply(encoding_Step_Name)
    test_x['Step Name'] = test_x['Step Name'].apply(encoding_Step_Name)

    train_x.to_csv('train_pre.csv', sep='\t', index=False)
    test_x.to_csv('test_pre.csv', sep='\t', index=False)


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


prepare()
train()

