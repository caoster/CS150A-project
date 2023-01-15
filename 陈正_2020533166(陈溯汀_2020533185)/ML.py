import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
import lightgbm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
preprocess = StandardScaler()
preprocess_minmax=MinMaxScaler()
BIGGER_DATA = False

train_file = 'data/train.csv'
test_file = 'data/test.csv'

def prepare():
    onehotencoder=OneHotEncoder()
    #'Problem Unit', 'Problem Section',
    cols = ['CFA_Personal', 'CFA_problem', 'CFA_Step', 'CFA_KC','Anon Student Id', 'Problem Unit', 'Problem Section',
             'Problem Name', 'Step Name','Problem View','KC_num','Oppo_Mean','Oppo_Min', 'Correct First Attempt']
    #cols = ['Anon Student Id','Problem Hierarchy','Problem Name',
            #'Step Name', 'Problem View', 'Correct First Attempt', 'KC(Default)','Opportunity(Default)']
    traindata = pd.read_csv(train_file, sep='\t')
    testdata = pd.read_csv(test_file, sep='\t')

    def function_CFA_Personal(x):
        if x in pcaf_student.keys():
            return pcaf_student[x]
        else:
            return mean_pcaf_student

    def function_CFA_problem(x):
        if x in pcaf_problem.keys():
            return pcaf_problem[x]
        else:
            return mean_pcaf_problem 

    def function_CFA_Step(x):
        if x in pcaf_step.keys():
            return pcaf_step[x]
        else:
            return mean_pcaf_step

    def function_CFA_KC(x):
        if x in pcaf_kc.keys():
            return pcaf_kc[x]
        else:
            return mean_pcaf_kc

    def function_KC_num(x):
        if pd.isnull(x):
            return 0
        else:
            return x.count('~~') + 1
    """
    def func_kc(x):
        kc_list = []
        if not pd.isnull(x):
            kc_list = x.split("~~")
        P_kc = 1
        if kc_list:
            for kc in kc_list:
                if kc in KC:
                    P_kc *= KC_dic[kc][0]/KC_dic[kc][1]
                else:
                    P_kc *= KC_avg
        else:
            P_kc = KC_avg
        return P_kc
    """
    def function_Oppo_Mean(x):
        return np.mean(list(map(int, x.replace('nan', '0').split('~~'))))
    def function_Oppo_Min(x):
        return np.min(list(map(int, x.replace('nan', '0').split('~~'))))

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
    pcaf_kc = {}
    # CFAR Calculation
    for student, group in traindata.groupby(['Anon Student Id']):
        pcaf_student[student] = 1.0 * len(group[group['Correct First Attempt'] == 1]) / len(group['Correct First Attempt'])
    
    for problem, group in traindata.groupby(['Problem Name']):
        pcaf_problem[problem] = 1.0 * len(group[group['Correct First Attempt'] == 1]) / len(group['Correct First Attempt'])
    
    for step, group in traindata.groupby(['Step Name']):
        pcaf_step[step] = 1.0 * len(group[group['Correct First Attempt'] == 1]) / len(group['Correct First Attempt'])
    
    for kc, group in traindata.groupby(['KC(Default)']):
        pcaf_kc[kc] = 1.0 * len(group[group['Correct First Attempt'] == 1]) / len(group['Correct First Attempt'])
    """
    KC = []
    for _,row in traindata.iterrows():
        if pd.isnull(row['KC(Default)']):
            continue
        KC.extend(row['KC(Default)'].split("~~"))
    KC = np.unique(KC)

    KC_dic={}
    for kc in KC:
        KC_dic[kc] = [0, 0]
    for _,row in traindata.iterrows():
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
    """

    mean_pcaf_student = np.mean(list(pcaf_student.values()))
    mean_pcaf_problem = np.mean(list(pcaf_problem.values()))
    mean_pcaf_step = np.mean(list(pcaf_step.values())) 
    mean_pcaf_kc = np.mean(list(pcaf_kc.values())) 
    #KC_avg = correct/alll

    traindata['CFA_Personal'] = traindata['Anon Student Id'].apply(function_CFA_Personal)
    traindata['CFA_problem'] = traindata['Problem Name'].apply(function_CFA_problem)
    traindata['CFA_Step'] = traindata['Step Name'].apply(function_CFA_Step)
    traindata['CFA_KC'] = traindata['KC(Default)'].apply(function_CFA_KC)

    traindata['Problem Unit'] = traindata['Problem Hierarchy'].str.split(',', 1).str[0]
    traindata['Problem Section'] = traindata['Problem Hierarchy'].str.split(',', 1).str[1]

    traindata['KC_num'] = traindata['KC(Default)'].astype("str").apply(function_KC_num)
    traindata['Oppo_Mean'] = traindata['Opportunity(Default)'].astype("str").apply(function_Oppo_Mean)
    traindata['Oppo_Min'] = traindata['Opportunity(Default)'].astype("str").apply(function_Oppo_Min)


    testdata['CFA_Personal'] = testdata['Anon Student Id'].apply(function_CFA_Personal)
    testdata['CFA_problem'] = testdata['Problem Name'].apply(function_CFA_problem)
    testdata['CFA_Step'] = testdata['Step Name'].apply(function_CFA_Step)
    testdata['CFA_KC'] = testdata['KC(Default)'].apply(function_CFA_KC)

    testdata['Problem Unit'] = testdata['Problem Hierarchy'].str.split(',', 1).str[0]
    testdata['Problem Section'] = testdata['Problem Hierarchy'].str.split(',', 1).str[1]

    testdata['KC_num'] = testdata['KC(Default)'].astype("str").apply(function_KC_num)
    testdata['Oppo_Mean'] = testdata['Opportunity(Default)'].astype("str").apply(function_Oppo_Mean)
    testdata['Oppo_Min'] = testdata['Opportunity(Default)'].astype("str").apply(function_Oppo_Min)

    train_x = traindata[cols].copy()
    test_x = testdata[cols].copy()


    sid_dict = {}
    names_dict = {}
    units_dict = {}
    sections_dict = {}
    sname_dict = {}
    """
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
    
    
    #one-hot encode
    
    one_column = ['Anon Student Id','Problem Unit', 'Problem Section', 'Step Name', 'Problem Name']
    train_x = pd.get_dummies(train_x, columns = one_column, dummy_na=True)
    test_x = pd.get_dummies(test_x, columns = one_column, dummy_na=True)
    """
    """
    KC = []
    for _,row in train_x.iterrows():
        if pd.isnull(row['KC(Default)']):
            continue
        KC.extend(row['KC(Default)'].split("~~"))

    for _,row in test_x.iterrows():
        if pd.isnull(row['KC(Default)']):
            continue
        KC.extend(row['KC(Default)'].split("~~"))
    KC = np.unique(KC)
    
    oppo_num={}
    oppo_value={}
    for kc in KC:
       oppo_value[kc]=0.0
       oppo_num[kc]=0.0

    for _,row in train_x.iterrows():
        if pd.isnull(row['KC(Default)']):
            continue
        kc_list=row['KC(Default)'].split("~~")
        oppo_list=row['Opportunity(Default)'].split("~~")
        for index,kc in enumerate(kc_list):
            oppo_value[kc]=oppo_value[kc]+float(oppo_list[index])
            oppo_num[kc]=oppo_num[kc]+1.0

    for _,row in test_x.iterrows():
        if pd.isnull(row['KC(Default)']):
            continue
        kc_list=row['KC(Default)'].split("~~")
        oppo_list=row['Opportunity(Default)'].split("~~")
        for index,kc in enumerate(kc_list):
            oppo_value[kc]=oppo_value[kc]+float(oppo_list[index])
            oppo_num[kc]=oppo_num[kc]+1.0
    oppo_mean={}
    for kc in KC:
        oppo_mean[kc]=oppo_value[kc]/oppo_num[kc]
    
    #print(train_df.type)
    for item in KC:
        #train_x=train_x.assign(item=0)
        #test_x=test_x.assign(item=0)
        train_x.insert(len(train_x.columns),item,0, allow_duplicates=False)
        test_x.insert(len(test_x.columns),item,0, allow_duplicates=False)

    for  i,row in train_x.iterrows():
        if pd.isnull(row['KC(Default)']):
            continue
        kc_list=row['KC(Default)'].split("~~")
        oppo_list=row['Opportunity(Default)'].split("~~")
        len_kc=len(kc_list)
        for j in range(len_kc):
            train_x.loc[i,kc_list[j]]=oppo_list[j]

    for  i,row in test_x.iterrows():
        if pd.isnull(row['KC(Default)']):
            continue
        kc_list=row['KC(Default)'].split("~~")
        oppo_list=row['Opportunity(Default)'].split("~~")
        len_kc=len(kc_list)
        for j in range(len_kc):
            test_x.loc[i,kc_list[j]]=oppo_list[j]
    """
    #data=set(train_df).union(set(test_df))
    data=pd.concat([train_x,test_x],axis=0)
    #'Problem Hierarchy','Problem Unit', 'Problem Section'
    features_labelencoder=['Anon Student Id','Problem Unit', 'Problem Section','Step Name', 'Problem Name']
    for i in features_labelencoder:
        labelencoder=LabelEncoder()
        labelencoder=labelencoder.fit(data[i])
        col=labelencoder.transform(train_x[i])
        col=pd.DataFrame(col,columns=[i])
        train_x.drop(columns=[i],inplace=True)
        train_x=pd.concat([train_x,col],axis=1)

        col=labelencoder.transform(test_x[i])
        col=pd.DataFrame(col,columns=[i])
        test_x.drop(columns=[i],inplace=True)
        test_x=pd.concat([test_x,col],axis=1)
    """   
    cols=['KC(Default)','Opportunity(Default)']   
    for col in cols: 
        del train_x[col]
        del test_x[col]
        #train_x.drop(columns=[col],inplace=True) 
        #test_x.drop(columns=[col],inplace=True) 
    """
    train_x.to_csv('train_pre.csv', sep='\t', index=False)
    test_x.to_csv('test_pre.csv', sep='\t', index=False)


def train():
    pd.set_option('display.max_columns', None)
    #cols = ['CFA_Personal', 'CFA_problem','Oppo_Mean', 'KC_num', 'CFA_Step', 'CFA_KC']
    train_pre = pd.read_csv('train_pre.csv', sep='\t')
    test_pre = pd.read_csv('test_pre.csv', sep='\t')
    """
    for col in cols:
        train_df.drop(columns=[col],inplace=True) 
        test_df.drop(columns=[col],inplace=True) 
    #one_column = ['KC(Default)']
    #train_df = pd.get_dummies(train_df, columns = one_column, dummy_na=True)
    #test_df = pd.get_dummies(test_df, columns = one_column, dummy_na=True)
    
    features_onehotencoder=['KC(Default)'] 
    for i in features_onehotencoder:
        onehotencoder=onehotencoder.fit(np.array(data['KC(Default)']).reshape(1, -1))
        col=onehotencoder.transform(train_df[i])
        col=pd.DataFrame(col,columns=[i])
        train_df.drop(columns=[i],inplace=True)
        train_df=pd.concat([train_df,col],axis=1)

        onehotencoder=onehotencoder.fit(data[i])
        col=onehotencoder.transform(test_df[i])
        col=pd.DataFrame(col,columns=[i])
        test_df.drop(columns=[i],inplace=True)
        test_df=pd.concat([test_df,col],axis=1)
    """
    #print(train_df.columns)
    #print(test_df.columns)
    X = train_pre.dropna()
    y = np.array(X['Correct First Attempt'])
    del X['Correct First Attempt']
    XX = test_pre.dropna()
    yy = np.array(XX['Correct First Attempt'])
    del XX['Correct First Attempt']

    print(X.columns)
    #Normalization
    """
    one_column = ['Anon Student Id']
    
    X = pd.get_dummies(X, columns = one_column, dummy_na=True)
    XX = pd.get_dummies(XX, columns = one_column, dummy_na=True)
    
    pv=pd.DataFrame(preprocess_minmax.fit_transform(X['Problem View'].values.reshape(-1, 1)),columns=['Problem View standard'])
    om=pd.DataFrame(preprocess_minmax.fit_transform( X['Oppo_Mean'].values.reshape(-1, 1)),columns=['Oppo_Mean standard'])
    kcn=pd.DataFrame(preprocess_minmax.fit_transform(X['KC_num'].values.reshape(-1, 1)),columns=['KC_num standard'])

    del X['Problem View']
    del X['Oppo_Mean']
    del X['KC_num']

    X=pd.concat([X,pv,om,kcn],axis=1)

    pv_test=pd.DataFrame(preprocess_minmax.fit_transform(XX['Problem View'].values.reshape(-1, 1)),columns=['Problem View standard'])
    om_test=pd.DataFrame(preprocess_minmax.fit_transform(XX['Oppo_Mean'].values.reshape(-1, 1)),columns=['Oppo_Mean standard'])
    kcn_test=pd.DataFrame(preprocess_minmax.fit_transform(XX['KC_num'].values.reshape(-1, 1)),columns=['KC_num standard'])
    XX.set_index(pd.Index([i for i in range(len(pv_test))]), inplace=True)

    del XX['Problem View']
    del XX['Oppo_Mean']
    del XX['KC_num']

    XX=pd.concat([XX,pv_test,om_test,kcn_test],axis=1)
    """
    """
    for item in cols:
        X[item] = preprocess.fit_transform(np.array(X[item]).reshape(-1, 1))
        XX[item] = preprocess.fit_transform(np.array(XX[item]).reshape(-1, 1)) 
    """
    #X=preprocess.fit_transform(X)
    #XX=preprocess.fit_transform(XX)
    #X=preprocess_minmax.fit_transform(X)
    #XX=preprocess_minmax.fit_transform(XX)
    """
    model_svm = SGDClassifier()
    model_svm.fit(X, y)
    calibrator = CalibratedClassifierCV(model_svm, cv='prefit')
    model=calibrator.fit(X, y)
    y_svm = model.predict_proba(XX)[:, 1]
    print ('Svm RMSE:', np.sqrt(mean_squared_error(y_svm, yy)))
    
    model_naiveBayes = GaussianNB()
    model_naiveBayes.fit(X, y)
    y_naiveBayes = model_naiveBayes.predict(XX)
    print ('NaiveBayes RMSE:', np.sqrt(mean_squared_error(y_naiveBayes, yy)))


    model_logistic = LogisticRegression()
    model_logistic.fit(X, y)
    y_logistic = model_logistic.predict(XX)
    print ('Logistic RMSE:', np.sqrt(mean_squared_error(y_logistic, yy)))
    

    param_dist = {
    'n_estimators': range(80, 200, 4),
    'max_depth': range(2, 15, 1),
    'learning_rate': np.linspace(0.01, 2, 20),
    'subsample': np.linspace(0.7, 0.9, 20),
    'colsample_bytree': np.linspace(0.5, 0.98, 10),
    'min_child_weight': range(1, 9, 1)
    }
    grid = RandomizedSearchCV(estimator=XGBClassifier(),
                        param_distributions=param_dist, scoring='neg_mean_squared_error', verbose=3, n_iter=100,n_jobs=-1, iid=False, cv=3)
    grid.fit(X, y)
    #返回最优的训练器
    print(grid.best_params_)
    best_estimator = grid.best_estimator_
    print(best_estimator)
    #输出最优训练器的精度
    print(grid.best_score_)
    

    model_RandomForest = LinearSVC()
    model_RandomForest.fit(X, y)
    y_RandomForest = model_RandomForest.predict(XX)
    print ('RandomForest RMSE:', np.sqrt(mean_squared_error(y_RandomForest, yy)))
    
    n_estimators = range(80, 200, 4)   
    max_features = ["auto", "sqrt", "log2"]   
    max_depth =range(2, 15, 1) 
    min_samples_split = np.arange(2, 10, step=2)  
    min_samples_leaf = [1, 2, 4]  
    bootstrap = [True, False]
    param_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }
    grid = RandomizedSearchCV(estimator=RandomForestClassifier(),
                        param_distributions=param_grid, scoring='neg_mean_squared_error', verbose=3, n_iter=100,n_jobs=-1, iid=False, cv=3)
    grid.fit(X, y)
    #返回最优的训练器
    print(grid.best_params_)
    best_estimator = grid.best_estimator_
    print(best_estimator)
    #输出最优训练器的精度
    print(grid.best_score_)
    
    param_grid = {
        'n_estimators': list(map(int,np.linspace(50, 4000, 4))),
        'max_depth': range(2, 15, 5),
        'num_leaves': range(31,63,20),
        'learning_rate': np.linspace(0.01, 2, 2),
        'subsample': np.linspace(0.7, 0.9, 2),
        'min_child_weight': range(1, 9, 5),
        'reg_lambda': np.linspace(0, 0.5, 2)
    }
    grid = RandomizedSearchCV(estimator=lightgbm.LGBMClassifier(),
                        param_distributions=param_grid, scoring='neg_mean_squared_error', verbose=3, n_iter=100,n_jobs=-1, iid=False, cv=3)
    grid.fit(X, y)
    #返回最优的训练器
    print(grid.best_params_)
    best_estimator = grid.best_estimator_
    print(best_estimator)
    #输出最优训练器的精度
    print(grid.best_score_)

    """
    """
    model_XGBoost  = XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
        colsample_bylevel=1, colsample_bynode=1,
        colsample_bytree=0.6066666666666667, early_stopping_rounds=None,
        enable_categorical=False, eval_metric=None, feature_types=None,
        gamma=0, gpu_id=-1, grow_policy='depthwise', importance_type=None,
        interaction_constraints='', learning_rate=0.01, max_bin=256,
        max_cat_threshold=64, max_cat_to_onehot=4, max_delta_step=0,
        max_depth=13, max_leaves=0, min_child_weight=3, 
        monotone_constraints='()', n_estimators=104, n_jobs=0,
        num_parallel_tree=1, predictor='auto', random_state=0)
    model_RandomForest = RandomForestClassifier(max_depth=11, min_samples_split=8, n_estimators=136)
    model_LGBM = lightgbm.LGBMClassifier(boosting_type = 'gbdt', objective='binary', max_depth=4, num_leaves = 31, learning_rate=0.1, n_estimators=2150, n_jobs=-1,
                        silent=0, min_child_weight=1, seed=33, subsample=0.85, subsample_freq = 1, boost_from_average = False, reg_lambda = 0.12, verbose = -1)
    
    model_XGBoost .fit(X, y)
    y_XGBoost = model_XGBoost.predict(XX)
    print ('XGBoost  RMSE:', np.sqrt(mean_squared_error(y_XGBoost , yy)))
    y_XGBoost = model_XGBoost.predict_proba(XX)[:, 1]
    print ('XGBoost  RMSE:', np.sqrt(mean_squared_error(y_XGBoost , yy)))

    model_LGBM.fit(X, y)
    y_LGBM = model_LGBM.predict(XX)
    print ('LGBM RMSE:', np.sqrt(mean_squared_error(y_LGBM, yy)))
    y_LGBM = model_LGBM.predict_proba(XX)[:, 1]
    print ('LGBM RMSE:', np.sqrt(mean_squared_error(y_LGBM, yy)))

    model_RandomForest.fit(X, y)
    y_RandomForest = model_RandomForest.predict(XX)
    print ('RandomForest RMSE:', np.sqrt(mean_squared_error(y_RandomForest, yy)))
    y_RandomForest = model_RandomForest.predict_proba(XX)[:, 1]
    print ('RandomForest RMSE:', np.sqrt(mean_squared_error(y_RandomForest, yy)))

    
    clf = VotingClassifier(estimators = [('rf', model_RandomForest),('lgbm', model_LGBM),('xgboost',model_XGBoost)], n_jobs=-1,voting = 'soft', weights=[1,1,1]) 
    clf.fit(X, y)
    y_pred = clf.predict(XX)
    print ('VoteClassifier', np.sqrt(mean_squared_error(y_pred, yy)))
    y_pred = clf.predict_proba(XX)[:, 1]
    print ('VoteClassifier', np.sqrt(mean_squared_error(y_pred, yy)))


    clf = VotingClassifier(estimators = [('rf', RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=.01,\
    n_jobs=4, random_state=None, verbose=0)),('lgbm', lightgbm.LGBMClassifier(boosting_type = 'gbdt', objective='binary', max_depth=4, num_leaves = 31,\
            learning_rate=0.1, n_estimators=2150, n_jobs=-1,silent=0, min_child_weight=1, seed=33, subsample=0.85, subsample_freq = 1, boost_from_average = False, reg_lambda = 0.12, \
                verbose = -1))], voting = 'soft', weights=[1,1.5])
    clf.fit(X, y)
    y_pred = clf.predict(XX)
    print ('VoteClassifier', np.sqrt(mean_squared_error(y_pred, yy)))
    y_pred = clf.predict_proba(XX)[:, 1]
    print ('VoteClassifier', np.sqrt(mean_squared_error(y_pred, yy)))



    
    model_LGBM.fit(X, y)
    y_LGBM = model_LGBM.predict(XX)
    print ('LGBM RMSE:', np.sqrt(mean_squared_error(y_LGBM, yy)))
    
    
    print(model_RandomForest.feature_importances_)
    importances = model_RandomForest.feature_importances_
    indices = np.argsort(importances)[::-1] 
    for f in range(len(X.columns)):   # x_train.shape[1]=13
        print("%2d) %-*s %f" % \
            (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))
    threshold = 0.05
    x_selected = np.array(XX)[:, importances > threshold]
"""

    model_LGBM = lightgbm.LGBMClassifier(boosting_type = 'gbdt', objective='binary', max_depth=4, num_leaves = 31, learning_rate=0.1, n_estimators=2150, n_jobs=-1,
                        silent=0, min_child_weight=1, seed=33, subsample=0.85, subsample_freq = 1, boost_from_average = False, reg_lambda = 0.12, verbose = -1)
    model_LGBM.fit(X, y)
    #print ('LGBM RMSE:', np.sqrt(mean_squared_error(y_LGBM, yy)))
    XX_withnan = test_pre
    #XX_withnan=preprocess_minmax.fit_transform(XX_withnan)
    yy_withnan = np.array(XX_withnan['Correct First Attempt']).astype(float).ravel()

    del XX_withnan['Correct First Attempt']

    y_LGBM = model_LGBM.predict(XX_withnan)
    for index, val in enumerate(yy_withnan):
        if np.isnan(val):
            yy_withnan[index] = y_LGBM[index]
    test_res = pd.read_csv(test_file, sep='\t')
    test_res['Correct First Attempt'] = yy_withnan

    test_res.to_csv('test.csv', sep='\t', index=False)
    

#prepare()
train()

