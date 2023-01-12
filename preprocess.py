import numpy as np
import pandas as pd
from pyspark.sql.functions import udf, col
from sklearn.preprocessing import StandardScaler
from pyspark import SparkContext, SQLContext

preprocess = StandardScaler()
BIGGER_DATA = False

train_file = 'data/train.csv'
test_file = 'data/test.csv'

cols = ['CFA_Personal', 'CFA_problem', 'Anon Student Id', 'Problem Unit', 'Problem Section',
        'Step Name', 'Problem Name', 'Problem View', 'Correct First Attempt', 'KC_num', 'CFA_Step', 'CFA_KC']
sc = SparkContext('local')
sqlContext = SQLContext(sc)

traindata = sqlContext.read.csv(train_file, sep='\t', header=True)
testdata = sqlContext.read.csv(test_file, sep='\t', header=True)


@udf
def function_CFA_Personal(x):
    if x in pcaf_student.keys():
        return pcaf_student[x][0]
    else:
        return mean_SCFAR


@udf
def function_CFA_problem(x):
    if x in pcaf_problem.keys():
        return pcaf_problem[x]
    else:
        return mean_PCFAR


@udf
def function_CFA_Step(x):
    if x in pcaf_step.keys():
        return pcaf_step[x]
    else:
        return mean_STCFAR


@udf
def function_KC_num(x):
    if pd.isnull(x):
        return 0
    else:
        return x.count('~~') + 1


@udf
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


@udf
def function_Oppo_Mean(x):
    if x is None or x == "None":
        return 0
    return np.mean(list(map(int, x.replace('nan', '0').split('~~'))))


@udf
def encoding_Id(x):
    return sid_dict[x]


@udf
def encoding_Problem_Name(x):
    return names_dict[x]


@udf
def encoding_Problem_Unit(x):
    return units_dict[x]


@udf
def encoding_Problem_Section(x):
    return sections_dict[x]


@udf
def encoding_Step_Name(x):
    return sname_dict[x]


# CFAR Calculation
ID_groups = [x[0] for x in traindata.select('Anon Student Id').distinct().collect()]
ID_groups_list = [(x, traindata.filter(col('Anon Student Id') == x)) for x in ID_groups]
pcaf_student = {student: (group.filter(col('Correct First Attempt') == 1).count(), group.select('Correct First Attempt').count()) for student, group in ID_groups_list}

Problem_name_groups = [x[0] for x in traindata.select('Problem Name').distinct().collect()]
Problem_name_groups_list = [(x, traindata.filter(col('Problem Name') == x)) for x in ID_groups]
pcaf_problem = {problem: 1.0 * group.filter(col('Correct First Attempt') == 1).count() / group.select('Correct First Attempt').count() for problem, group in Problem_name_groups_list}

Step_name_groups = [x[0] for x in traindata.select('Problem Name').distinct().collect()]
Step_name_groups_list = [(x, traindata.filter(col('Problem Name') == x)) for x in ID_groups]
pcaf_step = {step: 1.0 * group.filter(col('Correct First Attempt') == 1).count() / group.select('Correct First Attempt').count() for step, group in Step_name_groups_list}

KC = np.unique([row['KC(Default)'].split("~~") for row in traindata if not pd.isnull(row['KC(Default)'])])

KC_dic = {kc: [0, 0] for kc in KC}
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

sids = list(set(train_x['Anon Student Id']).union(set(test_x['Anon Student Id'])))
sid_dict = {sid: index for index, sid in enumerate(sids)}

names = list(set(train_x['Problem Name']).union(set(test_x['Problem Name'])))
names_dict = {name: index for index, name in enumerate(names)}

units = list(set(train_x['Problem Unit']).union(set(test_x['Problem Unit'])))
units_dict = {hierarchy: index for index, hierarchy in enumerate(units)}

sections = list(set(train_x['Problem Section']).union(set(test_x['Problem Section'])))
sections_dict = {hierarchy: index for index, hierarchy in enumerate(sections)}

sname = list(set(train_x['Step Name']).union(set(test_x['Step Name'])))
sname_dict = {name: index for index, name in enumerate(sname)}

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
