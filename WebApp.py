import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
import numpy as np
from PIL import Image

image = Image.open('/home/tm/Documents/Ki1/LapTrinhPhanMemNangCao/Python_Project/dataset-cover.jpg')
st.image(image, use_column_width=True)

st.write("""
# Credit scoring model
Objective: Create a credit scoring algorithm that predicts the chance of a given loan applicant defaulting on loan repayment.
""")

# get the data
df = pd.read_csv(
    '/home/tm/Documents/Ki1/LapTrinhPhanMemNangCao/Python_Project/hmeq.csv')

# set a subheader
st.subheader('Data information: ')
# show the data as a table
st.dataframe(df)
# show statistics on the data
st.write(df.describe())

# Imputing the input variables
# Nominal features
# Replacement using majority class
# majority class in case of JOB variable is Other
# majority class in case of REASON varibale is DebtCon
df["REASON"].fillna(value="DebtCon", inplace=True)
df["JOB"].fillna(value="Other", inplace=True)
df["DEROG"].fillna(value=0.0, inplace=True)
df["DELINQ"].fillna(value=0.0, inplace=True)
# Numeric features
# Replacement using mean of each class
df.fillna(value=df.mean(), inplace=True)

# Feature transformation
df.loc[df["CLAGE"] >= 600, "CLAGE"] = 600
df.loc[df["VALUE"] >= 400000, "VALUE"] = 400000
df.loc[df["MORTDUE"] >= 300000, "MORTDUE"] = 300000
df.loc[df["DEBTINC"] >= 100, "DEBTINC"] = 100
df["B_DEROG"] = (df["DEROG"] >= 1)*1
df["B_DELINQ"] = (df["DELINQ"] >= 1)*1
df["REASON_1"] = (df["REASON"] == "HomeImp")*1
df["REASON_2"] = (df["REASON"] != "HomeImp")*1
df["JOB_1"] = (df["JOB"] == "Other")*1
df["JOB_2"] = (df["JOB"] == "Office")*1
df["JOB_3"] = (df["JOB"] == "Sales")*1
df["JOB_4"] = (df["JOB"] == "Mgr")*1
df["JOB_5"] = (df["JOB"] == "ProfExe")*1
df["JOB_6"] = (df["JOB"] == "Self")*1
df.drop(["JOB", "REASON"], axis=1, inplace=True)
df["YOJ"] = df["YOJ"].apply(lambda t : np.log(t+1))

# show the data as a chart
# chart = st.bar_chart(df)

# Split the data into independent 'X' an dependent 'Y' variables
# removing the features BAD,JOB,REASON from the input features set
df_new2 = pd.DataFrame(SelectKBest(f_classif, k=18).fit_transform(
    df.drop(["BAD"], axis=1), df["BAD"]))

# st.write(df_new2)
x = df_new2
y = df["BAD"]

# split the data set
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=.33, random_state=1)

# get the feature input from user

def get_user_input():
    loan = st.sidebar.slider('Amount of the loan request', min_value=1000, value=1700, max_value=90000, step=1000)
    mortdue = st.sidebar.slider('Amount due on existing mortgage', min_value=2000, value=97800, max_value=300000, step=2000)
    value_of_curr_prop = st.sidebar.slider('Value of current property', min_value=8000, value=112000, max_value=400000, step=2000)
    yoj = st.sidebar.slider('Years at present job', min_value=0.0, value=1.4, max_value=4.0, step=0.1)
    derog = st.sidebar.slider('Number of major derogatory reports', min_value=0, value=0, max_value=10)
    delinq = st.sidebar.slider('Number of delinquent credit lines', min_value=0, value=0, max_value=15)
    clage = st.sidebar.slider('Age of oldest trade line in months', min_value=0.0, value=93.3, max_value=600.0)
    ninq = st.sidebar.slider('Number of recent credit lines', min_value=0, value=0, max_value=17)
    clno = st.sidebar.slider('Number of credit lines', min_value=0, value=14, max_value=71)
    debtinc = st.sidebar.slider('Debt-to-income ratio', min_value = 0.5, value=93.8, max_value=100.0)

    # store dictionary into a variable
    user_data = {
        'Loan': loan,
        'Mortdue': mortdue,
        'Value': value_of_curr_prop,
        'YOJ': np.log(yoj+1),
        'Derog': derog,
        'REASON': 'HomeImp',
        'JOB': 'Office',
        'Delinq': delinq,
        'Clage': clage,
        'Ninq': ninq,
        'Clno': clno,
        'Debtinc': debtinc,
    }

    # transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features


# store the user input into a variable
user_input = get_user_input()
user_input["REASON_1"] = (user_input["REASON"] == "HomeImp")*1
user_input["REASON_2"] = (user_input["REASON"] != "HomeImp")*1
user_input["JOB_1"] = (user_input["JOB"] == "Other")*1
user_input["JOB_2"] = (user_input["JOB"] == "Office")*1
user_input["JOB_3"] = (user_input["JOB"] == "Sales")*1
user_input["JOB_4"] = (user_input["JOB"] == "Mgr")*1
user_input["JOB_5"] = (user_input["JOB"] == "ProfExe")*1
user_input["JOB_6"] = (user_input["JOB"] == "Self")*1
user_input.drop(["JOB", "REASON"], axis=1, inplace=True)

# set a subheader and display the user input
st.subheader('User input:')
st.write(user_input)


# create and train the model
clf_tree = DecisionTreeClassifier()
clf_tree.max_depth = 100
clf_tree.fit(x_tr, y_tr)
y_pre = clf_tree.predict(x_te)

accuracy = accuracy_score(y_te, y_pre) * 100
f1 = f1_score(y_te, y_pre, average="macro") * 100
precision = precision_score(y_te, y_pre, average="macro") * 100
recall = recall_score(y_te, y_pre, average="macro") * 100

# store the score
scores_dict = {
    'Accuracy': accuracy,
    'F1': f1,
    'Precision': precision,
    'Recall': recall,
}

scores = pd.DataFrame(scores_dict, index=[0])
# show the models metrics
st.subheader('Model test score:')
st.write(scores)

# store the models predictions into a variable
prediction = clf_tree.predict(user_input)

# set a subheader and display the classification
st.subheader('Classification:')
st.write(prediction)