import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.model_selection import train_test_split
# imprt tree
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score
from sklearn import tree
# import metrics
from sklearn.preprocessing import OneHotEncoder

df_raw = pd.read_csv("C:/Users/krdeg/dev/ozp/Skipping/labeled_data/labeled_df_15.csv", encoding_errors="ignore", on_bad_lines='error', sep=",", index_col=False,
                    usecols=['SessionID','Activity','anomaly'])

df_raw["anomaly"] = df_raw["anomaly"].astype(int)

# count the number of unique SessionID where anomaly == True
count_anomaly_raw = df_raw[df_raw["anomaly"] == 1]["SessionID"].nunique()
count_normal_raw = df_raw[df_raw["anomaly"] == 0]["SessionID"].nunique()
print(f'Amount of anomalous sessions in the dataset:   {count_anomaly_raw}' )
print(f'Amount of normal sessions in the dataset:      {count_normal_raw}')
print(f'total sessions in the dataset:                 {count_anomaly_raw + count_normal_raw}')
distribution =  count_anomaly_raw / count_normal_raw 
print(f'Distribution:                                  {distribution * 100} %')

import random
random.seed(101)

df_anomaly_og = df_raw[df_raw["anomaly"] == 1].copy()
df_normal = df_raw[df_raw["anomaly"] == 0].copy()
nr_of_sessions_used = 50000
injection_rate = nr_of_sessions_used / count_normal_raw
injection_amount = int(injection_rate * count_anomaly_raw)

# get 20 random sessionIDs from the anomaly dataset
anomaly_sessionIDs = random.sample(list(df_anomaly_og["SessionID"].unique()), injection_amount)

df_50k_only_normal = df_normal[df_normal["SessionID"].isin(df_normal["SessionID"].unique()[:nr_of_sessions_used])].copy()
print(df_50k_only_normal["SessionID"].nunique())

df_50k = df_50k_only_normal.append(df_anomaly_og[df_anomaly_og["SessionID"].isin(anomaly_sessionIDs)]).copy()
print(df_50k["SessionID"].nunique())

# remove the sessions with ID in anomaly_sessionIDs from the df_anomaly dataset
# df_anomaly = df_anomaly_og[~df_anomaly_og["SessionID"].isin(anomaly_sessionIDs)].copy()

# df_anomaly.to_csv("gen_sessions/1000/an.csv")


# count the number of unique SessionID where anomaly == True
count_anomaly = df_50k[df_50k["anomaly"] == 1]["SessionID"].nunique()
count_normal = df_50k[df_50k["anomaly"] == 0]["SessionID"].nunique()
print(f'Amount of anomalous sessions in the sampled dataset:   {count_anomaly}')
print(f'Amount of normal sessions in the sampled dataset:      {count_normal}')
distribution =  count_anomaly / count_normal 
print(f'Distribution:                                          {distribution * 100} %')

#  Helper function

# function that add a column 'transition' to the df with activity and consecutive
def create_transition_df(_df:pd.DataFrame) -> pd.DataFrame:
  df = _df.copy()
  # create consecutive column
  df['consecutive'] = df.groupby("SessionID")['Activity'].shift(-1).fillna('END')
  # drop row if Activity == consecutive
  df = df[df["Activity"] != df["consecutive"]]
    # create column with the transition
  df['transition'] = df['Activity'] + "->" + df['consecutive']
  # drop activity and consecutive
  df = df.drop(columns=["Activity", "consecutive"])
  return df

def transition_count(_df):
  # function that counts the number of times a all transitions occurs in a session
  df = _df.copy()
  df = df.groupby("SessionID")["transition"].value_counts().unstack().fillna(0)
  return df
  
def add_anomaly_col(_df:pd.DataFrame, _df_anomaly:pd.DataFrame) -> pd.DataFrame:
  df = _df.copy()
  df_anomaly = _df_anomaly.copy()
  df_anomaly = df_anomaly[["SessionID", "anomaly"]]
  df_anomaly = df_anomaly.drop_duplicates()
  df = df.merge(df_anomaly, on="SessionID", how="left")
  df["anomaly"] = df["anomaly"].fillna(0)
  return df

df_trans = create_transition_df(df_50k)
base_data_1 = transition_count(df_trans)
base_data = add_anomaly_col(base_data_1, df_anomaly_og)


if 'SessionID' in base_data.columns:
  base_data = base_data.drop(columns=["SessionID"])
  
# Function to split the data into train and test data
def split_data(_df):
  df = _df.copy()
  X = df.drop(columns=["anomaly"])
  y = df["anomaly"]
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
  return X_train, X_test, y_train, y_test

# Base test and train data
X_train_BASE, X_test_BASE, y_train_BASE, y_test_BASE = split_data(base_data)

# import the generated sessions:
ses_amount = 5000
base_path = f"C:/Users/krdeg/dev/ozp/Skipping/gen_sessions/{str(ses_amount)}/"

gen_sessions_paths = [
  # base_path + f'5_{ses_amount}.csv',
  # base_path + f'10_{ses_amount}.csv',
  # base_path + f'25_{ses_amount}.csv',
  # base_path + f'50_{ses_amount}.csv',
  # base_path + f'75_{ses_amount}.csv',
  # base_path + f'100_{ses_amount}.csv',
  # base_path + 'an.csv',
  
  # base_path + '75_10000.csv',
  # base_path + '100_10000.csv',
  'C:/Users/krdeg/dev/ozp/Skipping/gen_sessions/only_patterns.csv'
]

auc_score_dicts = {}
dict_prec_and_rec_list = {}

amount_anomalies_list_total = []
precision_score_list_total = []
recall_score_list_total = []


for sessions in gen_sessions_paths:
    amount_anomalies_list = []
    precision_score_list = []
    recall_score_list = []
    
    # build the dataFrame
    cvs = pd.read_csv(sessions)
    # check if column Unnamed: 0 exists
    if "Unnamed: 0" in cvs.columns : cvs = cvs.drop(columns=["Unnamed: 0"])
    # rename 
    cvs = cvs.rename(columns={'URL_FILE':'Activity'})
    
    df_trans_an = create_transition_df(cvs)
    an_df = transition_count(df_trans_an)
    if 'SessionID' in an_df.index:
        an_df = an_df.drop(index=["SessionID"])

    ready_df = an_df.copy()
    ready_df['anomaly'] = 1
    
    for amount_gen in [0,30]:	
        
        if amount_gen != 0 : 
          df_gen = ready_df.head(amount_gen)
        else: 
          df_gen = pd.DataFrame()
          df_gen["anomaly"] = 1
          
        # get the amount of rows in the generated data
        X_train = X_train_BASE.copy()
        X_test = X_test_BASE.copy()
        y_train = y_train_BASE.copy()
        
        # Add the generated anomalies to the training dataset
        X_train_extra = pd.concat([X_train, df_gen.drop(columns=["anomaly"])]).fillna(0)
        y_train_extra = pd.concat([y_train, df_gen["anomaly"]])        

        #Make sure that both dataframes have the same columns
        for column_name in X_train_extra.columns:
            if column_name not in X_test.columns:
                X_test[column_name] = 0

        #Make sure that both dataframes have the same columns
        for column_name in X_test.columns:
            if column_name not in X_train_extra.columns:
                X_test.drop(columns=[column_name], inplace=True)
        
        clf = tree.DecisionTreeClassifier(random_state=42)
        clf.fit(X_train_extra, y_train_extra)
    
        predictions = clf.predict(X_test)
        # test_predictions = clf.predict(X_train_extra)
        
        #AUC predict
        print(sessions, amount_gen)
        print(f'test data accuracy_score: {accuracy_score(y_true=y_test_BASE,y_pred = predictions)}')
        print(f'test data balanced_accuracy_score: {balanced_accuracy_score(y_true=y_test_BASE,y_pred = predictions)}')
        print(f'test data precision_score: {precision_score(y_true=y_test_BASE,y_pred = predictions)}')
       
       
        # print(f'train data accuracy_score: {accuracy_score(y_true=y_train_extra,y_pred = test_predictions)}')
        # print(f'train data balanced_accuracy_score: {balanced_accuracy_score(y_true=y_train_extra,y_pred = test_predictions)}')
        print()
