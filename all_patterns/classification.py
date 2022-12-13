import pandas as pd
import numpy as np
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.model_selection import train_test_split
# imprt tree
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn import tree


print('import data')
print()
# import labelled data from all 3 patterns
df_raw = pd.read_csv('C:/Users/krdeg/dev/ozp/Skipping/labeled_data/skipping.csv', encoding_errors="ignore", on_bad_lines='error', sep=",", index_col=False,
                    usecols=['SessionID','Activity','anomaly'])
skip_ano = df_raw[df_raw["anomaly"] == 1].copy()
skip_sessionID = skip_ano["SessionID"].unique()

df_raw = pd.read_csv('C:/Users/krdeg/dev/ozp/Replaced/labeled_data/Replaced.csv', encoding_errors="ignore", on_bad_lines='error', sep=",", index_col=False,
                    usecols=['SessionID','Activity','anomaly'])
replaced_ano = df_raw[df_raw["anomaly"] == 1].copy()
replaced_SessionID = replaced_ano["SessionID"].unique()

df_raw = pd.read_csv('C:/Users/krdeg/dev/ozp/Swapped/labeled_data/Swapped.csv', encoding_errors="ignore", on_bad_lines='error', sep=",", index_col=False,
                    usecols=['SessionID','Activity','anomaly'])
swapped_ano = df_raw[df_raw["anomaly"] == 1].copy()
swapped_SessionID = swapped_ano["SessionID"].unique()

print(f'skip_sessionID: {len(skip_sessionID)}')
print(f'replaced_SessionID: {len(replaced_SessionID)}')
print(f'swap_sessionID: {len(swapped_SessionID)}')
print(f'total number of anomalous sessions: {len(skip_sessionID) + len(replaced_SessionID) + len(swapped_SessionID)}')

all_anomalous_ID = np.concatenate((skip_sessionID, replaced_SessionID, swapped_SessionID), axis=0)

# Helper functions:

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
  df = df.groupby("SessionID")["Activity"].value_counts().unstack().fillna(0)
  return df
  
def add_anomaly_col(_df:pd.DataFrame, _df_anomaly:pd.DataFrame) -> pd.DataFrame:
  df = _df.copy()
  df_anomaly = _df_anomaly.copy()
  df_anomaly = df_anomaly[["SessionID", "anomaly"]]
  df_anomaly = df_anomaly.drop_duplicates()
  df = df.merge(df_anomaly, on="SessionID", how="left")
  df["anomaly"] = df["anomaly"].fillna(0)
  return df

print()
print(f'create base data')
_base_data = df_raw[~df_raw["SessionID"].isin(all_anomalous_ID)].copy()
# get first 50000 sessions
anomaly_df = df_raw[df_raw["SessionID"].isin(all_anomalous_ID)].copy()

_base_data = _base_data[_base_data["SessionID"].isin(_base_data["SessionID"].unique()[:30000])].copy()


df_trans = create_transition_df(_base_data)
# df_trans = _base_data
_base_data = transition_count(df_trans)

df_trans_an = create_transition_df(anomaly_df)
# df_trans_an = anomaly_df
base_data__an = transition_count(df_trans_an)

_base_data['anomaly'] = 0
base_data__an['anomaly'] = 1


base_data = pd.concat([_base_data, base_data__an]).fillna(0)

# base_data = add_anomaly_col(base_data_1, df_anomaly_og)

if 'SessionID' in base_data.columns:
  base_data = base_data.drop(columns=["SessionID"])
  
# Function to split the data into train and test data
def split_data(_df):
  df = _df.copy()
  X = df.drop(columns=["anomaly"])
  y = df["anomaly"]
  X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)
  return X_train, X_test, y_train, y_test

# Base test and train data
X_train_BASE, X_test_BASE, y_train_BASE, y_test_BASE = split_data(base_data)
print(f'X_train_BASE: {y_train_BASE.sum()}')
print(f'X_train_BASE: {y_test_BASE.sum()}')

# import the generated sessions:
ses_amount = 5000
path_swapped = f"C:/Users/krdeg/dev/ozp/Swapped/gen_sessions/{str(ses_amount)}/"
path_skipped = f"C:/Users/krdeg/dev/ozp/Skipping/gen_sessions/{str(ses_amount)}/"
path_replaced = f"C:/Users/krdeg/dev/ozp/Replaced/gen_sessions/{str(ses_amount)}/"

deviation_paths = [
    '5_{ses_amount}.csv',
    # '10_{ses_amount}.csv',
    # '25_{ses_amount}.csv',
    # '50_{ses_amount}.csv',
    # '75_{ses_amount}.csv',
    # '100_{ses_amount}.csv'
    ]


# function that creates a df with the generated sessions
def create_df(index:int):
    swapped = pd.read_csv(path_swapped + deviation_paths[index].format(ses_amount=ses_amount), encoding_errors="ignore", on_bad_lines='error', sep=",", index_col=False)
    swapped['SessionID'] = 'swa' + swapped['SessionID'].astype(str)
    skipped = pd.read_csv(path_skipped + deviation_paths[index].format(ses_amount=ses_amount), encoding_errors="ignore", on_bad_lines='error', sep=",", index_col=False)
    skipped['SessionID'] = 'skp' + skipped['SessionID'].astype(str)
    replaced = pd.read_csv(path_replaced + deviation_paths[index].format(ses_amount=ses_amount), encoding_errors="ignore", on_bad_lines='error', sep=",", index_col=False)
    replaced['SessionID'] = 'rpl' + replaced['SessionID'].astype(str)
    final_df = pd.concat([swapped, skipped, replaced], axis=0)
    # shuffle the df
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    return final_df
    
res_list = []

for index,path in enumerate(deviation_paths):

    # build the dataFrame
    cvs = create_df(index)
    
    # check if column Unnamed: 0 exists
    if "Unnamed: 0" in cvs.columns : cvs = cvs.drop(columns=["Unnamed: 0"])
    # rename 
    cvs = cvs.rename(columns={'URL_FILE':'Activity'})
    
    df_trans_an = create_transition_df(cvs)
    # df_trans_an = cvs
    an_df = transition_count(df_trans_an)
    if 'SessionID' in an_df.index:
        an_df = an_df.drop(index=["SessionID"])

    ready_df = an_df.copy()
    ready_df['anomaly'] = 1
    
    
    for amount_gen in [0,50,100,250,500,750,1000,2500,5000]:
        
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
        print(path, amount_gen)
        
        acc = accuracy_score(y_true=y_test_BASE,y_pred = predictions)
        print(f'test data accuracy_score: {acc}')
        
        ball_acc = balanced_accuracy_score(y_true=y_test_BASE,y_pred = predictions)
        print(f'test data balanced_accuracy_score: {ball_acc}')
        
        _precc = precision_score(y_true=y_test_BASE,y_pred = predictions)
        print(f'test data precision_score: {_precc}')
        
        _recall = recall_score(y_true=y_test_BASE,y_pred = predictions)
        print(f'test data recall_score: {_recall}')
        
        _roc_auc_score = roc_auc_score(y_true=y_test_BASE,y_score = predictions)
        print(f'test data auc_score: {_roc_auc_score}')
        
        fpr, tpr, thresholds = roc_curve(y_true=y_test_BASE,y_score = predictions)
        
        res_dict = {}
        
        res_dict.update({"sessions": path,
                         'amount_gen': amount_gen,
                         'accuracy_score': acc,
                         'ball_acc': ball_acc,
                         'precision': _precc,
                         'recall': _recall,
                         'roc_auc_score': _roc_auc_score,
                         'fpr': fpr,
                          'tpf': tpr,
                          'thresholds': thresholds                     
                         })
        
        print()
        res_list.append(res_dict)
        
res_df = pd.DataFrame(res_list)

res_df.to_csv(f"C:/Users/krdeg/dev/ozp/all_patterns/patt.csv")