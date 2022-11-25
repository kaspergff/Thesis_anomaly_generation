def find_paths(_df:pd.DataFrame,len:int = 3):
  df = _df.copy()
  col = df.columns.values
  row  = df.index.values
  s  = row.size
  res = []
  while s > 0:
    current_row = df.iloc[row.size - s]
    i = 0
    while i < current_row.size:
      current_col = df.columns[i]
      chance = current_row.iloc[i]
      if chance > 0:
        a = current_row.name
        b = current_col
        
        if (df.index == b).any() :
          _r = df.loc[b]
          index = 0
          for r in _r:
            if r > 0:
              res.append({'a':a,'b':b,'c':col.item(index),"a-b-c":r*chance,"a-b":chance,"b-c":r})  
            index += 1        
            
      i += 1
    s -= 1

  return res


# paths = find_paths(probability_matrix)

# remove the probabilities to go go back to the same activity and adjust the other probabilities in the row
def remove_self_loops(df):
  df = df.copy()
  for rowIndex,rowValues in df.iterrows():
   if rowIndex in rowValues.index:
    if df.loc[rowIndex, rowIndex] > 0:
        df.loc[rowIndex, rowIndex] = 0
        df.loc[rowIndex, :] = df.loc[rowIndex, :] / df.loc[rowIndex, :].sum()
  return df
    
    
    # find transitions with high probability that are lower than 1 and  where from and to are not the same
def transition_no_circles(df:pd.DataFrame, threshold:float = 0.81):
  df = df.copy()
  res = []
  for col in df.columns.values:
    if col == 'end_session':
      continue
    row = df.loc[col,:].sort_values(ascending=False)
    for index, value in row.items():
      if value < 1 and value > threshold and col != index:
        res.append({"from":col, "to":index, "probability":value})
  return res

# test = transition_no_circles(probability_matrix, 0.2)
# test.sort(key=lambda x: x["value"], reverse=True)
# test


# a session is anomalous if it contains a transition from Activity_2 to Activity that are in the same row in the swapping_patterns DataFrame
# check if a session in the probability matrix is an anomalous session
def check_session(df: pd.DataFrame, swapping_patterns: pd.DataFrame):
  df = df.copy()
  swapping_patterns = swapping_patterns.copy()
  res = []
  for sessionID, session in df.groupby("SessionID"):
    activities = session['Activity'].values
    # loop over the activities in the session
    for i in range(len(activities)):
      # if the activity is in the swapping_patterns["Activity_2"] column
      if activities[i] in swapping_patterns["second"].values:
        # check if the next activity is on the same row in the swapping_patterns["c"] column
        if i + 1 < len(activities) and activities[i+1] in swapping_patterns[swapping_patterns["second"] == activities[i]]["first"].values:
          # if it is add the sessionID to the list
          res.append(sessionID)
          # we dont need to check the rest of the session
          break
  return res




# Function to find the skipping patterns in a session.
# It checks for each activity if the next activity is in a skipping pattern as a c activity.
# If it is then it checks if the current activity is in a skipping pattern as a a activity.
# If it is then it checks if the activities are in the same row in the skipping pattern DataFrame.
# it returns a list with the sessionIDs of sessions where a swapping pattern is found

def find_anomalous_sessions(df:pd.DataFrame,skipping_patterns:pd.DataFrame) -> list[int]:
  sessions_df = df.copy()
  skipping_patterns = skipping_patterns.copy()
  # Create a new column with the consecutive activity
  sessions_df["Consecutive_Activity"] = sessions_df.groupby("SessionID")["Activity"].shift(periods=-1)  
  print(len(sessions_df))
  
  # # Create a column that checks if the Consecutive_Activity is in the skipping_patterns DataFrame in the column c
  # sessions_df['Consecutive_Activity_is_in_c'] = sessions_df['Consecutive_Activity'].isin(skipping_patterns['c'])
  
  # # Remove rows where the Consecutive_Activity is not in the skipping_patterns DataFrame in the column c
  # filter_Consecutive_Activity = sessions_df[sessions_df['Consecutive_Activity_is_in_c'] == True].copy()
  # print(len(filter_Consecutive_Activity))
  
  # # Create a column that checks if the Activity in the skipping_patterns DataFrame in columns a
  # filter_Consecutive_Activity['Activity_is_in_a'] = filter_Consecutive_Activity['Activity'].isin(skipping_patterns['a'])
  
  # # filter on 'Activity_is_in_a'
  # filter_Activity_is_in_a = filter_Consecutive_Activity[filter_Consecutive_Activity['Activity_is_in_a'] == True].copy()
  
  # print(len(filter_Activity_is_in_a))
  #  Create a column called anomaly that is True if:
  #  1. There is a row in skipping_patterns where the c column is equal to the Consecutive_Activity 
  #  and 
  #  2. the a column is equal to the Activity
  # filter_Activity_is_in_a['anomaly'] = filter_Activity_is_in_a.apply(lambda x: True if x['Activity'] in skipping_patterns[skipping_patterns['c'] == x['Consecutive_Activity']]['a'].values else False, axis=1)
  
  # Vectorized version of the above apply function
  to_merge = skipping_patterns[['a','c']].copy()
  merged = pd.merge(sessions_df, to_merge, left_on=['Activity','Consecutive_Activity'], right_on=['a','c'], how='left',indicator='Anomaly')
  merged['Anomaly'] = np.where(merged.Anomaly == 'both', True, False)
  
  
  
  
  anomalous_session = merged[merged['Anomaly'] == True]
  # create a list with all SessionIDs that have an anomaly
  anomaly_sessions = anomalous_session['SessionID'].unique()
  
  return anomaly_sessions


# Function to find the swapping patterns in a session.
# It checks for each activity if the next activity is in a swapping pattern as a first activity.
# If it is then it checks if the current activity is the second activity in the swapping pattern where the next activity is the first activity.
# it returns a list with the sessionIDs of sessions where a swapping pattern is found

def find_anomalous_sessions(df:pd.DataFrame,swapping_patterns:pd.DataFrame) -> list[int]:
  sessions_df = df.copy()
  swapping_patterns = swapping_patterns.copy()
  # Create a new column with the consecutive activity
  sessions_df["Consecutive_Activity"] = sessions_df.groupby("SessionID")["Activity"].shift(periods=-1)
  # Create a column that checks if the Consecutive_Activity is in the swapping_patterns DataFrame in the column first
  sessions_df['Consecutive_Activity_is_in_first'] = sessions_df['Consecutive_Activity'].isin(swapping_patterns['first'])
  # Remove rows where the Consecutive_Activity is not in the swapping_patterns DataFrame in the column first
  filter_Consecutive_Activity = sessions_df[sessions_df['Consecutive_Activity_is_in_first'] == True].copy()
  # Create a column that checks if the Activity in the swapping_patterns DataFrame in columns second
  filter_Consecutive_Activity['Activity_is_in_second'] = filter_Consecutive_Activity['Activity'].isin(swapping_patterns['second'])
  # filter on 'Activity_is_in_second'
  filter_Activity_is_in_second = filter_Consecutive_Activity[filter_Consecutive_Activity['Activity_is_in_second'] == True].copy()
  print(len(filter_Activity_is_in_second))
  #  Create a column called anomaly that is True if:
  #  1. There is a row in swapping_patterns where the first column is equal to the Consecutive_Activity 
  #  and 
  #  2. the second column is equal to the Activity
  filter_Activity_is_in_second['anomaly'] = filter_Activity_is_in_second.apply(lambda x: True if x['Activity'] in swapping_patterns[swapping_patterns['first'] == x['Consecutive_Activity']]['second'].values else False, axis=1)
  anomalous_session = filter_Activity_is_in_second[filter_Activity_is_in_second['anomaly'] == True]
  # create a list with all SessionIDs that have an anomaly
  anomaly_sessions = anomalous_session['SessionID'].unique()
  return anomaly_sessions


def skip_event(_df:pd.DataFrame,event:str,deviation_rate):
  df = _df.copy()
  # print(f"Decrease the probability to reach event: {event} with deviation rate: {deviation_rate}")
  old_probability = df.loc[:,event]
  deviation_rate = deviation_rate / 100
  
  # decrease the probability
  for i,prob in old_probability.items():
    # skip row of event
    if i == event: continue
    
    # start changing the prob 
    if prob > 0:
      decrease = prob * (1-deviation_rate)
      df.loc[i,event] -= prob - decrease      
      for _i,_prob in df.loc[i,:].items():
        df.loc[i,_i] += (prob - decrease) * df.loc[event,_i]
  
  
  # change row of event
  if df.loc[event,event] > 0:
    decrease = df.loc[event,event] * (1 - deviation_rate)
    original_value = df.loc[event,event]
    count = df.loc[event]
    count = count[count > 0].__len__() - 1
    
    for i,prob in  df.loc[event,:].items():
      if df.loc[event,i] > 0:
        # Circle case
        if i == event: 
          df.loc[event,i] -= prob - decrease
        else: df.loc[event,i] += (original_value - decrease) / count 
      
  return df