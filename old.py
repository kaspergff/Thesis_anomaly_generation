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
    