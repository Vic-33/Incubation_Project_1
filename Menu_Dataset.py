import pandas as pd

df = pd.read_csv('raw_menu.csv')

# Collecting all columns
remove_columns = df.columns.tolist()

# Removing required columns from list
for i in range(4):
    remove_columns.pop(0)

# Removing unnecessary columns
df.drop(remove_columns, axis=1, inplace=True)

# Exporting as cvs file
df.to_csv('Menu.csv', index=False)

print(df)