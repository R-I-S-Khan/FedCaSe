import pandas as pd
import pickle
# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('client_device_capacity.csv')

dic = {}
for col in df.columns:
	if 'Unnamed' not in col:
		dic[int(col)] = {'computation': df[col][0], 'communication': df[col][1], 'sample_size': df[col][2]}


with open('client_device_capacity', 'wb') as fp:
    pickle.dump(dic, fp)
