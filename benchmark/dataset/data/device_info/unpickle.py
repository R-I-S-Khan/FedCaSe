import pickle as pkl
import pandas as pd

#ob = {}
with open("client_device_capacity", "rb") as f:
    object = pkl.load(f)

ob = object.get(2)
print(type(object))
print(type(ob))
comp = ob['computation']
comm = ob['communication']
size = ob['sample_size']

print(f"comp : {comp}, comm: {comm}, size: {size}")
df = pd.DataFrame(object)
df.to_csv(r'client_device_capacity_again.csv')
