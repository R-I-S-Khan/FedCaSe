import redis
from rediscluster import RedisCluster
import argparse
import os
import csv

parser = argparse.ArgumentParser(description="Check keys of a client")
parser.add_argument('--start_id', type=str, help='start of clientId list')
parser.add_argument('--end_id', type=str, help='start of clientId list')
parser.add_argument('--file', type=str, help='start of clientId list')
args = parser.parse_args()

start = int(args.start_id)
end = int(args.end_id)
filename = args.file

path = os.path.join('/home/cc/', filename + '.csv')

#startup_nodes = [{"host": '10.52.1.124', "port": '6379'}]
startup_nodes = [{"host": '10.52.0.181', "port": '6379'}]
key_id_map = RedisCluster(startup_nodes=startup_nodes)
all_keys = key_id_map.keys()
all_keys_string = [key.decode('ascii') for key in all_keys]


def write_mapping_in_csv(new_mapping, file_path):

		with open(file_path, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['client', 'keys'])  # Write header
			writer.writerows(new_mapping.items())

mapping = {}
for i in range(start, end+1):
	prefix = str(i) + '_'
	matching_keys = [key for key in all_keys_string if key.startswith(prefix)]
	print(f"clientID: {i}, len(matching_keys): {len(matching_keys)}")
	mapping[prefix] = len(matching_keys)
	#print(f"matching_keys: {matching_keys}")

write_mapping_in_csv(mapping, path)
#prefix = args.client_id + '_'#1210_' #964

# matching_keys = [key for key in all_keys_string if key.startswith(prefix)]
# print(f"len(matching_keys): {len(matching_keys)}")
# print(f"matching_keys: {matching_keys}")