import redis
from rediscluster import RedisCluster
import argparse
import os
import csv

startup_nodes = [{"host": '10.52.0.181', "port": '6379'}]
key_id_map = RedisCluster(startup_nodes=startup_nodes)
all_keys = key_id_map.keys()

key1 = all_keys[0]
key2 = all_keys[1]

for key in all_keys:
	return_tup = key_id_map.hgetall(key)
	byte_image = return_tup[b'imgbyte']
	image_timestamp = int(return_tup[b'timestamp'].decode('utf-8'))

	if key == key1 or key == key2:
		print(f"key: {key}, earlier version: {return_tup}")
	currtimestamp = 0
	tup_val = {"imgbyte": byte_image, "timestamp": currtimestamp}
	key_id_map.hmset(key, tup_val)

print("---------------CHECK-----------------------")
return_tup = key_id_map.hgetall(key1)
print(f"key: {key1}, updated version: {return_tup}")
return_tup = key_id_map.hgetall(key2)
print(f"key: {key2}, updated version: {return_tup}")
