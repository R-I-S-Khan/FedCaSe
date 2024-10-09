import redis
from rediscluster import RedisCluster
import argparse
import os
import csv

startup_nodes = [{"host": '10.52.0.181', "port": '6379'}]
key_id_map = RedisCluster(startup_nodes=startup_nodes)
all_keys = key_id_map.keys()

k = "2599" + "_" + "591432"
return_tup = key_id_map.hgetall(k)
print("-----------------------------------")
print(return_tup)
print("-----------------------------------")
byte_image = return_tup[b'imgbyte']
print(byte_image)
print("-----------------------------------")
image_timestamp = int(return_tup[b'timestamp'].decode('utf-8'))
print(image_timestamp)
currtimestamp = 7
updated_timestamp = currtimestamp
tup_val = {"imgbyte": byte_image, "timestamp": currtimestamp}

key_id_map.hmset(k, tup_val)
print(key_id_map.exists(k))
return_tup = key_id_map.hgetall(k)
print(f"return_tup: {return_tup}")

print("-----------------------------------")
print(key_id_map.hgetall('367_81604'))

if key_id_map.exists('367_81604'):
    key_id_map.delete('367_81604')

print(key_id_map.hgetall('367_81604'))


