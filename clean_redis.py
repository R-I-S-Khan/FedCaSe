import redis
from rediscluster import RedisCluster
#startup_nodes = [{"host": "10.52.1.124", "port": "6379"}]
startup_nodes = [{"host": "10.52.0.181", "port": "6379"}]
key_id_map = RedisCluster(startup_nodes=startup_nodes)
key_id_map.flushall()
print(len(key_id_map.keys()))