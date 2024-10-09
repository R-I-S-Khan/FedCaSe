#!/bin/bash

#tmux new -s plasma 'plasma_store -m 110000000000 -s /tmp/plasma'
screen -S redis-server /home/cc/redis-stable/src/redis-server /home/cc/redis-stable/redis.conf
#tmux new -s redis-server '/home/cc/redis-stable/src/redis-server /home/cc/redis-stable/redis.conf'
#tmux new -s redis-server '/home/redwan/redis-stable/src/redis-server /home/redwan/redis-stable/redis.conf'
