#!/bin/bash

echo 3 > /proc/sys/vm/drop_caches
sysctl vm.overcommit_memory=1
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo 1024 > /proc/sys/net/core/somaxconn
#sudo sed -i 's/net.core.somaxconn=.*/net.core.somaxconn=65535/' /etc/sysctl.conf
echo 'net.core.somaxconn=65535' >> /etc/sysctl.conf