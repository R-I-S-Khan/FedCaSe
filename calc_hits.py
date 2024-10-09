import argparse
import os
import csv
import subprocess

# Usage: python calc_hits.py --file femnist_logging
parser = argparse.ArgumentParser(description="Calculate client_hits in a session")
parser.add_argument('--file', default='femnist_logging', type=str, help='log file of the run')
args = parser.parse_args()

filename = args.file
command = f"cat {filename} | grep 'Training of (CLIENT:' > hitmissinfo"
subprocess.run(command, shell=True)

total_client_hits = 0
total_client_misses = 0
samples_per_client = 160 #this is the number of samples that each client will train on each round. 

#count = 0
with open('hitmissinfo', 'r') as file:
    for line in file:
        if "from_cache:" in line:
            # split from "from_cache"
            partitions = line.split("from_cache:")

            # check if such partitions exist, should be equal or greater than 2
            if len(partitions) >= 2:
                # Extract the sample number after "from_cache". need to clean it.
                samples_from_cache = partitions[1].split(',')[0].strip()

                #print(samples_from_cache)
                # Check if the value is integer or if we made a mistake.
                if samples_from_cache.isdigit():
                    samples_from_cache = int(samples_from_cache)
                else:
                    print(f"Invalid value in: {line}")
                    continue

                client_hits = samples_from_cache
                client_misses = max(0, samples_per_client - samples_from_cache)

                # Add to the total client_hits and client_misses
                total_client_hits += client_hits
                total_client_misses += client_misses
                #count+=1
                #if count == 5:
                #    break

# Calculate hit rate and miss rate
total_hit_rate = total_client_hits / (total_client_hits + total_client_misses)
total_miss_rate = total_client_misses / (total_client_hits + total_client_misses)

print(f"total_client_hits: {total_client_hits}")
print(f"total_client_misses: {total_client_misses}")
print(f"total_hit_rate: {total_hit_rate}")
print(f"total_miss_rate: {total_miss_rate}")
