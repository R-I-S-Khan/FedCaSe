import subprocess
import csv

command = "find ~/client/ -type f -name '*numcall*' -exec awk -F ',' 'FNR == 1 { next } { print $0 }' {} + | sort -t ',' -k2 -n > client_calls_sorted.csv"

subprocess.run(command, shell=True)

file_in = "client_calls_sorted.csv"
file_out = "count_numcalls.csv"

d = {}

with open(file_in, "r") as file:
	csv_reader = csv.reader(file)
	next(csv_reader) 
	for row in csv_reader:
		idname = str(row[0])
		client_numcall_count = float(row[1])
		d[client_numcall_count] = 1 + d.get(client_numcall_count, 0)

with open(file_out, "w", newline="") as file:
	csv_writer = csv.writer(file)
	for client_numcall_count, occurr in d.items():
		csv_writer.writerow([client_numcall_count, occurr])
