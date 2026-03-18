import sys

args = sys.argv[1:]
file = args[0]

with open(f'/Users/cds26/PycharmProjects/unified-planning/experiments/counters/instances/{file}.txt', 'r') as f:
    lines = f.readlines()

max_int_n = lines[0]
values_counters = lines[1]
values_rate = lines[1]

print(max_int_n)
print("---")
print(values_counters)
print("---")
print(values_rate)