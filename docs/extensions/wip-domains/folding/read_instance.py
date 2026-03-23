import sys

args = sys.argv[1:]
file = args[0]

with open(f'/Users/cds26/PycharmProjects/unified-planning/docs/extensions/domains/folding/instances/{file}.txt', 'r') as f:
    lines = f.readlines()

rows = lines[0]
cols = lines[1]
initial_state = lines[2]
goal_state = lines[3]

print((rows, cols))
print("---")
print(initial_state)
print("---")
print(goal_state)