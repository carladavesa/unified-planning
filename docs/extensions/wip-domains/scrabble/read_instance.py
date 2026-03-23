timeout_limit = 3600

#args = sys.argv[1:]
#file = args[0]

with open('/Users/cds26/PycharmProjects/unified-planning/experiments/sokoban/probs/i_1.txt', 'r') as f:
    lines = f.readlines()

matrix = [list(line.rstrip('\n')) for line in lines]
initial_state = {}
defined_positions = []
goal_positions = []
undefined_positions = []

row_limits = {}
col_limits = {}
# Row limits
for i, row in enumerate(matrix[1:-1]):
    if all(x == '#' for x in row) or '#' not in row:
        row_limits[i] = (0, len(row)-3)
    else:
        start = row.index('#')
        while row[start] == '#':
            start += 1
        end = len(row) - 1
        while row[end] == '#':
            end -= 1
        row_limits[i] = (start-1, end-1)

# Column limits
columns = []
num_cols = max(len(row) for row in matrix)
padded_matrix = [list(''.join(row).ljust(num_cols)) for row in matrix]
for col_idx in range(num_cols):
    column = [row[col_idx] for row in padded_matrix]
    columns.append(column)
for i, col in enumerate(columns[1:-1]):
    if all(x == '#' for x in col) or '#' not in col:
        col_limits[i] = (0, len(col)-3)
    else:
        start = col.index('#')
        while col[start] == '#':
            start += 1
        end = len(col) - 1 - col[::-1].index('#')
        while col[end] == '#':
            end -= 1
        col_limits[i] = (start-1, end-1)

for ri, row in enumerate(matrix[1:-1]):
    for ci, cell in enumerate(row[1:-1]):
        if cell != '#' and (ri >= col_limits[ci][0] and ri <= col_limits[ci][1]) and (ci >= row_limits[ri][0] and ci <= row_limits[ri][1]):
            defined_positions.append((ri, ci))
            if cell == '@':
                initial_state[(ri, ci)] = 'P' # Person
            elif cell == '$':
                initial_state[(ri, ci)] = 'B' # Box
            elif cell == '.':
                goal_positions.append((ri, ci))
            elif cell == '*':
                goal_positions.append((ri, ci))
                initial_state[(ri, ci)] = 'B' # Box

rows = max(x[0] for x in defined_positions) + 1
columns = max(x[1] for x in defined_positions) + 1
all_positions = {(f, c) for f in range(rows) for c in range(columns)}

print(initial_state)
print("---")
print(sorted(all_positions - set(defined_positions)))
print("---")
print(goal_positions)
print("---")
print(rows)
print("---")
print(columns)