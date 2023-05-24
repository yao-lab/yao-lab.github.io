import numpy as np

# read txt file
with open('snp500.txt', 'r') as f:
    lines = f.readlines()
X_list = [list(map(float, line.strip().split())) for line in lines]
# Qa
Y_list = np.log(X_list)
print(Y_list)

# Qb
Y_jump = []
for i in range(len(Y_list)-1):
    Y_jump.append(Y_list[i+1] - Y_list[i])
print(len(Y_jump))

# Qc
Y_c = []
for i in range(len(Y_jump[0])):
    temp_i = []
    for j in range(len(Y_jump[0])):
        temp_i.append(sum([a * b for a, b in zip(Y_jump[i], Y_jump[j])]))
    Y_c.append(temp_i)
print(len(Y_c))

# Qd
eigenvalues, eigenvectors = np.linalg.eig(np.array(Y_c))
print(sorted(eigenvalues, reverse=True))


