import pulp

x = [[pulp.LpVariable(f'x{i}{j}', lowBound=0, cat=pulp.LpInteger) for j in [1, 2, 3, 4, 5, 6]] for i in [1, 2, 3, 4]]
print(x)

z = [[500, 550, 630, 1000, 800, 700],
     [800, 700, 600, 950, 900, 930],
     [1000, 960, 840, 650, 600, 700],
     [1200, 1040, 980, 860, 880, 780]]
# np_z = np.array(z)
# np_z_T = np_z.T
# z = np_z_T.tolist()
# print(z)
print(pulp.lpDot(z, x))  # 不需要转置

y1 = [42, 56, 44, 39, 60, 59]
y2 = [76, 88, 96, 40]

print(pulp.lpDot(z[0], x[0]))
print([pulp.lpDot(z[i], x[i]) for i in range(4)])

m = pulp.LpProblem(sense=pulp.LpMaximize)
m += pulp.lpDot(z, x)
for i in range(4):
    m += (pulp.lpSum(x[i]) <= y2[i])

print()
for j in range(6):
    m += (pulp.lpSum([x[i][j] for i in range(4)]) <= y1[j])

print(m)
m.solve()
result = {'objective': pulp.value(m.objective), 'var': [[pulp.value(x[i][j]) for j in range(6)] for i in
                                                        range(4)]}

print(f'最大值为{result["objective"]}')
print('各变量的取值为：')
print(result['var'])
