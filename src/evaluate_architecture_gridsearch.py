
with open('best_params.txt') as f:
    data1 = f.read()
with open('best_params_2.txt') as f:
    data2 = f.read()

def split_trials(data, data_index, with_lr):
    results = list()
    for d in data:
        d = d.split()
        num_layers = 23 if with_lr else 21
        results.append((round(float(d[4]), 6), int(d[num_layers]), data_index, list(map(int, [d[11], d[13], d[15], d[17], d[19]]))))
    for i in range(len(results)):
        results[i] = (*results[i][:3], [*results[i][3][:results[i][1]], results[i][3][-1]])
    return results

data1 = data1.split('[Trial summary]')[1:]
data2 = data2.split('[Trial summary]')[1:]
data1 = split_trials(data1, 1, True)
data2 = split_trials(data2, 2, False)

data = sorted([*data1, *data2], reverse=True, key=lambda x: x[0])
for i,d in enumerate(data):
    print(f'{i+1:>2}:  {round(d[0], 6):>6}\t{d[2]}   {d[3]}')