import numpy as np

name = 'walk_joint_pos5_from_csv'
raw_data = np.loadtxt(name + '.txt', delimiter=",")
print(raw_data.shape)

frequency = 2 # faster: > 1; lower: < 1

adjust_data = raw_data[::2, :]
print(adjust_data.shape)

# save as txt
np.savetxt(name + '_x' + str(frequency) + '.txt', adjust_data, fmt=['%-9.5f'] * adjust_data.shape[1],
           delimiter=", ")