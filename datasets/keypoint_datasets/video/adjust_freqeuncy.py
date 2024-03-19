import numpy as np

name = 'video10runandjump_changethin_from_csv'
raw_data = np.loadtxt(name + '.txt', delimiter=",")
print(raw_data.shape)

frequency = 0.4 # faster: > 1; lower: < 1

if frequency > 1:
    adjust_data = raw_data[::frequency, :]
    print(adjust_data.shape)
else:
    # linear interpolation
    x = np.arange(0, raw_data.shape[0], 1)
    x_freq = np.arange(0, raw_data.shape[0], frequency)
    y_interp_list = []
    for i in range(raw_data.shape[1]):
        y = raw_data[:, i]
        y_interp = np.interp(x_freq, x, y)
        y_interp_list += [y_interp]
    adjust_data = np.array(y_interp_list).T
    print(adjust_data.shape)

# save as txt
np.savetxt(name + '_x' + str(frequency) + '.txt', adjust_data, fmt=['%-9.5f'] * adjust_data.shape[1],
           delimiter=", ")