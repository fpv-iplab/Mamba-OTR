import os
import numpy as np


def thumos_target_perframe_to_actionstart(featPath: str, outputPath: str):
    for file in os.listdir(featPath):
        data = np.load(os.path.join(featPath, file))
        for i in range(data.shape[0]):
            for j in range(1, data[i].shape[0]):
                if data[i, j] == 1:
                    if i < data.shape[0] - 1:
                        k = i + 1
                    else:
                        continue
                    while k < data.shape[0] - 1 and data[k, j] == 1 and data[k + 1, j] == 1:
                        data[k, j] = 0
                        data[k, 0] = 1 #! set middle frames of action as background
                        k = k + 1
                    if k < data.shape[0] - 1 and data[k, j] == 1 and data[k + 1, j] == 0: # remove frame of end action
                        data[k, j] = 0
                        data[k, 0] = 1 #! count as background
        np.save(os.path.join(outputPath, file), data)
