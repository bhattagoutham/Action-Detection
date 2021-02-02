import pathlib
import numpy as np

def get_labels():
    
    n_classes = 4
    n_frames = 2

    data_root = pathlib.Path('../train/')
    n_imgs = len(list(data_root.glob('./*.jpeg')))
    f = open('../train/action_labels.txt')

    # trimmimg labels data to multiple of 3 (since using 3d-cnn of 3 frames)
    n_mul_3 = n_imgs-1
    n = int(n_mul_3/n_frames)


    one_hot_label_data = np.zeros((n, n_classes))


    for i in range(n+1):
        if i == 0:
            f.readline()
            continue
        temp = np.array([]).astype(int)
        
        for k in range(n_frames):
            temp = np.append(temp, int(f.readline()[0]))
        
        # finds the majority of temp array and assigns it as the action for the set of 3 frames
        cnt = np.bincount(temp)

        # shape - (13, 4)
        one_hot_label_data[i-1, np.argmax(cnt)] = 1
    
    # print(one_hot_label_data[50:60])
    return one_hot_label_data


# get_labels()

  