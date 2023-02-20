import glob
import os
import random

all_files = glob.glob('./doc3D/img/1/*.png')
random.shuffle(all_files)

train_files = all_files[:int(len(all_files)*0.8)]
val_files = all_files[int(len(all_files)*0.8):]

# for overfitting test
# train_files = [train_files[0] for _ in range(len(train_files))]
# val_files = [train_files[0] for _ in range(len(val_files))]

with open('doc3D/train.txt', 'w') as f:
    for file_ in train_files:
        content = '1/'+os.path.basename(file_)[:-4]
        print(content)
        f.write(content)
        f.write('\n')

with open('doc3D/val.txt', 'w') as f:
    for file_ in val_files:
        content = '1/'+os.path.basename(file_)[:-4]
        print(content)
        f.write(content)
        f.write('\n')
