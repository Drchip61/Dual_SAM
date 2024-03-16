import cv2
import numpy as np
import os
import shutil

# if the masks are incorrect, then use cv2.threshold to process the images
refile = 'test_usod_final'
outPath = 'test_usod_final1'

if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)
name = os.listdir(refile)
print(len(name))
for i in range(len(name)):
    label_file = os.path.join(refile, name[i])
    a = cv2.imread(label_file, 0)
    #t = np.mean(a)
    b, photo = cv2.threshold(a,65, 255, cv2.THRESH_BINARY)
    try:
        cv2.imwrite(outPath + '/' + name[i], photo)
    except:
        print(name)
