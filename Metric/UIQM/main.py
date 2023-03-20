import os
import numpy as np
import cv2
import natsort
from uqim_utils import getUIQM


if __name__ == '__main__':
    pass
folder_dcp ="E://desktop//recent_files//graduate_design//myfiles//03reading_paper//mathematic model//DCP//prj//prj2//OutputImages"
folder_udcp = "E://desktop//recent_files//graduate_design//myfiles//03reading_paper//mathematic model//UDCP//UDCPprj//prj3//OutputDepth"
folder_dl = "E://desktop//recent_files//graduate_design//myfiles//03reading_paper//deep learning//waterGan//prj_restore//prj7g//test_result1"
folder_he = "E://desktop//recent_files//graduate_design//myfiles//03reading_paper//optimization-based//histogram//prj1//result"
folder_ori = "E://desktop//recent_files//graduate_design//myfiles//03reading_paper//optimization-based//histogram//prj1//img"
folder = folder_dcp
files = os.listdir(folder)
files =  natsort.natsorted(files)
values=list([])
for i in range(len(files)):
    file = files[i]
    filepath = folder + "/" + file
    prefix = file.split('.')[0]

    if os.path.isfile(filepath):
        img = cv2.imread(folder +'/' + file)
        newvalue=getUIQM(img)
        values.append(newvalue)
print("folder_dcp",np.mean(np.array(values)))
folder = folder_ori
files = os.listdir(folder)
files =  natsort.natsorted(files)
values=list([])
for i in range(len(files)):
    file = files[i]
    filepath = folder + "/" + file
    prefix = file.split('.')[0]

    if os.path.isfile(filepath):
        img = cv2.imread(folder +'/' + file)
        newvalue=getUIQM(img)
        values.append(newvalue)
print("original",np.mean(np.array(values)))

folder = folder_udcp
files = os.listdir(folder)
files =  natsort.natsorted(files)
values=[]
for i in range(len(files)):
    file = files[i]
    filepath = folder + "/" + file
    prefix = file.split('.')[0]

    if os.path.isfile(filepath):
        img = cv2.imread(folder +'/' + file)
        values.append(getUIQM(img))
print("folder_udcp",np.mean(np.array(values)))

folder = folder_dl
files = os.listdir(folder)
files =  natsort.natsorted(files)
values=[]
for i in range(len(files)):
    file = files[i]
    filepath = folder + "/" + file
    prefix = file.split('.')[0]

    if os.path.isfile(filepath):
        img = cv2.imread(folder +'/' + file)
        values.append(getUIQM(img))
print("folder_dl",np.mean(np.array(values)))

folder = folder_he
files = os.listdir(folder)
files =  natsort.natsorted(files)
values=[]
for i in range(len(files)):
    file = files[i]
    filepath = folder + "/" + file
    prefix = file.split('.')[0]

    if os.path.isfile(filepath):
        img = cv2.imread(folder +'/' + file)
        values.append(getUIQM(img))
print("folder_he",np.mean(np.array(values)))






