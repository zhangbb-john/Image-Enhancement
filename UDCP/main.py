import os
import numpy as np
import cv2
import natsort

from RefinedTramsmission import Refinedtransmission
from getAtomsphericLight import getAtomsphericLight
from getGbDarkChannel import getDarkChannel
from getTM import getTransmission
from sceneRadiance import sceneRadianceRGB

np.seterr(over='ignore')
if __name__ == '__main__':
    pass

# folder = "C:/Users/Administrator/Desktop/UnderwaterImageEnhancement/Physical/UDCP"
folder = "E://desktop//recent_files//graduate_design//myfiles//03reading_paper//metrics//ssim//img_select//result"
path = folder
files = os.listdir(path)
files =  natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    filepath = path + "/" + file
    prefix = file.split('.')[0]
    if os.path.isfile(filepath):
        print('********    file   ********',file)
        img = cv2.imread(folder +'/' + file)


        print('img',img)

        blockSize = 9
        GB_Darkchannel = getDarkChannel(img, blockSize)
        AtomsphericLight,Ax,Ay = getAtomsphericLight(GB_Darkchannel, img)

        print('AtomsphericLight', AtomsphericLight)
        # print('img/AtomsphericLight', img/AtomsphericLight)

        # AtomsphericLight = [231, 171, 60]

        transmission = getTransmission(img, AtomsphericLight, blockSize)

        # cv2.imwrite('OutputImages/' + prefix + '_UDCP_Map.jpg', np.uint8(transmission))

        transmission = Refinedtransmission(transmission, img)
        sceneRadiance = sceneRadianceRGB(img, transmission, AtomsphericLight,Ax,Ay)
        # print('AtomsphericLight',AtomsphericLight)



        cv2.imwrite('OutputImages/' + prefix + '_UDCP_TM.jpg', np.uint8(transmission* 255))
        cv2.imwrite('OutputDepth/' + prefix + '_UDCP.jpg', sceneRadiance)


