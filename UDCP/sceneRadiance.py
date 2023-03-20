import numpy as np

def sceneRadianceRGB(img, transmission, AtomsphericLight, Ax, Ay):
    AtomsphericLight = np.array(AtomsphericLight)
    img = np.float64(img)
    sceneRadiance = np.zeros(img.shape)

    transmission = np.clip(transmission, 0.2, 0.9)
    for i in range(0, 3):
        sceneRadiance[:, :, i] = (img[:, :, i] - AtomsphericLight[i]) / transmission  + AtomsphericLight[i]
    # for i in range(0,10):
    #     for j in range(0,10):
    #         if(0<Ax-5+i<img.shape[0] and 0<Ay-5+j<img.shape[1]):
    #             sceneRadiance[Ax-5+i, Ay-5+j, 0] = 255
    #             sceneRadiance[Ax-5+i, Ay-5+j, 1] = 255
    #             sceneRadiance[Ax-5+i, Ay-5+j, 2] = 0
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance



