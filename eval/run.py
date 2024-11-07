import os
import numpy as np

if __name__ == '__main__':
    baseFolder = "D:/UNS/"
    # sceneId = "3db0a1c8f3"
    # decimateRatio = 0.7
    # resolutionDownsample = 1

    # os.system(f'"..\\bin\\RTRenderer.exe" {baseFolder} {sceneId} {decimateRatio} {resolutionDownsample}')

    for sceneId in os.listdir(baseFolder):
        if not os.path.isdir(os.path.join(baseFolder, sceneId)):
            continue

        print(sceneId)
        for decimateRatio in np.linspace(0.1, 1, 4):
            for resolutionDownsample in [1,2]:
                os.system(f'"..\\bin\\RTRenderer.exe" {baseFolder} {sceneId} {decimateRatio} {resolutionDownsample}')
                # os.system(f"python ../../Pointersect/infer.py {baseFolder} {sceneId} {decimateRatio} {resolutionDownsample}")