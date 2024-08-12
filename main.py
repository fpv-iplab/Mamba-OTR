import os
import multiprocessing

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import warnings
warnings.filterwarnings("ignore")

from tools.test_net import test
from tools.train_net import train
from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.utils.actionstartend_utils import target_perframe_to_actionstartend





def main(cfg):
    cpu_count = multiprocessing.cpu_count()
    cfg.DATA_LOADER.NUM_WORKERS = max(cfg.DATA_LOADER.NUM_WORKERS, cpu_count)
    print("Using {} workers for data loading".format(cfg.DATA_LOADER.NUM_WORKERS))

    isEK = "ek" in cfg.DATA.DATA_NAME.lower()
    targetPath = os.path.join(cfg.DATA.DATA_ROOT, cfg.INPUT.TARGET_PERFRAME)
    task = "start" in targetPath
    inputPath = targetPath.replace("start_" if task else "end_", "")

    if isEK:
        #* Convert EK ACTION TARGET perframe to action start/end
        if not os.path.exists(targetPath) or len(os.listdir(targetPath)) == 0:
            print("Converting EK target perframe to action start/end")
            target_perframe_to_actionstartend(inputPath, targetPath, type="start" if task else "end")

        #* Convert EK VERB target perframe to action start/end
        if not os.path.exists(targetPath.replace('target', 'verb')) or len(os.listdir(targetPath.replace('target', 'verb'))) == 0:
            print("Converting EK verb target perframe to action start/end")

            inputVerbPath = inputPath.replace('target', 'verb')
            outputVerbPath = targetPath.replace('target', 'verb')
            target_perframe_to_actionstartend(inputVerbPath, outputVerbPath, type="start" if task else "end")

        #* Convert EK NOUN target perframe to action start/end
        if not os.path.exists(targetPath.replace('target', 'noun')) or len(os.listdir(targetPath.replace('target', 'noun'))) == 0:
            print("Converting EK noun target perframe to action start/end")

            inputNounPath = inputPath.replace('target', 'noun')
            outputNounPath = targetPath.replace('target', 'noun')
            target_perframe_to_actionstartend(inputNounPath, outputNounPath, type="start" if task else "end")
    else:
        if not os.path.exists(targetPath) or len(os.listdir(targetPath)) == 0:
            print("Converting THUMOS target perframe to action start/end")

            inputPath = targetPath.replace("start_", "") if task else targetPath.replace("end_", "")
            target_perframe_to_actionstartend(inputPath, targetPath, type="start" if task else "end")

    if cfg.TRAIN:
        print("Training model")
        train(cfg)



if __name__ == "__main__":
    main(load_cfg())
