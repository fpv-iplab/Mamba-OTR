import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from tools.test_net import test
from tools.train_net import train
from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.utils.actionstartend_utils import target_perframe_to_actionstartend





def main(cfg):
    isEK = "ek" in cfg.DATA.DATA_NAME.lower()
    targetPath = os.path.join(cfg.DATA.DATA_ROOT, cfg.INPUT.TARGET_PERFRAME)
    task = "start" in targetPath

    if isEK and not os.path.exists(targetPath.replace('target', 'verb')) or len(os.listdir(targetPath.replace('target', 'verb'))) == 0:
        print("EK dataset detected, converting perframe to action start/end")

        inputPath = targetPath.replace("start_", "") if task else targetPath.replace("end_", "")
        inputVerbPath = inputPath.replace('target', 'verb')
        inputNounPath = inputPath.replace('verb', 'noun')

        print("Converting verb target perframe to action start/end")
        target_perframe_to_actionstartend(inputVerbPath, targetPath.replace('target', 'verb'), type="start" if task else "end")

        print("Converting noun target perframe to action start/end")
        target_perframe_to_actionstartend(inputNounPath, targetPath.replace('target', 'noun'), type="start" if task else "end")

        print("Converting target perframe to action start/end")
        target_perframe_to_actionstartend(inputPath, targetPath, type="start" if task else "end")

    elif not os.path.exists(targetPath) or len(os.listdir(targetPath)) == 0:
        inputPath = targetPath.replace("start_", "") if task else targetPath.replace("end_", "")

        print("Converting target perframe to action start/end")
        target_perframe_to_actionstartend(inputPath, targetPath, type="start" if task else "end")

    if cfg.TRAIN:
        train(cfg)



if __name__ == "__main__":
    main(load_cfg())
