import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from tools.test_net import test
from tools.train_net import train
from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.utils.actionstartend_utils import thumos_target_perframe_to_actionstartend





def main(cfg):
    targetPath = os.path.join(cfg.DATA.DATA_ROOT, cfg.INPUT.TARGET_PERFRAME)
    if not os.path.exists(targetPath) or len(os.listdir(targetPath)) == 0:
        task = "start" in targetPath

        inputPath = targetPath.replace("start_", "") if task else targetPath.replace("end_", "")
        thumos_target_perframe_to_actionstartend(inputPath, targetPath, type="start" if task else "end")

    if cfg.TRAIN:
        train(cfg)



if __name__ == "__main__":
    main(load_cfg())
