import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from tools.test_net import test
from tools.train_net import train
from rekognition_online_action_detection.utils.parser import load_cfg





def main(cfg):
    if cfg.TRAIN:
        train(cfg)
    else:
        test(cfg)




if __name__ == "__main__":
    main(load_cfg())