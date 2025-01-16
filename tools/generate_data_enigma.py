import os
import json
import lmdb
import argparse
import numpy as np



def generate_targets(ann_data, out_path):
    data = dict.fromkeys(ann_data["videos"].keys())
    for video in ann_data["videos"]:
        frame_count = ann_data["videos"][video]["frame_count"]
        data[video] = np.zeros((frame_count, len(list(ann_data["interaction_types"].keys())) + 1))
    for frame in ann_data["frame_annotations"]:
        video_id = frame.split("_")[0]
        frame_num = int(frame.split("_")[1]) - 1 # -1 because frame starts from 1
        for interaction in ann_data["frame_annotations"][frame]["interactions"]:
            action = interaction["interaction_category"]
            try:
                data[video_id][frame_num, action + 1] = 1   # + 1 because 0 is reserved for background
            except IndexError:
                print("Annotation error in video: ", video_id, " frame: ", frame_num)
                continue
    for video in data:
        indexis = np.where(~data[video].any(axis=1))[0]
        data[video][indexis, 0] = 1
    for video in data:
        np.save(os.path.join(out_path, video + ".npy"), data[video])



def generate_features(ann_data, env, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with env.begin() as txn:
        cursor = txn.cursor()
        for video in ann_data['videos']:
            data = []
            tot_frame = int(ann_data['videos'][video]["frame_count"])
            for n in range(1, tot_frame + 1):
                name = f"{video}_{n:010d}.jpg"
                frame = cursor.get(name.encode())
                if frame is not None:
                    data.append(np.frombuffer(frame, dtype=np.float32))
                else:
                    ann_data['videos'][video]["frame_count"] = n - 1
                    ann_data["videos"][video]["duration_seconds"] = (n - 1) / ann_data["videos"][video]["fps"]
                    break
            data = np.array(data)
            np.save(os.path.join(out_path, f"{video}.npy"), data)
            print(f"Video {video} done")
    with open(os.path.join(os.path.dirname(out_path), "ENIGMA-51_annotations.json"), "w") as f:
        json.dump(ann_data, f)



def main(args):
    ENIGMA_VIDEO_COUNT = 51 - 2 # 2 videos are missing
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data path does not exist")

    target_path = os.path.join(data_path, "target_end_perframe")
    ann_file_path = os.path.join(data_path, "ENIGMA-51_annotations.json")
    if not os.path.exists(ann_file_path):
        raise FileNotFoundError("Annotations file does not exist")

    feature_out_path = os.path.join(data_path, "DINOv2")
    feature_file_path = os.path.join(data_path, "raw_data")
    if not os.path.exists(feature_file_path):
        raise FileNotFoundError("Feature file does not exist")

    with open(ann_file_path, "r") as f:
        ann_data = json.load(f)
    env = lmdb.open(feature_file_path, readonly=True, lock=False)

    if not os.path.exists(feature_out_path):
        os.makedirs(feature_out_path)
    if len(os.listdir(feature_out_path)) != ENIGMA_VIDEO_COUNT:
        generate_features(ann_data, env, feature_out_path)

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if len(os.listdir(target_path)) != 1:
        generate_targets(ann_data, target_path)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate targets for ENIGMA-51 Dataset')
    parser.add_argument('--data-path', type=str, default=os.path.join(os.path.dirname(__file__), "data", "ENIGMA-51"))
    args = parser.parse_args()
    main(args)