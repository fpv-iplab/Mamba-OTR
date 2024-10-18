import os
import json
import lmdb
import argparse
import numpy as np




def fix_fps(ann_data, out_path):
    for video in ann_data["videos"]:
        ann_data["videos"][video]["fps"] = ann_data["videos"][video]["frame_count"] / ann_data["videos"][video]["duration_seconds"]
    with open(out_path, 'w') as f:
        json.dump(ann_data, f, indent=4)


def fix_frame_annotations(ann_data, out_path, OLD_FPS=30):
    frames = list(ann_data["frame_annotations"].keys())
    for frame in frames:
        video = frame.split("_")[0]
        frame_id = int(frame.split("_")[1])

        video_fps = ann_data["videos"][video]["fps"]
        action_sec = frame_id / OLD_FPS

        new_frame_id = int(action_sec * video_fps)
        ann_data["frame_annotations"][f"{video}_{new_frame_id}"] = ann_data["frame_annotations"].pop(frame)
    with open(out_path, 'w') as f:
        json.dump(ann_data, f, indent=4)


def fix_frame_count(ann_data, env, out_path):
    with env.begin() as txn:
        cursor = txn.cursor()

        for video in ann_data['videos']:
            tot_frame_raw = int(ann_data['videos'][video]["frame_count"])
            tot_frame_actual = tot_frame_raw

            for n in range(1, tot_frame_raw):
                name = f"{video}_{n:010d}.jpg"
                frame = cursor.get(name.encode())
                if frame is None:
                    tot_frame_actual = int(name.split('.jpg')[0].split('_')[-1].lstrip("0")) - 1
                    break
            ann_data['videos'][video]["frame_count"] = tot_frame_actual
            print(f"Video {video} - Frame Count: {tot_frame_raw} -> {ann_data['videos'][video]['frame_count']}")

        with open(out_path, 'w') as f:
            json.dump(ann_data, f, indent=4)


def fix_annotations(ann_data, env, out_path):
    fix_frame_count(ann_data, env, out_path)
    fix_fps(ann_data, out_path)
    fix_frame_annotations(ann_data, out_path, OLD_FPS=30)


def generate_features(ann_data, env, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with env.begin() as txn:
        cursor = txn.cursor()
        for video in ann_data['videos']:
            data = []
            tot_frame = ann_data['videos'][video]["frame_count"]

            for n in range(1, tot_frame + 1):
                name = f"{video}_{n:010d}.jpg"
                frame = cursor.get(name.encode())
                if frame is not None:
                    data.append(np.frombuffer(frame, dtype=np.float32))
            data = np.array(data)
            np.save(os.path.join(out_path, f"{video}.npy"), data)


def generate_targets(ann_data, out_path):
    data = dict.fromkeys(ann_data["videos"].keys())
    for video in ann_data["videos"]:
        frame_count = ann_data["videos"][video]["frame_count"]
        data[video] = np.zeros((frame_count, len(list(ann_data["interaction_types"].keys())) + 1))

    for frame in ann_data["frame_annotations"]:
        video_id = frame.split("_")[0]
        frame_num = int(frame.split("_")[1])
        for interaction in ann_data["frame_annotations"][frame]["interactions"]:
            action = interaction["interaction_category"]
            data[video_id][frame_num, action + 1] = 1   # + 1 because 0 is reserved for background

    for video in data:
        indexis = np.where(~data[video].any(axis=1))[0]
        data[video][indexis, 0] = 1

    for video in data:
        np.save(os.path.join(out_path, video + ".npy"), data[video])





def main(args):
    ENIGMA_VIDEO_COUNT = 51
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist")

    ann_file_path = os.path.join(data_path, "annotations_raw.json")
    if not os.path.exists(ann_file_path):
        raise FileNotFoundError("Annotations file does not exist")

    feature_file_path = os.path.join(data_path, "raw_data")
    if not os.path.exists(feature_file_path):
        raise FileNotFoundError("Feature file does not exist")

    feature_out_path = os.path.join(data_path, "DINOv2")
    ann_out_path = os.path.join(data_path, "annotations_fix.json")
    target_path = os.path.join(data_path, "target_perframe")


    with open(ann_file_path, "r") as f:
        ann_data = json.load(f)
    env = lmdb.open(feature_file_path, readonly=True, lock=False)

    if not os.path.exists(ann_out_path):
        fix_annotations(ann_data, env, ann_out_path)

    if not os.path.exists(feature_out_path):
        os.makedirs(feature_out_path)
    if len(os.listdir(feature_out_path)) != ENIGMA_VIDEO_COUNT:
        generate_features(ann_data, env, feature_out_path)
    
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if len(os.listdir(target_path)) != ENIGMA_VIDEO_COUNT:
        generate_targets(ann_data, target_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate targets for ENIGMA-51 Dataset')
    parser.add_argument('--data-path', type=str, default=os.path.join(os.path.dirname(__file__), "..", "data",  "ENIGMA"))
    args = parser.parse_args()
    main(args)
