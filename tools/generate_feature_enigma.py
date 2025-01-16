import os
import json
import lmdb
import argparse
import numpy as np




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
            data = np.array(data)
            np.save(os.path.join(out_path, f"{video}.npy"), data)
            print(f"Video {video} done")
    with open(os.path.join(os.path.dirname(out_path), "ENIGMA-51_annotations.json"), "w") as f:
        json.dump(ann_data, f)



def main(args):
    ENIGMA_VIDEO_COUNT = 51
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data path does not exist")

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






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate targets for ENIGMA-51 Dataset')
    parser.add_argument('--data-path', type=str, default=os.path.join(os.path.dirname(__file__), "data", "ENIGMA-51"))
    args = parser.parse_args()
    main(args)