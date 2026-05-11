import numpy as np
import argparse
from sklearn_extra.cluster import KMedoids
import os
import json
import cv2

def farthest_point_sampling(X, n_samples):
    selected = [0]
    distances = np.linalg.norm(X - X[selected[0]], axis=1)
    for _ in range(1, n_samples):
        idx = np.argmax(distances)
        selected.append(idx)
        distances = np.minimum(distances, np.linalg.norm(X - X[idx], axis=1))
    return np.array(selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=50, help="Number of training frames to select")
    parser.add_argument("--sample", type=str, default="165", help="Sample identifier")
    parser.add_argument("--method", type=str, default="fps", help="Selection method: kmedoids or fps")
    parser.add_argument("--test", action="store_true", help="Whether to run in test set")
    args = parser.parse_args()

    # get paths
    base_path = os.path.join("/home/yanp/HeadAvatar/data", args.sample)
    if not args.test:
        raise RuntimeError("Do not change train_ids!")
        # transforms_path = os.path.join(base_path, "UNION", "transforms_train.json")
        # show_image_path = os.path.join(base_path, "train_images.png")
        # ids_path = os.path.join(base_path, "train_ids.txt")
    else:
        transforms_path = os.path.join(base_path, "UNION", "transforms_test.json")
        show_image_path = os.path.join(base_path, "test_images.png")
        ids_path = os.path.join(base_path, "test_ids.txt")
    features = []
    image_paths = []
    timesteps = []
    with open(transforms_path) as f:
        transforms = json.load(f)
        data = transforms["frames"]
        for frame in data:
            timestep = frame["timestep_index"]
            camera_id = frame["camera_index"]
            if camera_id != 0:
                continue
            flame_path = os.path.join(base_path, "UNION", frame["flame_param_path"])
            image_path = os.path.join(base_path, "UNION", frame["file_path"])
            image_name = os.path.basename(image_path).split("_")[0] + "_08.png"
            image_path = os.path.join(os.path.dirname(image_path), image_name)
            flame_params = np.load(flame_path, allow_pickle=True)
            expr_params = flame_params["expr"]
            rotation = flame_params["rotation"]
            neck_pose = flame_params["neck_pose"]
            jaw_pose = flame_params["jaw_pose"]
            eyes_pose = flame_params["eyes_pose"]
            pose = np.concatenate([rotation, neck_pose, jaw_pose, eyes_pose], axis=1)
            feature = np.concatenate([expr_params, pose], axis=1)
            features.append(feature)
            image_paths.append(image_path)
            timesteps.append(timestep)
    
    if args.method == "kmedoids":      
        features = np.concatenate(features, axis=0)
        cluster = KMedoids(n_clusters=args.num_frames, metric="euclidean", random_state=0).fit(features)
        idxs = cluster.medoid_indices_
    elif args.method == "fps":
        idxs = farthest_point_sampling(np.concatenate(features, axis=0), args.num_frames)
    else:
        raise RuntimeError("Unknown selection method: {}".format(args.method))
    
    idxs = sorted(idxs)
    print(idxs)
    with open(ids_path, "w") as f:
        for idx in idxs:
            f.write("{}\n".format(timesteps[idx]))
    
    imgs = []
    for idx in idxs:
        image_path = image_paths[idx]
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
        imgs.append(img)
        
    n = 100
    if len(imgs) < n:
        h, w = imgs[0].shape[:2]
        for _ in range(n - len(imgs)):
            imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
    imgs = imgs[:n]
    
    rows = []
    for i in range(10):
        row = np.concatenate(imgs[i * 10:(i + 1) * 10], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    
    cv2.imwrite(show_image_path, grid)
    
    
    
    

    
    