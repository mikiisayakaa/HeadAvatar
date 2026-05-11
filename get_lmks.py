import torch
import face_alignment
from skimage import io
import numpy as np
import cv2
import os

def predictions_to_img(prediction, image, name):
    H = image.shape[0]
    W = image.shape[1]
    
    prediction = torch.from_numpy(prediction)
    img = torch.from_numpy(image)
    
    valid = (prediction[:, 0] < W) & (prediction[:, 1] < H) & \
        (prediction[:, 0] >= 0) & (prediction[:, 1] >= 0)
    valid_preds = prediction[valid]
    hmin = torch.clamp(valid_preds[:, 1] - 1, 0, H - 1).to(int)
    hmax = torch.clamp(valid_preds[:, 1] + 2, 0, H - 1).to(int)
    wmin = torch.clamp(valid_preds[:, 0] - 1, 0, W - 1).to(int)
    wmax = torch.clamp(valid_preds[:, 0] + 2, 0, W - 1).to(int)
    
    color = torch.tensor([100, 0, 0], dtype=torch.uint8)[None, None, :]
    add = torch.tensor([2, 0, 0], dtype=torch.uint8)[None, None, :]
    
    for j in range(valid_preds.shape[0]):
        img[hmin[j].item():hmax[j].item(), wmin[j].item():wmax[j].item()] = color
        color += add
    
    img = img.cpu().numpy()[:, :, ::-1]
    cv2.imwrite(name + ".png", img)

if __name__ == "__main__":
    device = 'cuda:2'
    root = "/home/yanp/HeadAvatar/data/1"
    save_root = os.path.join(root, "landmarks")
    image_root = os.path.join(root, "images")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
    for idx in range(131):
        predictions = []
        valids = []
        name = str(idx).zfill(5)
        for i in range(16):
            img_name = name + "_" + str(i).zfill(2) + ".png"
            img_path = os.path.join(image_root, img_name)
            if not os.path.exists(img_path):
                continue
            print("Processing " + img_name + ":")
            input = io.imread(img_path)
            preds = fa.get_landmarks_from_image(input)
            if preds is None:
                preds = -np.ones((51, 2), dtype=np.int32)
            else:
                preds = preds[0][17:].astype(np.int32)
            valid = (preds[:, 0] >= 0) & (preds[:, 1] >= 0) & \
                (preds[:, 0] < input.shape[0]) & (preds[:, 1] < input.shape[1])

            predictions_to_img(preds, input, os.path.join(save_root, name + "_" + str(i).zfill(2)))
            predictions.append(preds)
            valids.append(valid)

        
        prediction = np.stack(predictions, axis=0)
        valid = np.stack(valids, axis=0)

        path = os.path.join(save_root, name + ".npz")
        np.savez(path, landmarks=prediction, validation=valid)
    
        