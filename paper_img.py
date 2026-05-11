import torch
import numpy as np
import os
from PIL import Image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper Image Generation Script")
    parser.add_argument('-i', '--input_image', type=str, required=True, help='Path to the input image')
    parser.add_argument('-m', '--mode', type=str, default='3x2')
    args = parser.parse_args()

    image = Image.open(args.input_image).convert("RGB")
    print(image.size)
    # split image into pieces
    images = []
    width_range = int(args.mode.split('x')[0])
    height_range = int(args.mode.split('x')[1])
    piece_width = image.width // width_range
    piece_height = image.height // height_range
    print(piece_width, piece_height)
    for i in range(width_range):
        for j in range(height_range):
            left = i * piece_width
            upper = j * piece_height
            right = left + piece_width
            lower = upper + piece_height
            img_piece = image.crop((left, upper, right, lower))
            images.append(img_piece)
            
    os.makedirs("paper_img", exist_ok=True)
            
    for i in range(len(images)):
        images[i].save(os.path.join("paper_img", str(i) + ".png"))