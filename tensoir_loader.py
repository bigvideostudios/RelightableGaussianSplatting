import json
import os
import numpy as np
import imageio.v2 as imageio
import torch
from PIL import Image

def get_split(data_dir, json_name):
    with open(os.path.join(data_dir, json_name)) as f:
        metadata = json.load(f)

    paths = []
    c2ws = []
    for frame in metadata["frames"]:
        paths.append(os.path.join(data_dir, frame["file_path"]) + ".png")
        transform = np.array(frame["transform_matrix"])
        transform[:3, 1:3] *= -1
        c2ws.append(transform)

    return paths, np.stack(c2ws), metadata["camera_angle_x"]


def get_intrinsics(angle, W, H):
    focal = (W/2)/np.tan(angle/2)
    k_intrinsic = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], dtype=np.float32)
    return k_intrinsic


def add_black_background(path, size = None):
    #img = imageio.imread(path).astype(np.float32)/255.0
    img = imageio.imread(path)
    if size is not None:
        img = np.array(Image.fromarray(img).resize(size, Image.Resampling.BICUBIC))
    img = img.astype(np.float32) /255.0
    rgb = img[:,:,:3]
    alpha = img[:,:,3:4]
    out= rgb * alpha
    return out * 255.0


def check_alignment(cameratoworld):
    view = cameratoworld[:, :3, 2]
    cam_pos = cameratoworld[:, :3, 3]
    to_obj = -cam_pos
    to_obj = to_obj / np.linalg.norm(to_obj, axis=1, keepdims=True)
    prod = (view * to_obj).sum(axis=1)
    print(f"z-alignment {prod.min()}")
    down = cameratoworld[:, :3, 1]
    down  = down @ np.array([0,0,-1])
    print(f"y-alignment {down.min()}")


class Parser:
    def __init__(self, data_dir, factor = 1, normalize = False, test_every = 8):
        self.data_dir = data_dir
        self.factor = factor

        self.splits = {}
        angle = None
        for split in ["train", "val", "test"]:
            name =  f"transforms_{split}.json"

            path, c2ws, split_angle = get_split(data_dir, name)
            if split == "val":
                path = path[::test_every]
                c2ws = c2ws[::test_every]
            if split == "train":
                angle = split_angle
            self.splits[split] = {"path": path, "c2ws": c2ws}

        h,w = imageio.imread(self.splits["train"]["path"][0]).shape[:2]
        self.h = h//self.factor
        self.w = w//self.factor
        self.K = get_intrinsics(angle, self.w, self.h)
        
        self.Ks_dict = {0: self.K} #only one camera
        self.imsize_dict = {0: (self.w, self.h)}
        self.camtoworlds = self.splits["train"]["c2ws"]

        #will not be used as TensoIR has no point cloud, but exists so Praser has same attributes as gsplat
        self.points = np.zeros((0,3))
        self.points_rgb = np.zeros((0,3), dtype=np.uint8)

        cam_pos = self.splits["train"]["c2ws"][:, :3, 3]
        self.scene_scale = float(np.max(np.linalg.norm(cam_pos - cam_pos.mean(axis=0), axis = 1)))


class Dataset:

    def __init__(self, parser, split = "train", patch_size = None, load_depths = False):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depth = load_depths
        self.paths = parser.splits[self.split]["path"]
        self.c2ws = parser.splits[self.split]["c2ws"]

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, idx):
        size = None
        if self.parser.factor > 1:
            size = (self.parser.w, self.parser.h)

        img = add_black_background(self.paths[idx], size=size)

        return {
            "camtoworld": torch.from_numpy(self.c2ws[idx]).float(),
            "K": torch.from_numpy(self.parser.K).float(),
            "image": torch.from_numpy(img).float(),
            "image_id": idx
        }

if __name__ == "__main__":
    data_dir = "lego"
    parser = Parser(data_dir, factor = 1)
    trainset = Dataset(parser, split = "train")
    valset = Dataset(parser, split = "val")
    check_alignment(parser.camtoworlds)
    print(f"Instrinsic {parser.K}")
    print(f"# images: {len(trainset)}")
    print(f"Scene scale: {parser.scene_scale}")

    item = trainset[0]
    print(item)
