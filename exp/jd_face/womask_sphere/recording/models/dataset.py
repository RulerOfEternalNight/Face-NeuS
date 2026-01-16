# import torch
# import torch.nn.functional as F
# import cv2 as cv
# import numpy as np
# import os
# from glob import glob
# from icecream import ic
# from scipy.spatial.transform import Rotation as Rot
# from scipy.spatial.transform import Slerp


# # This function is borrowed from IDR: https://github.com/lioryariv/idr
# def load_K_Rt_from_P(filename, P=None):
#     if P is None:
#         lines = open(filename).read().splitlines()
#         if len(lines) == 4:
#             lines = lines[1:]
#         lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
#         P = np.asarray(lines).astype(np.float32).squeeze()

#     out = cv.decomposeProjectionMatrix(P)
#     K = out[0]
#     R = out[1]
#     t = out[2]

#     K = K / K[2, 2]
#     intrinsics = np.eye(4)
#     intrinsics[:3, :3] = K

#     pose = np.eye(4, dtype=np.float32)
#     pose[:3, :3] = R.transpose()
#     pose[:3, 3] = (t[:3] / t[3])[:, 0]

#     return intrinsics, pose


# class Dataset:
#     def __init__(self, conf):
#         super(Dataset, self).__init__()
#         print('Load data: Begin')
#         self.device = torch.device('cuda')
#         self.conf = conf

#         self.data_dir = conf.get_string('data_dir')
#         self.render_cameras_name = conf.get_string('render_cameras_name')
#         self.object_cameras_name = conf.get_string('object_cameras_name')

#         self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
#         self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

#         camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
#         self.camera_dict = camera_dict
#         self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
#         self.n_images = len(self.images_lis)
#         self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
#         self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
#         self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

#         # world_mat is a projection matrix from world to image
#         self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

#         self.scale_mats_np = []

#         # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
#         self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

#         self.intrinsics_all = []
#         self.pose_all = []

#         for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
#             P = world_mat @ scale_mat
#             P = P[:3, :4]
#             intrinsics, pose = load_K_Rt_from_P(None, P)
#             self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
#             self.pose_all.append(torch.from_numpy(pose).float())

#         self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
#         self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
#         self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
#         self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
#         self.focal = self.intrinsics_all[0][0, 0]
#         self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
#         self.H, self.W = self.images.shape[1], self.images.shape[2]
#         self.image_pixels = self.H * self.W

#         object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
#         object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
#         # Object scale mat: region of interest to **extract mesh**
#         object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
#         object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
#         object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
#         self.object_bbox_min = object_bbox_min[:3, 0]
#         self.object_bbox_max = object_bbox_max[:3, 0]

#         print('Load data: End')

#     def gen_rays_at(self, img_idx, resolution_level=1):
#         """
#         Generate rays at world space from one camera.
#         """
#         l = resolution_level
#         tx = torch.linspace(0, self.W - 1, self.W // l)
#         ty = torch.linspace(0, self.H - 1, self.H // l)
#         pixels_x, pixels_y = torch.meshgrid(tx, ty)
#         p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
#         p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
#         rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
#         rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
#         rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
#         return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

#     # def gen_random_rays_at(self, img_idx, batch_size):
#     #     """
#     #     Generate random rays at world space from one camera.
#     #     """
#     #     pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
#     #     pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
#     #     color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
#     #     mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
#     #     p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
#     #     p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
#     #     rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
#     #     rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
#     #     rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
#     #     return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10


#     def gen_random_rays_at(self, img_idx, batch_size):
#         """
#         Generate random rays at world space from one camera.
#         """
#         # These are likely generated on the GPU (default device: self.device='cuda')
#         pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
#         pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        
#         # --- 🛠️ THE FIX IS APPLIED HERE ---
#         # Move indices to the CPU to match the device of self.images and self.masks
#         pixels_x_cpu = pixels_x.cpu()
#         pixels_y_cpu = pixels_y.cpu()
        
#         # Use CPU indices for indexing CPU tensors
#         color = self.images[img_idx][(pixels_y_cpu, pixels_x_cpu)]    # batch_size, 3
#         mask = self.masks[img_idx][(pixels_y_cpu, pixels_x_cpu)]      # batch_size, 3
#         # ----------------------------------
        
#         # The coordinates tensor 'p' still needs the GPU-based pixels_x/y for matrix multiplication
#         # with the GPU-based intrinsics_all_inv, so we continue using the original (GPU) pixels_x/y.
#         p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        
#         p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
#         rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
#         rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
#         rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        
#         # Note: color and mask are currently on CPU (inherited from self.images/masks).
#         # They will be moved to CUDA in the final line's concatenation.
#         return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()  # batch_size, 10

#     def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
#         """
#         Interpolate pose between two cameras.
#         """
#         l = resolution_level
#         tx = torch.linspace(0, self.W - 1, self.W // l)
#         ty = torch.linspace(0, self.H - 1, self.H // l)
#         pixels_x, pixels_y = torch.meshgrid(tx, ty)
#         p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
#         p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
#         rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
#         trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
#         pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
#         pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
#         pose_0 = np.linalg.inv(pose_0)
#         pose_1 = np.linalg.inv(pose_1)
#         rot_0 = pose_0[:3, :3]
#         rot_1 = pose_1[:3, :3]
#         rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
#         key_times = [0, 1]
#         slerp = Slerp(key_times, rots)
#         rot = slerp(ratio)
#         pose = np.diag([1.0, 1.0, 1.0, 1.0])
#         pose = pose.astype(np.float32)
#         pose[:3, :3] = rot.as_matrix()
#         pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
#         pose = np.linalg.inv(pose)
#         rot = torch.from_numpy(pose[:3, :3]).cuda()
#         trans = torch.from_numpy(pose[:3, 3]).cuda()
#         rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
#         rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
#         return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

#     def near_far_from_sphere(self, rays_o, rays_d):
#         a = torch.sum(rays_d**2, dim=-1, keepdim=True)
#         b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
#         mid = 0.5 * (-b) / a
#         near = mid - 1.0
#         far = mid + 1.0
#         return near, far

#     def image_at(self, idx, resolution_level):
#         img = cv.imread(self.images_lis[idx])
#         return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)


import os
import json
import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict

        # Images + masks
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        if self.n_images == 0:
            raise FileNotFoundError(f"No images found at: {os.path.join(self.data_dir, 'image/*.png')}")

        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0

        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        if len(self.masks_lis) == self.n_images:
            self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0
        else:
            # If masks are missing / count mismatch, fall back to all-ones masks
            H, W = self.images_np.shape[1], self.images_np.shape[2]
            self.masks_np = np.ones((self.n_images, H, W, 3), dtype=np.float32)

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        # scale_mat: used for coordinate normalization, assume scene inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)             # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)               # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        # Object bbox (for mesh extraction)
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        # ------------------------------------------------------------
        # Landmark JSON loading (optional)
        # Expecting: <data_dir>/landmarks/landmarks_2d.json
        # Example: public_data/jd_face/landmarks/landmarks_2d.json
        # ------------------------------------------------------------
        self.landmarks_enabled = False
        self.landmarks_dict = None
        self.landmark_names_to_indices = None

        lm_json_path = os.path.join(self.data_dir, "landmarks", "landmarks_2d.json")
        if os.path.exists(lm_json_path):
            try:
                with open(lm_json_path, "r") as f:
                    lm_data = json.load(f)

                self.landmark_names_to_indices = lm_data.get("landmark_names_to_indices", {})
                self.landmarks_dict = lm_data.get("images", {})

                # Basic sanity: do we have at least one entry?
                if isinstance(self.landmarks_dict, dict) and len(self.landmarks_dict) > 0:
                    self.landmarks_enabled = True
                    print(f"[Landmarks] Loaded: {lm_json_path} (entries: {len(self.landmarks_dict)})")
                else:
                    print(f"[Landmarks] Found JSON but it has no 'images' entries: {lm_json_path}")

            except Exception as e:
                print(f"[Landmarks] Failed to load landmarks JSON: {lm_json_path}")
                print(f"[Landmarks] Error: {e}")
        else:
            print(f"[Landmarks] Not found (ok): {lm_json_path}")

        print('Load data: End')

    # ------------------------------------------------------------
    # Existing NeuS ray generation
    # ------------------------------------------------------------
    # def gen_rays_at(self, img_idx, resolution_level=1):
    #     """
    #     Generate rays at world space from one camera.
    #     """
    #     l = resolution_level
    #     tx = torch.linspace(0, self.W - 1, self.W // l)
    #     ty = torch.linspace(0, self.H - 1, self.H // l)
    #     pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='ij' if hasattr(torch.meshgrid, "__call__") else None)
    #     p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3

    #     p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    #     rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
    #     rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    #     rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
    #     return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        Output shape: [H/l, W/l, 3]
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l, device=self.device)
        ty = torch.linspace(0, self.H - 1, self.H // l, device=self.device)

        # Correct: make grid in (y, x) order with ij indexing -> shape [H, W]
        pixels_y, pixels_x = torch.meshgrid(ty, tx, indexing='ij')

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # H, W, 3

        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[..., None]).squeeze(-1)  # H,W,3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # H,W,3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[..., None]).squeeze(-1)  # H,W,3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand_as(rays_v)  # H,W,3
        return rays_o, rays_v


    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        NOTE: This keeps your existing CPU-indexing fix.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])

        # CPU indices to index CPU tensors (self.images, self.masks)
        pixels_x_cpu = pixels_x.cpu()
        pixels_y_cpu = pixels_y.cpu()

        color = self.images[img_idx][(pixels_y_cpu, pixels_x_cpu)]  # [B,3]
        mask = self.masks[img_idx][(pixels_y_cpu, pixels_x_cpu)]    # [B,3]

        # Use original (GPU) pixels for ray construction
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # [B,3]
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # [B,3]
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # [B,3]
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # [B,3]
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # [B,3]

        # Return as [B,10] on CUDA
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()

    # def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
    #     """
    #     Interpolate pose between two cameras.
    #     """
    #     l = resolution_level
    #     tx = torch.linspace(0, self.W - 1, self.W // l)
    #     ty = torch.linspace(0, self.H - 1, self.H // l)
    #     pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='ij' if hasattr(torch.meshgrid, "__call__") else None)
    #     p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
    #     p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
    #     rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3

    #     trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
    #     pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
    #     pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
    #     pose_0 = np.linalg.inv(pose_0)
    #     pose_1 = np.linalg.inv(pose_1)
    #     rot_0 = pose_0[:3, :3]
    #     rot_1 = pose_1[:3, :3]
    #     rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
    #     key_times = [0, 1]
    #     slerp = Slerp(key_times, rots)
    #     rot = slerp(ratio)

    #     pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
    #     pose[:3, :3] = rot.as_matrix()
    #     pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
    #     pose = np.linalg.inv(pose)

    #     rot = torch.from_numpy(pose[:3, :3]).cuda()
    #     trans = torch.from_numpy(pose[:3, 3]).cuda()
    #     rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    #     rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
    #     return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l, device=self.device)
        ty = torch.linspace(0, self.H - 1, self.H // l, device=self.device)

        pixels_y, pixels_x = torch.meshgrid(ty, tx, indexing='ij')
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # H,W,3

        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[..., None]).squeeze(-1)  # H,W,3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # H,W,3

        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        slerp = Slerp([0, 1], rots)
        rot = slerp(ratio)

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)

        rot = torch.from_numpy(pose[:3, :3]).to(self.device)
        trans = torch.from_numpy(pose[:3, 3]).to(self.device)

        rays_v = torch.matmul(rot[None, None, :, :], rays_v[..., None]).squeeze(-1)  # H,W,3
        rays_o = trans[None, None, :].expand_as(rays_v)  # H,W,3
        return rays_o, rays_v


    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    # ------------------------------------------------------------
    # NEW: Landmark utilities
    # ------------------------------------------------------------
    def _img_basename(self, img_idx: int) -> str:
        return os.path.basename(self.images_lis[img_idx])

    def get_landmark_pixels(self, img_idx: int, names=None):
        """
        Returns landmark pixels for an image index.

        Output:
          pixels_xy: FloatTensor [K,2] on CUDA (x,y)
          names_out: list[str] length K (landmark names)
        """
        if not self.landmarks_enabled:
            return None, None

        base = self._img_basename(img_idx)
        entry = self.landmarks_dict.get(base, None)
        if entry is None or not entry.get("detected", False):
            return None, None

        lm2d = entry.get("landmarks_2d", None)
        if lm2d is None:
            return None, None

        # Choose which landmark names
        if names is None:
            names = list(lm2d.keys())

        pts = []
        names_out = []
        for n in names:
            if n in lm2d and lm2d[n] is not None:
                x, y = lm2d[n]
                # clamp just in case
                x = float(np.clip(x, 0.0, self.W - 1.0))
                y = float(np.clip(y, 0.0, self.H - 1.0))
                pts.append([x, y])
                names_out.append(n)

        if len(pts) == 0:
            return None, None

        pixels_xy = torch.tensor(pts, dtype=torch.float32, device=self.device)  # [K,2]
        return pixels_xy, names_out

    def gen_landmark_rays_at(self, img_idx: int, pixels_xy: torch.Tensor):
        """
        Generate rays for specific pixel coordinates (landmarks).

        Args:
          pixels_xy: Tensor [K,2] (x,y) in pixel coordinates

        Returns:
          rays_o: [K,3]
          rays_d: [K,3]
        """
        if pixels_xy is None or len(pixels_xy) == 0:
            return None, None

        if pixels_xy.device != self.device:
            pixels_xy = pixels_xy.to(self.device)

        x = pixels_xy[:, 0]
        y = pixels_xy[:, 1]
        ones = torch.ones_like(x)

        # p: [K,3] (x,y,1)
        p = torch.stack([x, y, ones], dim=-1).float()  # [K,3]

        # Camera-space directions
        Kinv = self.intrinsics_all_inv[img_idx, :3, :3]  # [3,3]
        p_cam = torch.matmul(Kinv[None, :, :], p[:, :, None]).squeeze(-1)  # [K,3]
        rays_d = p_cam / torch.linalg.norm(p_cam, ord=2, dim=-1, keepdim=True)  # [K,3]

        # World-space directions
        R = self.pose_all[img_idx, :3, :3]  # [3,3]
        t = self.pose_all[img_idx, :3, 3]   # [3]
        rays_d = torch.matmul(R[None, :, :], rays_d[:, :, None]).squeeze(-1)     # [K,3]
        rays_o = t[None, :].expand_as(rays_d)                                    # [K,3]

        return rays_o, rays_d

    def gen_random_landmark_rays_at(self, img_idx: int, n_rays: int, names=None):
        """
        Convenience helper: sample up to n_rays landmark rays from this image.
        If less landmarks exist, returns all available.

        Returns:
          rays_o [M,3], rays_d [M,3], pixels_xy [M,2], names_out list[str]
        """
        pixels_xy, names_out = self.get_landmark_pixels(img_idx, names=names)
        if pixels_xy is None:
            return None, None, None, None

        K = pixels_xy.shape[0]
        if n_rays is not None and n_rays > 0 and K > n_rays:
            # random subset
            idx = torch.randperm(K, device=self.device)[:n_rays]
            pixels_xy = pixels_xy[idx]
            names_out = [names_out[int(i)] for i in idx.detach().cpu().tolist()]

        rays_o, rays_d = self.gen_landmark_rays_at(img_idx, pixels_xy)
        return rays_o, rays_d, pixels_xy, names_out
