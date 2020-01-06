import argparse
import time

import open3d as o3d
import cv2
import numpy as np

import fusion

import platform
if platform.system() == "Darwin":
  print("Using macOS")
  import os
  os.environ['KMP_DUPLICATE_LIB_OK']='True'

TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}


def main(args):
  if args.example == 'cpp':
    print("Using PyTorch CPP.")
    from cpp.integrate import integrate
  elif args.example == 'jit':
    print("Using PyTorch JIT.")
    from jit.integrate import integrate
  elif args.example == 'py':
    print("Using vanilla PyTorch.")
    from python.integrate import integrate
  else:
    pass

  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 15
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
    # Read depth image and camera pose
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, 0.02, integrate)

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  times = []
  for i in range(n_imgs):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(i)), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0
    cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))

    # Integrate observation into voxel volume (assume color aligned with depth)
    tic = time.time()
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
    toc = time.time()
    times.append(toc-tic)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  times = [t*TIME_SCALES[args.scale] for t in times]
  print("Average integration time: {:.3f} {}".format(np.mean(times), args.scale))

  # Extract pointcloud
  point_cloud = tsdf_vol.extract_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.extract_triangle_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('example', choices=['pycuda', 'py', 'cpp', 'jit', 'cuda'])
  parser.add_argument('--scale', choices=['s', 'ms', 'us'], default='s')
  args = parser.parse_args()
  main(args)