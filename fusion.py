"""Andy's tsdf-fusion in pytorch, because why not.
"""

import numpy as np

from skimage import measure

import torch
import fusion_cpp


class TSDFVolume:
  """Volumetric TSDF Fusion of RGB-D Images.
  """
  def __init__(self, vol_bnds, voxel_size):
    """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
    vol_bnds = np.asarray(vol_bnds)
    assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      print("[!] No GPU detected. Defaulting to CPU.")
      self.device = torch.device("cpu")

    # Define voxel volume parameters
    self._vol_bnds = torch.from_numpy(vol_bnds).float().to(self.device)
    self._voxel_size = float(voxel_size)
    self._sdf_trunc = 5 * self._voxel_size
    self._const = 256*256

    # Adjust volume bounds
    self._vol_dim = torch.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).long()
    self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + (self._vol_dim * self._voxel_size)
    self._vol_origin = self._vol_bnds[:, 0]
    self._num_voxels = torch.prod(self._vol_dim).item()

    # Get voxel grid coordinates
    xv, yv, zv = torch.meshgrid(
      torch.arange(0, self._vol_dim[0]),
      torch.arange(0, self._vol_dim[1]),
      torch.arange(0, self._vol_dim[2]),
    )
    self._vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(self.device)

    # Convert voxel coordinates to world coordinates
    self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords)
    self._world_c = torch.cat([
      self._world_c, torch.ones(len(self._world_c), 1, device=self.device)], dim=1)

    self.reset()

    print("[*] voxel volume: {} x {} x {}".format(*self._vol_dim))
    print("[*] num voxels: {:,}".format(self._num_voxels))

  def reset(self):
    self._tsdf_vol = torch.ones(*self._vol_dim).to(self.device)
    self._weight_vol = torch.zeros(*self._vol_dim).to(self.device)
    self._color_vol = torch.zeros(*self._vol_dim).to(self.device)

  def integrate_cpp(self, color_im, depth_im, cam_intr, cam_pose, obs_weight):
    """Integrate an RGB-D frame into the TSDF volume.

    Args:
      color_im (ndarray): An RGB image of shape (H, W, 3).
      depth_im (ndarray): A depth image of shape (H, W).
      cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
      cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
      obs_weight (float): The weight to assign to the current observation.
    """
    cam_pose = torch.from_numpy(cam_pose).float().to(self.device)
    cam_intr = torch.from_numpy(cam_intr).float().to(self.device)
    color_im = torch.from_numpy(color_im).float().to(self.device)
    depth_im = torch.from_numpy(depth_im).float().to(self.device)
    im_h, im_w = depth_im.shape

    # Fold RGB color image into a single channel image
    color_im = torch.floor(color_im[..., 2]*self._const + color_im[..., 1]*256 + color_im[..., 0])

    fusion_cpp.integrate(
      self._world_c,
      self._vox_coords,
      self._weight_vol,
      self._tsdf_vol,
      self._color_vol,
      color_im,
      depth_im,
      cam_intr,
      cam_pose,
      im_h,
      im_w,
      self._sdf_trunc,
      obs_weight,
    )

  def extract_point_cloud(self):
    """Extract a point cloud from the voxel volume.
    """
    pass

  def extract_triangle_mesh(self):
    """Extract a triangle mesh from the voxel volume using marching cubes.
    """
    tsdf_vol = self._tsdf_vol.cpu().numpy()
    color_vol = self._color_vol.cpu().numpy()
    vol_origin = self._vol_origin.cpu().numpy()

    # Marching cubes
    verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts*self._voxel_size + vol_origin

    # Get vertex colors
    rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    colors_b = np.floor(rgb_vals / self._const)
    colors_g = np.floor((rgb_vals - colors_b*self._const) / 256)
    colors_r = rgb_vals - colors_b*self._const - colors_g*256
    colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
    colors = colors.astype(np.uint8)

    return verts, faces, norms, colors

  @property
  def sdf_trunc(self):
    return self._sdf_trunc

  @property
  def voxel_size(self):
    return self._voxel_size


def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = (transform @ xyz_h.T).T
  return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
  """Get corners of 3D camera view frustum of depth image
  """
  im_h = depth_im.shape[0]
  im_w = depth_im.shape[1]
  max_depth = np.max(depth_im)
  view_frust_pts = np.array([
    (np.array([0,0,0,im_w,im_w])-cam_intr[0,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[0,0],
    (np.array([0,0,im_h,0,im_h])-cam_intr[1,2])*np.array([0,max_depth,max_depth,max_depth,max_depth])/cam_intr[1,1],
    np.array([0,max_depth,max_depth,max_depth,max_depth])
  ])
  view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
  return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      colors[i,0], colors[i,1], colors[i,2],
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()