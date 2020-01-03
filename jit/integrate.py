import torch


@torch.jit.script
def integrate(
  color_im,
  depth_im,
  cam_intr,
  cam_pose,
  obs_weight: float,
  world_c,
  vox_coords,
  weight_vol,
  tsdf_vol,
  color_vol,
  sdf_trunc: float,
  im_h: int,
  im_w: int,
):
  const_val = 256*256

  # Fold RGB color image into a single channel image
  color_im = torch.floor(color_im[..., 2]*256*256 + color_im[..., 1]*256 + color_im[..., 0])

  # Convert world coordinates to camera coordinates
  world2cam = torch.inverse(cam_pose)
  cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

  # Convert camera coordinates to pixel coordinates
  fx, fy = cam_intr[0, 0], cam_intr[1, 1]
  cx, cy = cam_intr[0, 2], cam_intr[1, 2]
  pix_z = cam_c[:, 2]
  pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
  pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

  # Eliminate pixels outside view frustum
  valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
  valid_vox_x = vox_coords[valid_pix, 0]
  valid_vox_y = vox_coords[valid_pix, 1]
  valid_vox_z = vox_coords[valid_pix, 2]
  valid_pix_y = pix_y[valid_pix]
  valid_pix_x = pix_x[valid_pix]
  depth_val = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

  # Integrate tsdf
  depth_diff = depth_val - pix_z[valid_pix]
  dist = torch.clamp(depth_diff / sdf_trunc, max=1)
  valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)
  valid_vox_x = valid_vox_x[valid_pts]
  valid_vox_y = valid_vox_y[valid_pts]
  valid_vox_z = valid_vox_z[valid_pts]
  valid_pix_y = valid_pix_y[valid_pts]
  valid_pix_x = valid_pix_x[valid_pts]
  valid_dist = dist[valid_pts]
  w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
  tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
  w_new = w_old + obs_weight
  tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + valid_dist) / w_new
  weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

  # Integrate color
  old_color = color_vol[valid_vox_x, valid_vox_y, valid_vox_z]
  old_b = torch.floor(old_color / const_val)
  old_g = torch.floor((old_color-old_b*const_val) / 256)
  old_r = old_color - old_b*const_val - old_g*256
  new_color = color_im[valid_pix_y, valid_pix_x]
  new_b = torch.floor(new_color / const_val)
  new_g = torch.floor((new_color - new_b*const_val) / 256)
  new_r = new_color - new_b*const_val - new_g*256
  new_b = torch.clamp(torch.round((w_old*old_b + new_b) / w_new), max=255)
  new_g = torch.clamp(torch.round((w_old*old_g + new_g) / w_new), max=255)
  new_r = torch.clamp(torch.round((w_old*old_r + new_r) / w_new), max=255)
  color_vol[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*const_val + new_g*256 + new_r

  return weight_vol, tsdf_vol, color_vol