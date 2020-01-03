#include <torch/extension.h>

#include <vector>

// integrate an RGB-D frame into the TSDF volume
void fusion_integrate(
  torch::Tensor coords_world,
  torch::Tensor coords_vox,
  torch::Tensor& weight_vol,
  torch::Tensor& color_vol,
  torch::Tensor& tsdf_vol,
  torch::Tensor color_im,
  torch::Tensor depth_im,
  torch::Tensor intr,
  torch::Tensor extr,
  float im_height,
  float im_width,
  float sdf_trunc,
  float obs_weight) {
  // convert world coordinates to camera coordinates
  auto coords_cam = torch::inverse(extr).mm(coords_world.t()).t();

  // convert camera coordinates to pixel coordinates
  auto pixels = intr.mm(coords_cam.narrow(-1, 0, 3).t());
  auto pix_z = pixels.narrow(0, 2, 1).squeeze();
  auto pix_x = at::_cast_Long(torch::round(pixels.narrow(0, 0, 1) / pix_z)).squeeze();
  auto pix_y = at::_cast_Long(torch::round(pixels.narrow(0, 1, 1) / pix_z)).squeeze();

  // skip pixels and voxels outside view frustum
  auto mask_pix = (pix_x >= 0) * (pix_x < im_width) * (pix_y >= 0) * (pix_y < im_height) * (pix_z > 0);
  mask_pix = mask_pix.squeeze();
  auto pix_x_valid = pix_x.index(mask_pix);
  auto pix_y_valid = pix_y.index(mask_pix);
  auto pix_z_valid = pix_z.index(mask_pix);
  auto vox_x_valid = coords_vox.narrow(1, 0, 1).squeeze().index(mask_pix);
  auto vox_y_valid = coords_vox.narrow(1, 1, 1).squeeze().index(mask_pix);
  auto vox_z_valid = coords_vox.narrow(1, 2, 1).squeeze().index(mask_pix);

  // skip voxels with invalid depth values
  auto depth_val = depth_im.index({pix_y_valid, pix_x_valid});
  auto depth_diff = depth_val - pix_z_valid;
  auto dist = torch::clamp_max(depth_diff / sdf_trunc, 1);
  auto mask_vox = (depth_val > 0) * (depth_diff >= -sdf_trunc);
  mask_vox = mask_vox.squeeze();
  vox_x_valid = vox_x_valid.index(mask_vox);
  vox_y_valid = vox_y_valid.index(mask_vox);
  vox_z_valid = vox_z_valid.index(mask_vox);
  auto valid_dist = dist.index(mask_vox);

  // integrate sdf
  auto w_old = weight_vol.index({vox_x_valid, vox_y_valid, vox_z_valid});
  auto tsdf_old = tsdf_vol.index({vox_x_valid, vox_y_valid, vox_z_valid});
  auto w_new = w_old + obs_weight;
  tsdf_vol.index({vox_x_valid, vox_y_valid, vox_z_valid}) = (w_old * tsdf_old + valid_dist) / w_new;
  weight_vol.index({vox_x_valid, vox_y_valid, vox_z_valid}) = w_new;

  // integrate color
  auto color_old = color_vol.index({vox_x_valid, vox_y_valid, vox_z_valid});
  auto b_old = torch::floor(color_old / 256*256);
  auto g_old = torch::floor((color_old-b_old*256*256) / 256);
  auto r_old = color_old - b_old*256*256 - g_old*256;
  auto color_new = color_im.index({pix_y_valid.index(mask_vox), pix_x_valid.index(mask_vox)});
  auto b_new = torch::floor(color_new / 256*256);
  auto g_new = torch::floor((color_new - b_new*256*256) / 256);
  auto r_new = color_new - b_new*256*256 - g_new*256;
  b_new = torch::clamp_max(torch::round((w_old*b_old + b_new) / w_new), 255);
  g_new = torch::clamp_max(torch::round((w_old*g_old + g_new) / w_new), 255);
  r_new = torch::clamp_max(torch::round((w_old*r_old + r_new) / w_new), 255);
  color_vol.index({vox_x_valid, vox_y_valid, vox_z_valid}) = b_new*256*256 + g_new*256 + r_new;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("integrate", &fusion_integrate, "TSDF integrate");
}