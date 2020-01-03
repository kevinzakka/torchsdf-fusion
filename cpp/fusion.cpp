#include <torch/extension.h>

#include <vector>

const float const_val = 256*256;

// integrate an RGB-D frame into the TSDF volume
std::vector<at::Tensor> fusion_integrate(
  torch::Tensor coords_world,
  torch::Tensor coords_vox,
  torch::Tensor &weight_vol,
  torch::Tensor &tsdf_vol,
  torch::Tensor &color_vol,
  torch::Tensor color_im,
  torch::Tensor depth_im,
  torch::Tensor intr,
  torch::Tensor extr,
  float im_height,
  float im_width,
  float sdf_trunc,
  float obs_weight) {
  // fold color into single-channel
  color_im = color_im.narrow(2, 2, 1)*const_val + color_im.narrow(2, 1, 1)*256 + color_im.narrow(2, 0, 1);
  color_im = torch::floor(color_im).squeeze();

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

  // skip voxels with invalid depth values or outside truncation
  auto depth_val = depth_im.index({pix_y_valid, pix_x_valid});
  auto depth_diff = depth_val - pix_z_valid;
  auto dist = torch::clamp_max(depth_diff / sdf_trunc, 1.0);
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
  tsdf_vol.index_put_({vox_x_valid, vox_y_valid, vox_z_valid}, (w_old * tsdf_old + valid_dist) / w_new);
  weight_vol.index_put_({vox_x_valid, vox_y_valid, vox_z_valid}, w_new);

  // integrate color
  auto old_color = color_vol.index({vox_x_valid, vox_y_valid, vox_z_valid});
  auto old_b = torch::floor(old_color / const_val);
  auto old_g = torch::floor((old_color-old_b*const_val) / 256);
  auto old_r = old_color - old_b*const_val - old_g*256;
  auto new_color = color_im.index({pix_y_valid.index(mask_vox), pix_x_valid.index(mask_vox)});
  auto new_b = torch::floor(new_color / const_val);
  auto new_g = torch::floor((new_color - new_b*const_val) / 256);
  auto new_r = new_color - new_b*const_val - new_g*256;
  auto new_b_ = torch::clamp_max(torch::round((w_old*old_b + new_b) / w_new), 255);
  auto new_g_ = torch::clamp_max(torch::round((w_old*old_g + new_g) / w_new), 255);
  auto new_r_ = torch::clamp_max(torch::round((w_old*old_r + new_r) / w_new), 255);
  color_vol.index_put_({vox_x_valid, vox_y_valid, vox_z_valid}, new_b_*const_val + new_g_*256 + new_r_);

  return {weight_vol, tsdf_vol, color_vol};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("integrate", &fusion_integrate, "TSDF integrate");
}