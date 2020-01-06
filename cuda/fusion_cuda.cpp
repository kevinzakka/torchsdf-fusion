#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> fusion_cuda_integrate(
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
  float obs_weight);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fusion_integrate(
  torch::Tensor coords_world,
  torch::Tensor coords_vox,
  torch::Tensor weight_vol,
  torch::Tensor tsdf_vol,
  torch::Tensor color_vol,
  torch::Tensor color_im,
  torch::Tensor depth_im,
  torch::Tensor intr,
  torch::Tensor extr,
  float im_height,
  float im_width,
  float sdf_trunc,
  float obs_weight) {
  CHECK_INPUT(coords_world);
  CHECK_INPUT(coords_vox);
  CHECK_INPUT(weight_vol);
  CHECK_INPUT(tsdf_vol);
  CHECK_INPUT(color_vol);
  CHECK_INPUT(color_im);
  CHECK_INPUT(depth_im);
  CHECK_INPUT(intr);
  CHECK_INPUT(extr);

  return fusion_cuda_integrate(coords_world, coords_vox, weight_vol, tsdf_vol,
    color_vol, color_im, depth_im, intr, extr, im_height, im_width, sdf_trunc, obs_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("integrate", &fusion_integrate, "TSDF integrate (CUDA)");
}