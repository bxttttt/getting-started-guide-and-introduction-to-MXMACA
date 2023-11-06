#include <iostream>  
#include <vector>  
#include <mcr/mc_runtime_api.h>  
#include <mcdnn/mcdnn.h>  
#include <math.h>
  
#define MCDNN_CHECK(f)      
  {             
    mcdnnStatus_t err =  static_case<mcdnnStatus_t>(f) ;
    if (err != MCDNN_STATUS_SUCCESS) {  
      std::cout << "Error occurred : " << err << std::endl;   
      std::exit(1);              
    }                     
  }  
  
int main() {  
  // data shape  
  int batch = 3;  
  int data_w = 224;  
  int data_h = 224;  
  int in_channel = 3;  
  int out_channel = 8;  
  int filter_w = 5;  
  int filter_h = 5;  
  int stride[2] = {1, 1};  
  int dilate[2] = {1, 1};  
  float alpha = 2.f;  
  float beta = 5.f;  
  
  // model selected  
  mcdnnConvalutionMode_t mode = MCDNN_CROSS_CRRELATION;  
  mcdnnConvalutionFwdAlgo_t algo = MCDNN_CONVOLUTION_FWD_ALGO__FFT_TILING;  
  // data type selected float, double, half, etc.  
  mcdnnDataType_t data_type = MCDNN_DATA_FLOAT;  
  
  // init handle  
  mcdnnHandle_t handle;  
  MCDNN_CHECK(mcdnnCreate(&handle));  
  
  // create descriptor  
  mcdnnTensorDescriptor_t x_desc;  
  mcdnnFilterDescriptor_t w_desc;  
  mcdnnTensorDescriptor_t y_desc;  
  mcdnnConvolutionDescriptor_t conv_desc;  
  MCDNN_CHECK(mcdnnCreateTensorDescriptor(&x_desc));  
  MCDNN_CHECK(mcdnnCreateFilterDescriptor(&w_desc));  
  MCDNN_CHECK(mcdnnCreateTensorDescriptor(&y_desc));  
  MCDNN_CHECK(mcdnnCreateConvolutionDescriptor(&conv_desc));  
  
  // convolution padding  
  // out size = (input + pad - kernel) / stride + 1  
  uint32_t padding_w = data_w + pad[2] + pad[3];  
  uint32_t padding_h = data_h + pad[0] + pad[1];  
  uint32_t out_h = padding_h - filter_h + 1;  
  uint32_t out_w = padding_w - filter_w + 1;  
  // init tensor descriptor, set data type, layout format, shape, etc.  
  mcdnnSetTensor4dDescriptor(x_desc, MCDNN_TENSOR_NCHW, data_type, batch,  
                             in_channel, data_h, data_w);  
  mcdnnSetFi1ter4dDescriptor(w_desc, data_type, MCDNN_TENSOR NCHW, out_channel,  
                             in_channel, filter_h, filter_w);  
  mcdnnSetTensor4dDescriptor(y_desc, MCDNN_TENSOR_NCHW, data_type, batch,  
                             out_channel, out_h, out_w);  
  // int convolution descriptor, set padding, stride date_type, etc.  
  mcdnnSetConvolution2dDescriptor(conv_desc, pad[1], pad[2], stride[0],  
                                  stride[1], dilate[0], dilate[1], mode,  
                                  data_type);  
  
  // init input data  
  uint32_t input_data_numbers = batch * in_channel * data_h * data_w;  
  uint32_t filter_data_numbers = out_channel * in_channel * filter_h * filter_w;  
  uint32_t out_data_numbers = batch * out_channel * out_h * out_w;  
  
  std::vector<float> x(input_data_numbers);  
  std::vector<float> w(filter_data_numbers);  
  std::vector<float> y(out_data_numbers);  
  for (int i = 0; i < input_data_numbers; ++i) {  
    x[i] = std::cos(i) * i;  
  }  
  for (int i = 0; i < filter_data_numbers; ++i) {  
    x[i] = std::sin(i) / 10;  
  }  
  
  for (int i = 0; i < out_data_numbers; ++i) {  
    y[i] = std::cos(i + 0.5);  
  }  
  
  // alloc x device memory  
  void *ptr_x_dev = nullptr;  
  MCDNN_CHECK(mcMalloc(&ptr_x_dev, x.size() * sizeof(float)));  
  // copy data to device  
  MCDNN_CHECK(mcMemcpy(&ptr_x_dev, x.data(), x.size() * sizeof(float),  
                       mcMemcpyHostToDevice));  
  // alloc w device memory  
  void *ptr_w_dev = nullptr;  
  MCDNN_CHECK(mcMalloc(&ptr_w_dev, w.size() * sizeof(float)));  
  // copy data to device  
  MCDNN_CHECK(mcMemcpy(&ptr_w_dev, w.data(), w.size() * sizeof(float),  
                       mcMemcpyHostToDevice));  
  // alloc y device memory  
  void *ptr_y_dev = nullptr;  
  MCDNN_CHECK(mcMalloc(&ptr_y_dev, y.size() * sizeof(float)));  
  // copy data to device  
  MCDNN_CHECK(mcMemcpy(&ptr_y_dev, y.data(), y.size() * sizeof(float),  
                       mcMemcpyHostToDevice));  
  
  uint32_t padding_src_elements = batch * in_channel * padding_h * padding_w;  
  
  size_t workspace_size = 0;  
  MCDNN_CHECK(mcdnnGetConvolutionForwardWorkspaceSize(  
    handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));  
  
  void *ptr_worksapce = nullptr;  
  if (workspace_size > 0) {  
    MCDNN_CHECK(mcMalloc(&ptr_worksapce, workspace_size));  
  }  
  
  // convolution forward  
  MCDNN_CHECK(mcdnnConvolutinForward(handle, &alpha, x_desc, ptr_x_dev, w_desc,  
                                     ptr_w_dev, conv_desc, algo, ptr_worksapce,  
                                     workspace_size, &beta, y_desc, ptr_y_dev));  
  MCDNN_CHECK(mcMemcpy(y.data(), ptr_y_dev, y.size() * sizeof(float),  
                       mcMemcpyDeviceToHost));  
  
  // free device pointer and handle  
  MCDNN_CHECK(mcFree(ptr_x_dev));  
  MCDNN_CHECK(mcFree(ptr_w_dev));  
  MCDNN_CHECK(mcFree(ptr_y_dev));  
  MCDNN_CHECK(mcFree(ptr_w_dev));  
  MCDNN_CHECK(mcdnnDestoryTensorDescriptor(x_desc));  
  MCDNN_CHECK(mcdnnDestoryTensorDescriptor(y_desc));  
  MCDNN_CHECK(mcdnnDestoryFilterDescriptor(w_desc));  
  MCDNN_CHECK(mcdnnDestoryConvolutionDescriptor(conv_desc));  
  MCDNN_CHECK(mcdnnDestory(handle));  
  
  return 0;  
}
