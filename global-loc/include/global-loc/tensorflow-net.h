#ifndef GLOBAL_LOC_TENSORFLOW_NET_H_
#define GLOBAL_LOC_TENSORFLOW_NET_H_

#include <vector>
#include <memory>

#include <glog/logging.h>

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/graph_def_builder.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/graph.pb.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

#include <Eigen/Core>

#include <opencv2/opencv.hpp>

using tensorflow::Status;
using tensorflow::Tensor;

class TensorflowNet {
  public:
    typedef Eigen::Matrix<float, Eigen::Dynamic, 1> DescriptorType;

    TensorflowNet(const std::string model_path ,
                  const std::string input_tensor_name,
                  const std::string output_tensor_name):
                input_name_(input_tensor_name), output_name_(output_tensor_name) {
        CHECK(tensorflow::MaybeSavedModelDirectory(model_path));

        // Load model
        Status status = tensorflow::LoadSavedModel(
                tensorflow::SessionOptions(), tensorflow::RunOptions(),
                model_path, {tensorflow::kSavedModelTagServe}, &bundle_);
        if (!status.ok())
            LOG(FATAL) << status.ToString();

        // Check input and output shapes
        tensorflow::GraphDef graph_def = bundle_.meta_graph_def.graph_def();
        bool found_input = false, found_output = false;
        for (auto& node : graph_def.node()) {
            if(node.name() == input_tensor_name) {
                input_channels_ = node.attr().at("shape").shape().dim(3).size();
                found_input = true;
            }
            if(node.name() == output_tensor_name) {
                // Hack as the identity node does not have the shape attribute
                descriptor_size_ = node.attr().at(
                        "_output_shapes").list().shape(0).dim(1).size();
                found_output = true;
            }
        }
        CHECK(found_input) << "Could not find input node " << input_tensor_name;
        CHECK(found_output) << "Could not find output node " << output_tensor_name;
    }

    void PerformInference(const cv::Mat& image, DescriptorType* descriptor) {
        CHECK(image.data);
        CHECK(image.isContinuous());
        CHECK_EQ(image.channels(), input_channels_);

        unsigned height = image.size().height, width = image.size().width;
        // TODO: cleaner way to avoid copying when the image is already float
        // or combine with tensor creation
        cv::Mat *float_image_ptr, tmp;
        if(image.type() != CV_32F) {
            image.convertTo(tmp, CV_32F);
            float_image_ptr = &tmp;
        } else {
            float_image_ptr = &const_cast<cv::Mat&>(image);
        }

        // Prepare input tensor
        Tensor input_tensor(
                tensorflow::DT_FLOAT,
                tensorflow::TensorShape({1, height, width, input_channels_}));
        // TODO: avoid copy if possible
        tensorflow::StringPiece tmp_data = input_tensor.tensor_data();
        std::memcpy(const_cast<char*>(tmp_data.data()), float_image_ptr->data,
                    height * width * input_channels_ * sizeof(float));

        // Run inference
        std::vector<Tensor> outputs;
        Status status = bundle_.session->Run({{input_name_+":0", input_tensor}},
                                             {output_name_+":0"}, {}, &outputs);
        if (!status.ok())
            LOG(FATAL) << status.ToString();

        // Copy result
        float *descriptor_ptr = outputs[0].flat<float>().data();
        Eigen::Map<DescriptorType> descriptor_map(descriptor_ptr, descriptor_size_);
        *descriptor = descriptor_map;  // Copy
    }

    unsigned descriptor_size() {
        return descriptor_size_;
    }

  private:
    tensorflow::SavedModelBundle bundle_;
    unsigned descriptor_size_;
    unsigned input_channels_;
    const std::string input_name_;
    const std::string output_name_;
};

#endif  // GLOBAL_LOC_TENSORFLOW_NET_H_
