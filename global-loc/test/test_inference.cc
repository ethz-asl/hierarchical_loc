#include <iostream>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "global-loc/tensorflow-net.h"

using namespace std;

int main() {
    string model_path = string(MODEL_ROOT_PATH) + "mobilenetvlad_depth-0.35";
    string image_path = string(DATA_ROOT_PATH) + "images/tango_wet_sample.jpg";
    string input_name = "image", output_name = "descriptor";

    TensorflowNet network(model_path, input_name, output_name);

    cv::Mat image = cv::imread(image_path);
    CHECK_NOTNULL(image.data);
    cvtColor(image, image, cv::COLOR_RGB2GRAY);

    TensorflowNet::DescriptorType descriptor;
    descriptor.resize(network.descriptor_size(), Eigen::NoChange);
    network.PerformInference(image, &descriptor);

    cout << "Inference successfully performed." << endl;
}
