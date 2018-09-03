#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <vi-map/unique-id.h>
#include <aslam/common/timer.h>

#include "global-loc/descriptor_index.pb.h"
#include "global-loc/place-retrieval.h"

using namespace std;

int main () {
    string model_name = "mobilenetvlad_depth-0.35";
    string model_path = string(MODEL_ROOT_PATH) + model_name;
    string proto_path = string(DATA_ROOT_PATH)
        + "lindenhof_wet_aligned_mobilenet-d0.35.pb";
    string query_path = string(DATA_ROOT_PATH) + "images/tango_afternoon_sample.jpg";

    PlaceRetrieval retrieval(model_path);

    global_loc::proto::DescriptorIndex proto_index;
    fstream input(proto_path, ios::in | ios::binary);
    CHECK(proto_index.ParseFromIstream(&input));
    CHECK_EQ(model_name, proto_index.model_name());
    retrieval.LoadIndex(proto_index);

    cv::Mat query_image = cv::imread(query_path);
    CHECK_NOTNULL(query_image.data);
    cvtColor(query_image, query_image, cv::COLOR_RGB2GRAY);

    vi_map::VisualFrameIdentifierList retrieved_frames;
    retrieval.RetrieveNearestNeighbors(query_image, 10, 1, &retrieved_frames);

    cout << "Retrieved " << retrieved_frames.size() << " frames." << endl;
    timing::Timing::Print(cout);
}
