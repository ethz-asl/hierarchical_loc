#include <iostream>
#include <fstream>
#include <string>

#include <glog/logging.h>
#include <gflags/gflags.h>
#include <vi-map/vi-map.h>

#include "global-loc/descriptor_index.pb.h"
#include "global-loc/place-retrieval.h"

using namespace std;

DEFINE_string(map_name, "", "Name to the map in `maps/`.");
DEFINE_string(model_name, "", "Name of the Tensorflow model in `models/`.");
DEFINE_string(proto_name, "", "Name of the index protobuf in `data/`.");

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    CHECK(!FLAGS_map_name.empty());
    CHECK(!FLAGS_model_name.empty());
    CHECK(!FLAGS_proto_name.empty());

    string map_path = string(MAP_ROOT_PATH) + FLAGS_map_name;
    string model_path = string(MODEL_ROOT_PATH) + FLAGS_model_name;
    string proto_path = string(DATA_ROOT_PATH) + FLAGS_proto_name;

    vi_map::VIMap map;
    CHECK(map.loadFromFolder(map_path)) << "Loading of the vi-map failed.";

    global_loc::proto::DescriptorIndex proto_index;
    proto_index.set_model_name(FLAGS_model_name);
    proto_index.set_data_name(FLAGS_map_name);

    PlaceRetrieval retrieval(model_path);
    retrieval.BuildIndexFromMap(map, &proto_index);

    fstream output(proto_path, ios::out | ios::trunc | ios::binary);
    CHECK(proto_index.SerializeToOstream(&output));
}
