#include <iostream>
#include <string>

#include <vi-map/vi-map.h>

#include "global-loc/descriptor_index.pb.h"
#include "global-loc/place-retrieval.h"

using namespace std;

int main () {
    string map_path = string(MAP_ROOT_PATH) + "lindenhof_afternoon-wet_aligned";
    string model_path = string(MODEL_ROOT_PATH) + "mobilenetvlad_depth-0.35";

    global_loc::proto::DescriptorIndex proto_index;

    vi_map::VIMap map;
    CHECK(map.loadFromFolder(map_path)) << "Loading of the vi-map failed.";

    PlaceRetrieval retrieval(model_path);
    retrieval.BuildIndexFromMap(map, &proto_index);

    cout << "Processed " << proto_index.frames_size() << " frames." << endl;
}
