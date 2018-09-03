MAPLAB="../catkin_dependencies/maplab"
MAPLAB_COMMON="$MAPLAB/common/maplab-common/proto"

PROTO="../deep-relocalization/proto/deep-relocalization"
protoc -I=$MAPLAB_COMMON:$PROTO --python_out=./ \
    $PROTO/descriptor_index.proto \
    $MAPLAB_COMMON/maplab-common/id.proto \
    $MAPLAB_COMMON/maplab-common/eigen.proto
