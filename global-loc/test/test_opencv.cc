#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

using namespace std;

int main () {
    string image_path = string(DATA_ROOT_PATH) + "images/tango_wet_sample.jpg";
    cout << image_path << endl;
    cv::Mat image = cv::imread(image_path);
    if(!image.data) {
        cout <<  " No image data." << endl;
        return -1;
    }
    cout << "Image size: " << image.size() << endl;
}
