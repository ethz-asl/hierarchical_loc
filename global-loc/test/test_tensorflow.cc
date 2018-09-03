#include <iostream>

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>

using namespace std;
using namespace tensorflow;

int main()
{
    Session* session;
    tensorflow::SessionOptions options = SessionOptions();
    Status status = tensorflow::NewSession(options, &session);
    if (!status.ok()) {
        cout << status.ToString() << endl;
        return 1;
    }
    cout << "Session successfully created." << endl;
}
