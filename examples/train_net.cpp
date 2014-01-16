// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

//#include <cuda_runtime.h>

#include <cstring>

#include "caffe/caffe.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(ERROR) << "Usage: train_net solver_proto_file [resume_point_file]";
    return 0;
  }

  SolverParameter solver_param;
  ReadProtoFromTextFile(argv[1], &solver_param);

  LOG(INFO) << "Starting Optimization";
  SGDSolver<float> solver(solver_param);
  if (argc == 3) {
    LOG(INFO) << "Resuming from " << argv[2];
    solver.Solve(argv[2]);
  } else if (argc == 4) {
    if (strcmp(argv[2], "NULL") == 0) {
      LOG(INFO) << "Init params from " << argv[3];
      solver.Solve(NULL, argv[3]);
    }
  }
  else {
    LOG(INFO) << "Beginning Solve!";
    solver.Solve();
  }
  LOG(INFO) << "Optimization Done.";

  return 0;
}
