#include <cstring>
#include <cstdlib>
#include <google/gflags.h>
#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include "caffe/caffe.hpp"

using namespace caffe;

/*
struct CaffeNet
{
  CaffeNet(string param_file, string pretrained_param_file) {
    net_.reset(new Net<float>(param_file));
    net_->CopyTrainedLayersFrom(pretrained_param_file);
  }

  virtual ~CaffeNet() {}

  inline void check_array_against_blob(
      PyArrayObject* arr, Blob<float>* blob) {
    CHECK(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS);
    CHECK_EQ(PyArray_NDIM(arr), 4);
    CHECK_EQ(PyArray_ITEMSIZE(arr), 4);
    npy_intp* dims = PyArray_DIMS(arr);
    CHECK_EQ(dims[0], blob->num());
    CHECK_EQ(dims[1], blob->channels());
    CHECK_EQ(dims[2], blob->height());
    CHECK_EQ(dims[3], blob->width());
  }

  // The actual forward function. It takes in a python list of numpy arrays as
  // input and a python list of numpy arrays as output. The input and output
  // should all have correct shapes, are single-precisionabcdnt- and c contiguous.
  void Forward(list bottom, list top) {
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK_EQ(len(bottom), input_blobs.size());
    CHECK_EQ(len(top), net_->num_outputs());
    // First, copy the input
    for (int i = 0; i < input_blobs.size(); ++i) {
      object elem = bottom[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, input_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(input_blobs[i]->mutable_cpu_data(), PyArray_DATA(arr),
            sizeof(float) * input_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(input_blobs[i]->mutable_gpu_data(), PyArray_DATA(arr),
            sizeof(float) * input_blobs[i]->count(), cudaMemcpyHostToDevice);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
    //LOG(INFO) << "Start";
    const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    //LOG(INFO) << "End";
    for (int i = 0; i < output_blobs.size(); ++i) {
      object elem = top[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, output_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(PyArray_DATA(arr), output_blobs[i]->cpu_data(),
            sizeof(float) * output_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(PyArray_DATA(arr), output_blobs[i]->gpu_data(),
            sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }

  void Backward(list top_diff, list bottom_diff) {
    vector<Blob<float>*>& output_blobs = net_->output_blobs();
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK_EQ(len(bottom_diff), input_blobs.size());
    CHECK_EQ(len(top_diff), output_blobs.size());
    // First, copy the output diff
    for (int i = 0; i < output_blobs.size(); ++i) {
      object elem = top_diff[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, output_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(output_blobs[i]->mutable_cpu_diff(), PyArray_DATA(arr),
            sizeof(float) * output_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(output_blobs[i]->mutable_gpu_diff(), PyArray_DATA(arr),
            sizeof(float) * output_blobs[i]->count(), cudaMemcpyHostToDevice);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
    //LOG(INFO) << "Start";
    net_->Backward();
    //LOG(INFO) << "End";
    for (int i = 0; i < input_blobs.size(); ++i) {
      object elem = bottom_diff[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, input_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(PyArray_DATA(arr), input_blobs[i]->cpu_diff(),
            sizeof(float) * input_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(PyArray_DATA(arr), input_blobs[i]->gpu_diff(),
            sizeof(float) * input_blobs[i]->count(), cudaMemcpyDeviceToHost);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }

  // The caffe::Caffe utility functions.
  void set_mode_cpu() { Caffe::set_mode(Caffe::CPU); }
  void set_mode_gpu() { Caffe::set_mode(Caffe::GPU); }
  void set_phase_train() { Caffe::set_phase(Caffe::TRAIN); }
  void set_phase_test() { Caffe::set_phase(Caffe::TEST); }
  void set_device(int device_id) { Caffe::SetDevice(device_id); }

  // The pointer to the internal caffe::Net instant.
  shared_ptr<Net<float> > net_;
};
*/



//different to ReadImageToDatum, this is to read image and convert gray;
static bool ReadImageToGaryDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  cv::Mat cv_img, cv_gray_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
    cv::cvtColor(cv_img, cv_gray_img, CV_BGR2GRAY);
  } else {
    cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(cv_img, cv_gray_img, CV_BGR2GRAY);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  if (height > 0 && width > 0) {

  }
  datum->set_channels(1); //only gray image
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  /*for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }*/
  for (int h = 0; h < cv_img.rows; ++h) {
    for (int w = 0; w < cv_img.cols; ++w) {
      datum_string->push_back(static_cast<char>(cv_img.at<uint8_t>(h, w)));
    }
  }    
  return true;
}


inline static bool ReadImageToGaryDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, datum);
}

DEFINE_string(file, "35590867467.jpg", "The file that contains images.");
DEFINE_int32(row_col_num, 227, "Row/Column number");
DEFINE_string(model_def, "examples/imagenet_deploy.prototxt", "The model definition file.");
DEFINE_string(pretrained_model, "examples/imagenet_model", "The pretrained model.");
DEFINE_string(synset, "data/imagenet_category_name.txt", "The imagenet synset file.");
DEFINE_bool(gpu, false, "use gpu for computation");
DEFINE_int32(similarity_weight_layer, 20, "use this layer data to calc similarity");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  if (argc < 3) {
    LOG(ERROR) << "classify_main test_filename1 test_filename2";
    return -1;
  }
    
  //cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);

  if (FLAGS_gpu) {
    LOG(INFO) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  NetParameter test_net_param;
  ReadProtoFromTextFile(FLAGS_model_def, &test_net_param);
  Net<float> caffe_test_net(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(FLAGS_pretrained_model, &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

  std::ifstream ifs;
  ifs.open(FLAGS_synset.c_str(), std::ifstream::in);
  std::string synset_line;
  std::vector<std::string> name_vector;
  while (getline(ifs, synset_line)) {
    std::size_t found = synset_line.find_last_of("\t");
    std::string name = synset_line.substr(found+1);
    //printf("name:%s\n", name.c_str());
    name_vector.push_back(name);
  }
  //LOG(INFO) << "name_vector.size : " << name_vector.size();
  CHECK_EQ(name_vector.size(), 1000) << "name_vector.size != 1000";
  
  /*
  int total_iter = atoi(argv[3]);
  LOG(ERROR) << "Running " << total_iter << "Iterations.";

  double test_accuracy = 0;
  vector<Blob<float>*> dummy_blob_input_vec;
  for (int i = 0; i < total_iter; ++i) {
    const vector<Blob<float>*>& result =
        caffe_test_net.Forward(dummy_blob_input_vec);
    test_accuracy += result[0]->cpu_data()[0];
    LOG(ERROR) << "Batch " << i << ", accuracy: " << result[0]->cpu_data()[0];
  }
  test_accuracy /= total_iter;
  LOG(ERROR) << "Test accuracy:" << test_accuracy;
  */
  // ====================================================
  // read image testfile1
  Datum datum;
  const static int kUndefinedLabel = 0;
  if (!ReadImageToDatum(argv[1], kUndefinedLabel, FLAGS_row_col_num, FLAGS_row_col_num, &datum)) {
    LOG(ERROR) << "ReadImageToDatum Error"; 
    return -1;
  }
  vector<Blob<float>*>& input_blobs = caffe_test_net.input_blobs();
  //now set input_blobs with datum
  //uint8_t -> float
  //net , data layer, prefetch data
  CHECK_EQ(input_blobs.size(), 1) << "input_blobs.size() != 1";
  Blob<float>* blob = input_blobs[0];
  float* input_data = blob->mutable_cpu_data();
  float scale = 1;
  const string& data = datum.data();
  int datum_size_ = datum.channels() * datum.height() * datum.width();
  //printf("Datum size: %d\n", datum_size_);
  if (data.size()) {
    for (int j = 0; j < datum_size_; ++j) {
      input_data[j] = (static_cast<float>((uint8_t)data[j])) * scale;
    }
  } else {
    for (int j = 0; j < datum_size_; ++j) {
      input_data[j] = (datum.float_data(j)) * scale;
    }
  }
  
  //predict
  const vector<Blob<float>*>&  output_blobs = caffe_test_net.ForwardPrefilled();

  std::cout << "bottom_size: " << caffe_test_net.bottom_vecs_.size() << std::endl;
  std::cout << "top_size: " << caffe_test_net.top_vecs_.size() << std::endl;

  const vector<Blob<float>*>& conv5_out = caffe_test_net.top_vecs_[FLAGS_similarity_weight_layer];
  std::cout << "feature layer [" << FLAGS_similarity_weight_layer << "], size: " << conv5_out.size() << " num: "
  << conv5_out[0]->num() << " count: " << conv5_out[0]->count() << std::endl;
  float *conv5_data = new float[conv5_out[0]->count()];
  const float* conv5_indeed_data = conv5_out[0]->cpu_data();
  for (size_t i=0; i<conv5_out[0]->count(); i++) {
    conv5_data[i] = conv5_indeed_data[i];
  }
  /*
  for (size_t i=0; i<output_blobs.size(); i++) {
      int num = output_blobs[i]->num();
      int dim = output_blobs[i]->count() / output_blobs[i]->num();
      LOG(INFO) << "output_blobs:" << output_blobs[i]->num() << " " << output_blobs[i]->count();
      const float* bottom_data = output_blobs[i]->cpu_data();
      LOG(INFO) << "";
      const static int kTopNumber = 5;
      for (int k = 0; k < num; ++k) {        
        //top 5
        float maxval[kTopNumber] = {-1};
        int max_id[kTopNumber] = {0};
        for (int j = 0; j < dim; ++j) {
          float temp_value = bottom_data[k * dim + j];
          if (temp_value > maxval[0]) {
            for (int d=kTopNumber-1; d>=0; d--) {
              if (maxval[d] < temp_value) {
                for (int d2=0; d2 < d; d2++) {
                  maxval[d2] = maxval[d2+1];
                  max_id[d2] = max_id[d2+1];
                }
                maxval[d] = temp_value;
                max_id[d] = j;
                break;
              }
            }        
          }
        }

        //LOG(INFO) << "Maxid:" << max_id << ",maxval:" << maxval;
        printf("%s\n", argv[1]);
        for (int d=kTopNumber-1; d>=0; d--) {
          printf("Rank%d : %d, %s, %f\n", kTopNumber-d, max_id[d], name_vector[max_id[d]].c_str(), maxval[d]);
        }
        
        //if (max_id == (int)bottom_label[i]) {
        //  ++accuracy;
        //}
        //float prob = max(bottom_data[i * dim + (int)bottom_label[i]], kLOG_THRESHOLD);
        //logprob -= log(prob);
      }
  }
  */

  //==================================================
  // read image testfile2
  datum.Clear();
  if (!ReadImageToDatum(argv[2], kUndefinedLabel, FLAGS_row_col_num, FLAGS_row_col_num, &datum)) {
    LOG(ERROR) << "ReadImageToDatum Error"; 
    return -1;
  }
  input_blobs = caffe_test_net.input_blobs();
  //now set input_blobs with datum
  //uint8_t -> float
  //net , data layer, prefetch data
  CHECK_EQ(input_blobs.size(), 1) << "input_blobs.size() != 1";
  blob = input_blobs[0];
  input_data = blob->mutable_cpu_data();
  datum_size_ = datum.channels() * datum.height() * datum.width();
  //printf("Datum size: %d\n", datum_size_);
  if (datum.data().size()) {
    for (int j = 0; j < datum_size_; ++j) {
      input_data[j] = (static_cast<float>((uint8_t)datum.data()[j])) * scale;
    }
  } else {
    for (int j = 0; j < datum_size_; ++j) {
      input_data[j] = (datum.float_data(j)) * scale;
    }
  }
  const vector<Blob<float>*>&  output_blobs1 = caffe_test_net.ForwardPrefilled();


  const vector<Blob<float>*>& conv5_out_2 = caffe_test_net.top_vecs_[FLAGS_similarity_weight_layer];
  //std::cout << "conv5_out size: " << conv5_out.size() << " output_blobs[i]->num(): "
  //<< conv5_out[0]->num() << " output_blobs[i]->count() " << conv5_out[0]->count() << std::endl;
  float *conv5_data_2 = new float[conv5_out_2[0]->count()];
  const float* conv5_indeed_data_2 = conv5_out_2[0]->cpu_data();
  for (size_t i=0; i<conv5_out_2[0]->count(); i++) {
    conv5_data_2[i] = conv5_indeed_data_2[i];
  }
  //int same_num = 0;
  float ab = 0, a2 = 0, b2 = 0;
  for (int k=0; k<conv5_out_2[0]->count(); k++) {
    //std::cerr << conv5_data[k] << " " << conv5_data_2[k] << std::endl;
    //if (fabs(conv5_data[k] - conv5_data_2[k]) < 1) {
    //  same_num ++;
    //}
    ab += (conv5_data[k]*conv5_data_2[k]);
    a2 += (conv5_data[k]*conv5_data[k]);
    b2 += (conv5_data_2[k]*conv5_data_2[k]);
  }

  std::cout << "cosine :" << ab/sqrt(a2)/sqrt(b2) << std::endl;

  delete [] conv5_data;
  delete [] conv5_data_2;
  //output result
  /*
  LOG(INFO) << "output blobs size:" << output_blobs1.size();
  for (size_t i=0; i<output_blobs1.size(); i++) {
      int num = output_blobs1[i]->num();
      int dim = output_blobs1[i]->count() / output_blobs1[i]->num();
      LOG(INFO) << "output_blobs1:" << output_blobs1[i]->num() << " " << output_blobs1[i]->count();
      const float* bottom_data = output_blobs1[i]->cpu_data();
      LOG(INFO) << "";
      const static int kTopNumber = 5;
      for (int k = 0; k < num; ++k) {        
        //top 5
        float maxval[kTopNumber] = {-1};
        int max_id[kTopNumber] = {0};
        for (int j = 0; j < dim; ++j) {
          float temp_value = bottom_data[k * dim + j];
          if (temp_value > maxval[0]) {
            for (int d=kTopNumber-1; d>=0; d--) {
              if (maxval[d] < temp_value) {
                for (int d2=0; d2 < d; d2++) {
                  maxval[d2] = maxval[d2+1];
                  max_id[d2] = max_id[d2+1];
                }
                maxval[d] = temp_value;
                max_id[d] = j;
                break;
              }
            }        
          }
        }

        //LOG(INFO) << "Maxid:" << max_id << ",maxval:" << maxval;
        printf("%s\n", argv[2]);
        for (int d=kTopNumber-1; d>=0; d--) {
          printf("Rank%d : %d, %s, %f\n", kTopNumber-d, max_id[d], name_vector[max_id[d]].c_str(), maxval[d]);
        }
        
        //if (max_id == (int)bottom_label[i]) {
        //  ++accuracy;
        //}
        //float prob = max(bottom_data[i * dim + (int)bottom_label[i]], kLOG_THRESHOLD);
        //logprob -= log(prob);
      }
  }
  */
  return 0;
}
