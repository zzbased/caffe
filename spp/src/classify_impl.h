#ifndef CLASSIFY_IMPL_H
#define CLASSIFY_IMPL_H



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <maxent.h>

#include "image_interface.pb.h"
#include "caffe/caffe.hpp"
#include "image_resource.h"

using namespace caffe;

namespace image {

class ClassifyImpl {
 public:

  ClassifyImpl() : caffe_test_net_(NULL) {};
  ~ClassifyImpl() {
    UnInit();
  };
  bool Init(const ImageResource * resource);

  bool UnInit();

  int ImageClassify(const std::string & filename, int top_n_res = kTopNumber, int class_type = image::ClassifyRequest::CLASSIFY);

  int ImageSearch(const std::string & filename, int top_n_res = kTopNumber, int class_type = image::ClassifyRequest::SEARCH);

  int ImageSimilarity(const std::string & filename, const std::string & filename2);

  ::image::ClassifyRequest request_message_;
  ::image::ClassifyResponse response_message_;

 private:
  const static int kTopNumber = 8;
  // for neural network
  Net<float> * caffe_test_net_;

  // resource
  const ImageResource * image_resource_;

  // maxent
  ME_Model me_model_;

  SimWeightVector similarity_weight_vec_;

  // for search result container
  typedef std::map<int, float> SearchResultContainer;
  typedef std::vector< std::pair<int, float> > SearchResultSortVector;
  SearchResultContainer search_res_container_;
  SearchResultSortVector search_res_sort_vec_;

  int SortSearchResultMap(const SearchResultContainer & search_map, SearchResultSortVector & search_vec);
  int CalcSimilairtyInSearch(const SimWeightVector & other_weight_vec, float & similarity);
};

}

#endif
