

#ifndef CLASSIFY_IMPL_H
#define CLASSIFY_IMPL_H

#include <iostream>
#include "image_interface.pb.h"

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
#include <maxent.h>
#include "caffe/caffe.hpp"

using namespace caffe;


	
class ClassifyImpl {
public:

	ClassifyImpl():caffe_test_net_(NULL) {};
	~ClassifyImpl() { UnInit();};
	int Init();

	int UnInit();

	int ImageClassify(const std::string & filename, int top_n_res = kTopNumber, int class_type = image::ClassifyRequest::CLASSIFY);

	int ImageSearch(const std::string & filename, int top_n_res = kTopNumber);

	int ImageSimilarity(const std::string & filename, const std::string & filename2);

	::image::ClassifyRequest request_message_;
	::image::ClassifyResponse response_message_;

	typedef std::vector<size_t> PostingList;
	typedef std::map<int, PostingList> ImageCategoryIndex;
	typedef std::vector<float> SimWeightVector;
	typedef std::vector<SimWeightVector> SimWeightIndex;
	typedef std::vector<std::string> IdToFileName;
	typedef std::map<std::string, size_t> ImageUniquer;
private:
	const static int kTopNumber = 8;
	Net<float> * caffe_test_net_;
	ME_Model me_model_;		
	std::vector<std::string> name_vector_;
	std::vector<std::string> paipai_name_vector_;

	SimWeightVector similarity_weight_vec_;

	int LoadClassNameVector();
	// for index
	int LoadImageIndex(const char* filename);

	ImageUniquer image_uniquer_;
	IdToFileName index_id_to_filename_;
	SimWeightIndex sim_weight_index_;
	ImageCategoryIndex image_class_index_;


	// for search result container
	typedef std::map<int, float> SearchResultContainer;
	typedef std::vector< std::pair<int, float> > SearchResultSortVector;
	SearchResultContainer search_res_container_;
	SearchResultSortVector search_res_sort_vec_;

	int SortSearchResultMap(const SearchResultContainer & search_map, SearchResultSortVector & search_vec);
	int CalcSimilairtyInSearch(int image_id, float & similarity);
};

#endif
