//必须包含spp的头文件
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "glog/logging.h"

#include "sppincl.h"
#include "classify_impl.h"
#include "image_download.h"


static image::ClassifyImpl* g_classify_impl;
static image::ImageResource* g_image_resource;
//格式化时间输出
char *format_time(time_t tm);

//初始化方法（可选实现）
//arg1:	配置文件
//arg2:	服务器容器对象
//返回0成功，非0失败
extern "C" int spp_handle_init(void* arg1, void* arg2) {
  //插件自身的配置文件
  const char* etc = (const char*)arg1;
  //服务器容器对象
  CServerBase* base = (CServerBase*)arg2;
  base->log_.log_level(LOG_TRACE);
  base->log_.LOG_P_PID(LOG_ERROR, "spp_handle_init, config:%s, servertype:%d\n", etc, base->servertype());
  if (base->servertype() == SERVER_TYPE_WORKER) {
    ::google::AllowCommandLineReparsing();
    google::ReadFromFlagsFile(etc, program_invocation_name, true);
    //::google::InitGoogleLogging("image_classify");
    g_classify_impl = new image::ClassifyImpl();
    g_image_resource = new image::ImageResource();
    if (!g_image_resource->Init()) {
      LOG(ERROR) << "g_image_resource init error";
      return -1;
    }
    if (!g_classify_impl->Init(g_image_resource)) {
      LOG(ERROR) << "g_classify_impl init error";
      return -1;
    }
  }
  return 0;
}

//数据接收（必须实现）
//flow:	请求包标志
//arg1:	数据块对象
//arg2:	服务器容器对象
//返回值：> 0 表示数据已经接收完整且该值表示数据包的长度
// == 0 表示数据包还未接收完整
// < 0 负数表示出错，将会断开连接
extern "C" int spp_handle_input(unsigned flow, void* arg1, void* arg2) {
  //数据块对象，结构请参考tcommu.h
  blob_type* blob = (blob_type*)arg1;
  //extinfo有扩展信息
  TConnExtInfo* extinfo = (TConnExtInfo*)blob->extdata;
  //服务器容器对象
  CServerBase* base = (CServerBase*)arg2;

  base->log_.LOG_P(LOG_NORMAL, "spp_handle_input[recv time:%s] flow:%d, buffer len:%d, client ip:%s\n",
                   format_time(extinfo->recvtime_),
                   flow,
                   blob->len,
                   inet_ntoa(*(struct in_addr*)&extinfo->remoteip_));

  return blob->len;
}

//路由选择（可选实现）
//flow:	请求包标志
//arg1:	数据块对象
//arg2:	服务器容器对象
//返回值表示worker的组号
extern "C" int spp_handle_route(unsigned flow, void* arg1, void* arg2) {
  //服务器容器对象
  CServerBase* base = (CServerBase*)arg2;
  base->log_.LOG_P_FILE(LOG_NORMAL, "spp_handle_route, flow:%d\n", flow);
  return 1;
}

//数据处理（必须实现）
//flow:	请求包标志
//arg1:	数据块对象
//arg2:	服务器容器对象
//返回0表示成功，非0失败（将会主动断开连接）
extern "C" int spp_handle_process(unsigned flow, void* arg1, void* arg2) {
  //数据块对象，结构请参考tcommu.h
  blob_type* blob = (blob_type*)arg1;
  //数据来源的通讯组件对象
  CTCommu* commu = (CTCommu*)blob->owner;
  //extinfo有扩展信息
  TConnExtInfo* extinfo = (TConnExtInfo*)blob->extdata;
  //服务器容器对象
  CServerBase* base = (CServerBase*)arg2;

  /*
  base->log_.LOG_P_PID(LOG_NORMAL, "spp_handle_process[recv time:%s] flow:%d, buffer len:%d, client ip:%s\n",
                       format_time(extinfo->recvtime_),
                       flow,
                       blob->len,
                       inet_ntoa(*(struct in_addr*)&extinfo->remoteip_));
  */

  int ret = 0;
  do {
    //解析包头消息
    g_classify_impl->request_message_.Clear();
    g_classify_impl->response_message_.Clear();
    LOG(INFO) << "receive data len: " << blob->len;
    if (g_classify_impl->request_message_.ParseFromArray(blob->data, blob->len)) {
      LOG(INFO) << g_classify_impl->request_message_.Utf8DebugString();
    } else {
      LOG(ERROR) << "Request ParseFromString error";
      break;
    }

    // 确定待分类的图片文件名,这个文件名是全路径的.因为client和server是部署在一起的.
    std::string file_name;
    if (!g_classify_impl->request_message_.has_file_name()) {
      //没有传输文件名,则需要根据url下载图片,url的长度需要大于4
      if (g_classify_impl->request_message_.has_image_url() && g_classify_impl->request_message_.image_url().size() > 4) {
        if (!image::ImageDownload::Download(g_classify_impl->request_message_.image_url(), file_name)) {
          LOG(ERROR) << "ImageDownload::Download failed!";
          break;
        }
      } else {
        LOG(ERROR) << "image_url & file_name all miss";
        break;
      }
    } else {
      file_name = g_classify_impl->request_message_.file_name();
    }
    g_classify_impl->response_message_.set_classify_filename(file_name);

    if (g_classify_impl->request_message_.has_request_type()) {
      if (g_classify_impl->request_message_.request_type() == image::ClassifyRequest::CLASSIFY
          || g_classify_impl->request_message_.request_type() == image::ClassifyRequest::CLASSIFY_ALEX) {
        ret = g_classify_impl->ImageClassify(file_name,
                                             g_classify_impl->request_message_.top_n_result(),
                                             g_classify_impl->request_message_.request_type());
      } else if (g_classify_impl->request_message_.request_type() == image::ClassifyRequest::SIMILARITY) {
        if (g_classify_impl->request_message_.has_compare_file_name()) {
          ret = g_classify_impl->ImageSimilarity(g_classify_impl->request_message_.file_name(),
                                                 g_classify_impl->request_message_.compare_file_name());
        }
      } else if (g_classify_impl->request_message_.request_type() == image::ClassifyRequest::SEARCH
                 && FLAGS_open_search_function) {
        ret = g_classify_impl->ImageSearch(file_name,
                                           g_classify_impl->request_message_.top_n_result(),
                                           image::ClassifyRequest::SEARCH);
      } else if (g_classify_impl->request_message_.request_type() == image::ClassifyRequest::CLASSIFY_PAIPAI
                 && FLAGS_open_search_function) {
        ret = g_classify_impl->ImageClassify(file_name,
                                             g_classify_impl->request_message_.top_n_result(),
                                             g_classify_impl->request_message_.request_type());
      } else if (g_classify_impl->request_message_.request_type() == image::ClassifyRequest::SEMANTIC
                 && FLAGS_open_search_function) {
        ret = g_classify_impl->ImageSearch(file_name,
                                           g_classify_impl->request_message_.top_n_result(),
                                           image::ClassifyRequest::SEMANTIC);
      } else {
        LOG(ERROR) << "request_message_ request_type no recognize :" << g_classify_impl->request_message_.request_type();
        break;
      }
    } else {
      LOG(ERROR) << "request_message_ has no request_type";
      break;
    }
    if (ret != 0) {
      LOG(ERROR) << "classify_impl error";
      break;
    }
  } while (0);

  //发送protobuf包
  // LOG(INFO) << g_classify_impl->response_message_.Utf8DebugString();
  std::string send_buf;
  g_classify_impl->response_message_.SerializeToString(&send_buf);
  blob_type rspblob;
  rspblob.data = const_cast<char *>(send_buf.c_str());
  rspblob.len = send_buf.size();
  ret = commu->sendto(flow, &rspblob, arg2);
  if (ret != 0) {
    LOG(ERROR) << "send response error, ret: " << ret;
    return ret;
  }

  return 0;
}

//析构资源（可选实现）
//arg1:	保留参数
//arg2:	服务器容器对象
extern "C" void spp_handle_fini(void* arg1, void* arg2) {
  //服务器容器对象
  CServerBase* base = (CServerBase*)arg2;

  base->log_.LOG_P_PID(LOG_NORMAL, "spp_handle_fini\n");
}

char *format_time(time_t tm) {
  static char str_tm[1024];
  struct tm tmm;
  memset(&tmm, 0, sizeof(tmm));
  localtime_r((time_t *)&tm, &tmm);

  snprintf(str_tm, sizeof(str_tm), "[%04d-%02d-%02d %02d:%02d:%02d]",
           tmm.tm_year + 1900, tmm.tm_mon + 1, tmm.tm_mday,
           tmm.tm_hour, tmm.tm_min, tmm.tm_sec);

  return str_tm;
}
