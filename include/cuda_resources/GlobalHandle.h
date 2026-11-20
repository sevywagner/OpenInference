#ifndef IE_GLOBAL_HANDLE
#define IE_GLOBAL_HANDLE

#include <cublasLt.h>

class GlobalHandle {
 private:
  static GlobalHandle *_global_handle;
  cublasLtHandle_t _handle;

  GlobalHandle();
  ~GlobalHandle();
  
 public:
  static cublasLtHandle_t &instance();
};

#endif
