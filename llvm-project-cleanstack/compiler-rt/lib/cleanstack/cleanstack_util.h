//===-- cleanstack_util.h --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains utility code for CleanStack implementation.
//
//===----------------------------------------------------------------------===//

#ifndef CLEANSTACK_UTIL_H
#define CLEANSTACK_UTIL_H

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

namespace cleanstack {

#define SFS_CHECK(a)                                                  \
  do {                                                                \
    if (!(a)) {                                                       \
      fprintf(stderr, "cleanstack CHECK failed: %s:%d %s\n", __FILE__, \
              __LINE__, #a);                                          \
      abort();                                                        \
    };                                                                \
  } while (false)

inline size_t RoundUpTo(size_t size, size_t boundary) {
  SFS_CHECK((boundary & (boundary - 1)) == 0);
  return (size + boundary - 1) & ~(boundary - 1);
}

class MutexLock {
 public:
  explicit MutexLock(pthread_mutex_t &mutex) : mutex_(&mutex) {
    pthread_mutex_lock(mutex_);
  }
  ~MutexLock() { pthread_mutex_unlock(mutex_); }

 private:
  pthread_mutex_t *mutex_ = nullptr;
};

}  // namespace cleanstack

#endif  // CLEANSTACK_UTIL_H