//===-- cleanstack.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the runtime support for the unclean stack protection
// mechanism. The runtime manages allocation/deallocation of the unclean stack
// for the main thread, as well as all pthreads that are created/destroyed
// during program execution.
//
//===----------------------------------------------------------------------===//

#include "cleanstack_platform.h"
#include "cleanstack_util.h"

#include <errno.h>
#include <sys/resource.h>

#include "interception/interception.h"

using namespace cleanstack;

// TODO: To make accessing the unclean stack pointer faster, we plan to
// eventually store it directly in the thread control block data structure on
// platforms where this structure is pointed to by %fs or %gs. This is exactly
// the same mechanism as currently being used by the traditional stack
// protector pass to store the stack guard (see getStackCookieLocation()
// function above). Doing so requires changing the tcbhead_t struct in glibc
// on Linux and tcb struct in libc on FreeBSD.
//
// For now, store it in a thread-local variable.
extern "C" {
__attribute__((visibility(
    "default"))) __thread void *__cleanstack_unclean_stack_ptr = nullptr;
}

namespace {

// TODO: The runtime library does not currently protect the unclean stack beyond
// relying on the system-enforced ASLR. The protection of the (unclean) stack can
// be provided by three alternative features:
//
// 1) Protection via hardware segmentation on x86-32 and some x86-64
// architectures: the (unclean) stack segment (implicitly accessed via the %ss
// segment register) can be separated from the data segment (implicitly
// accessed via the %ds segment register). Dereferencing a pointer to the safe
// segment would result in a segmentation fault.
//
// 2) Protection via software fault isolation: memory writes that are not meant
// to access the clean stack can be prevented from doing so through runtime
// instrumentation. One way to do it is to allocate the clean stack(s) in the
// upper half of the userspace and bitmask the corresponding upper bit of the
// memory addresses of memory writes that are not meant to access the clean
// stack.
//
// 3) Protection via information hiding on 64 bit architectures: the location
// of the clean stack(s) can be randomized through secure mechanisms, and the
// leakage of the stack pointer can be prevented. Currently, libc can leak the
// stack pointer in several ways (e.g. in longjmp, signal handling, user-level
// context switching related functions, etc.). These can be fixed in libc and
// in other low-level libraries, by either eliminating the escaping/dumping of
// the stack pointer (i.e., %rsp) when that's possible, or by using
// encryption/PTR_MANGLE (XOR-ing the dumped stack pointer with another secret
// we control and protect better, as is already done for setjmp in glibc.)
// Furthermore, a static machine code level verifier can be ran after code
// generation to make sure that the stack pointer is never written to memory,
// or if it is, its written on the clean stack.
//
// Finally, while the unclean Stack pointer is currently stored in a thread
// local variable, with libc support it could be stored in the TCB (thread
// control block) as well, eliminating another level of indirection and making
// such accesses faster. Alternatively, dedicating a separate register for
// storing it would also be possible.

/// Minimum stack alignment for the unclean stack.
const unsigned kStackAlign = 16;

/// Default size of the clean stack. This value is only used if the stack
/// size rlimit is set to infinity.
const unsigned kDefaultUncleanStackSize = 0x2800000;

// Per-thread unclean stack information. It's not frequently accessed, so there
// it can be kept out of the tcb in normal thread-local variables.
__thread void *unclean_stack_start = nullptr;
__thread size_t unclean_stack_size = 0;
__thread size_t unclean_stack_guard = 0;

inline void *unclean_stack_alloc(size_t size, size_t guard) {
  SFS_CHECK(size + guard >= size);
  void *addr = Mmap(nullptr, size + guard, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON, -1, 0);
  SFS_CHECK(MAP_FAILED != addr);
  Mprotect(addr, guard, PROT_NONE);
  return (char *)addr + guard;
}

inline void unclean_stack_setup(void *start, size_t size, size_t guard) {
  SFS_CHECK((char *)start + size >= (char *)start);
  SFS_CHECK((char *)start + guard >= (char *)start);
  void *stack_ptr = (char *)start + size;
  SFS_CHECK((((size_t)stack_ptr) & (kStackAlign - 1)) == 0);

  __cleanstack_unclean_stack_ptr = stack_ptr;
  unclean_stack_start = start;
  unclean_stack_size = size;
  unclean_stack_guard = guard;
}

/// Thread data for the cleanup handler
pthread_key_t thread_cleanup_key;

/// clean stack per-thread information passed to the thread_start function
struct tinfo {
  void *(*start_routine)(void *);
  void *start_routine_arg;

  void *unclean_stack_start;
  size_t unclean_stack_size;
  size_t unclean_stack_guard;
};

/// Wrap the thread function in order to deallocate the unclean stack when the
/// thread terminates by returning from its main function.
void *thread_start(void *arg) {
  struct tinfo *tinfo = (struct tinfo *)arg;

  void *(*start_routine)(void *) = tinfo->start_routine;
  void *start_routine_arg = tinfo->start_routine_arg;

  // Setup the unclean stack; this will destroy tinfo content
  unclean_stack_setup(tinfo->unclean_stack_start, tinfo->unclean_stack_size,
                     tinfo->unclean_stack_guard);

  // Make sure out thread-specific destructor will be called
  pthread_setspecific(thread_cleanup_key, (void *)1);

  return start_routine(start_routine_arg);
}

/// Linked list used to store exiting threads stack/thread information.
struct thread_stack_ll {
  struct thread_stack_ll *next;
  void *stack_base;
  size_t size;
  pid_t pid;
  ThreadId tid;
};

/// Linked list of unclean stacks for threads that are exiting. We delay
/// unmapping them until the thread exits.
thread_stack_ll *thread_stacks = nullptr;
pthread_mutex_t thread_stacks_mutex = PTHREAD_MUTEX_INITIALIZER;

/// Thread-specific data destructor. We want to free the unclean stack only after
/// this thread is terminated. libc can call functions in cleanstack-instrumented
/// code (like free) after thread-specific data destructors have run.
void thread_cleanup_handler(void *_iter) {
  SFS_CHECK(unclean_stack_start != nullptr);
  pthread_setspecific(thread_cleanup_key, NULL);

  pthread_mutex_lock(&thread_stacks_mutex);
  // Temporary list to hold the previous threads stacks so we don't hold the
  // thread_stacks_mutex for long.
  thread_stack_ll *temp_stacks = thread_stacks;
  thread_stacks = nullptr;
  pthread_mutex_unlock(&thread_stacks_mutex);

  pid_t pid = getpid();
  ThreadId tid = GetTid();

  // Free stacks for dead threads
  thread_stack_ll **stackp = &temp_stacks;
  while (*stackp) {
    thread_stack_ll *stack = *stackp;
    if (stack->pid != pid ||
        (-1 == TgKill(stack->pid, stack->tid, 0) && errno == ESRCH)) {
      Munmap(stack->stack_base, stack->size);
      *stackp = stack->next;
      free(stack);
    } else
      stackp = &stack->next;
  }

  thread_stack_ll *cur_stack =
      (thread_stack_ll *)malloc(sizeof(thread_stack_ll));
  cur_stack->stack_base = (char *)unclean_stack_start - unclean_stack_guard;
  cur_stack->size = unclean_stack_size + unclean_stack_guard;
  cur_stack->pid = pid;
  cur_stack->tid = tid;

  pthread_mutex_lock(&thread_stacks_mutex);
  // Merge thread_stacks with the current thread's stack and any remaining
  // temp_stacks
  *stackp = thread_stacks;
  cur_stack->next = temp_stacks;
  thread_stacks = cur_stack;
  pthread_mutex_unlock(&thread_stacks_mutex);

  unclean_stack_start = nullptr;
}

void EnsureInterceptorsInitialized();

/// Intercept thread creation operation to allocate and setup the unclean stack
INTERCEPTOR(int, pthread_create, pthread_t *thread,
            const pthread_attr_t *attr,
            void *(*start_routine)(void*), void *arg) {
  EnsureInterceptorsInitialized();
  size_t size = 0;
  size_t guard = 0;

  if (attr) {
    pthread_attr_getstacksize(attr, &size);
    pthread_attr_getguardsize(attr, &guard);
  } else {
    // get pthread default stack size
    pthread_attr_t tmpattr;
    pthread_attr_init(&tmpattr);
    pthread_attr_getstacksize(&tmpattr, &size);
    pthread_attr_getguardsize(&tmpattr, &guard);
    pthread_attr_destroy(&tmpattr);
  }

  SFS_CHECK(size);
  size = RoundUpTo(size, kStackAlign);

  void *addr = unclean_stack_alloc(size, guard);
  // Put tinfo at the end of the buffer. guard may be not page aligned.
  // If that is so then some bytes after addr can be mprotected.
  struct tinfo *tinfo =
      (struct tinfo *)(((char *)addr) + size - sizeof(struct tinfo));
  tinfo->start_routine = start_routine;
  tinfo->start_routine_arg = arg;
  tinfo->unclean_stack_start = addr;
  tinfo->unclean_stack_size = size;
  tinfo->unclean_stack_guard = guard;

  return REAL(pthread_create)(thread, attr, thread_start, tinfo);
}

pthread_mutex_t interceptor_init_mutex = PTHREAD_MUTEX_INITIALIZER;
bool interceptors_inited = false;

void EnsureInterceptorsInitialized() {
  MutexLock lock(interceptor_init_mutex);
  if (interceptors_inited)
    return;

  // Initialize pthread interceptors for thread allocation
  INTERCEPT_FUNCTION(pthread_create);

  interceptors_inited = true;
}

}  // namespace

extern "C" __attribute__((visibility("default")))
#if !SANITIZER_CAN_USE_PREINIT_ARRAY
// On ELF platforms, the constructor is invoked using .preinit_array (see below)
__attribute__((constructor(0)))
#endif
void __cleanstack_init() {
  // Determine the stack size for the main thread.
  size_t size = kDefaultUncleanStackSize;
  size_t guard = 4096;

  struct rlimit limit;
  if (getrlimit(RLIMIT_STACK, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY)
    size = limit.rlim_cur;

  // Allocate unclean stack for main thread
  void *addr = unclean_stack_alloc(size, guard);
  unclean_stack_setup(addr, size, guard);

  // Setup the cleanup handler
  pthread_key_create(&thread_cleanup_key, thread_cleanup_handler);
}

#if SANITIZER_CAN_USE_PREINIT_ARRAY
// On ELF platforms, run cleanstack initialization before any other constructors.
// On other platforms we use the constructor attribute to arrange to run our
// initialization early.
extern "C" {
__attribute__((section(".preinit_array"),
               used)) void (*__cleanstack_preinit)(void) = __cleanstack_init;
}
#endif

extern "C"
    __attribute__((visibility("default"))) void *__get_unclean_stack_bottom() {
  return unclean_stack_start;
}

extern "C"
    __attribute__((visibility("default"))) void *__get_unclean_stack_top() {
  return (char*)unclean_stack_start + unclean_stack_size;
}

extern "C"
    __attribute__((visibility("default"))) void *__get_unclean_stack_start() {
  return unclean_stack_start;
}

extern "C"
    __attribute__((visibility("default"))) void *__get_unclean_stack_ptr() {
  return __cleanstack_unclean_stack_ptr;
}