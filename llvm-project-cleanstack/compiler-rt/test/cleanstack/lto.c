// REQUIRES: lto

// RUN: %clang_lto_cleanstack %s -o %t
// RUN: %run %t

// Test that clean stack works with LTO.

int main() {
  char c[] = "hello world";
  puts(c);
  return 0;
