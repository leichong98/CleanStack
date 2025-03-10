//===- CleanStackLayout.h - CleanStack frame layout --------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_CLEANSTACKLAYOUT_H
#define LLVM_LIB_CODEGEN_CLEANSTACKLAYOUT_H

#include "CleanStackColoring.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class raw_ostream;
class Value;

namespace cleanstack {

/// Compute the layout of an unclean stack frame.
class StackLayout {
  unsigned MaxAlignment;

  struct StackRegion {
    unsigned Start;
    unsigned End;
    StackColoring::LiveRange Range;

    StackRegion(unsigned Start, unsigned End,
                const StackColoring::LiveRange &Range)
        : Start(Start), End(End), Range(Range) {}
  };

  /// The list of current stack regions, sorted by StackRegion::Start.
  SmallVector<StackRegion, 16> Regions;

  struct StackObject {
    const Value *Handle;
    unsigned Size, Alignment;
    StackColoring::LiveRange Range;
  };

  SmallVector<StackObject, 8> StackObjects;

  DenseMap<const Value *, unsigned> ObjectOffsets;
  DenseMap<const Value *, unsigned> ObjectAlignments;

  void layoutObject(StackObject &Obj);

public:
  StackLayout(unsigned StackAlignment) : MaxAlignment(StackAlignment) {}

  /// Add an object to the stack frame. Value pointer is opaque and used as a
  /// handle to retrieve the object's offset in the frame later.
  void addObject(const Value *V, unsigned Size, unsigned Alignment,
                 const StackColoring::LiveRange &Range);

  /// Run the layout computation for all previously added objects.
  void computeLayout();

  /// Returns the offset to the object start in the stack frame.
  unsigned getObjectOffset(const Value *V) { return ObjectOffsets[V]; }

  /// Returns the alignment of the object
  unsigned getObjectAlignment(const Value *V) { return ObjectAlignments[V]; }

  /// Returns the size of the entire frame.
  unsigned getFrameSize() { return Regions.empty() ? 0 : Regions.back().End; }

  /// Returns the alignment of the frame.
  unsigned getFrameAlignment() { return MaxAlignment; }

  void print(raw_ostream &OS);
};

} // end namespace cleanstack

} // end namespace llvm

#endif // LLVM_LIB_CODEGEN_CLEANSTACKLAYOUT_H