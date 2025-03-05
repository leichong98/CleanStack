//===- CleanStack.cpp - Clean Stack Insertion -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass splits the stack into the clean stack (kept as-is for LLVM backend)
// and the unclean stack (explicitly allocated and managed through the runtime
// support library).
//
// http://clang.llvm.org/docs/CleanStack.html
//
//===----------------------------------------------------------------------===//

#include "CleanStackColoring.h"
#include "CleanStackLayout.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <utility>

using namespace llvm;
using namespace llvm::cleanstack;

#define DEBUG_TYPE "clean-stack"

namespace llvm {

STATISTIC(NumFunctions, "Total number of functions");
STATISTIC(NumUncleanStackFunctions, "Number of functions with unclean stack");
STATISTIC(NumUncleanStackRestorePointsFunctions,
          "Number of functions that use setjmp or exceptions");

STATISTIC(NumAllocas, "Total number of allocas");
STATISTIC(NumUncleanStaticAllocas, "Number of unclean static allocas");
STATISTIC(NumUncleanDynamicAllocas, "Number of unclean dynamic allocas");
STATISTIC(NumUncleanByValArguments, "Number of unclean byval arguments");
STATISTIC(NumUncleanStackRestorePoints, "Number of setjmps and landingpads");

} // namespace llvm

/// Use __cleanstack_pointer_address even if the platform has a faster way of
/// access clean stack pointer.
static cl::opt<bool>
    CleanStackUsePointerAddress("cleanstack-use-pointer-address",
                                  cl::init(false), cl::Hidden);


namespace {

/// Rewrite an SCEV expression for a memory access address to an expression that
/// represents offset from the given alloca.
///
/// The implementation simply replaces all mentions of the alloca with zero.
class AllocaOffsetRewriter : public SCEVRewriteVisitor<AllocaOffsetRewriter> {
  const Value *AllocaPtr;

public:
  AllocaOffsetRewriter(ScalarEvolution &SE, const Value *AllocaPtr)
      : SCEVRewriteVisitor(SE), AllocaPtr(AllocaPtr) {}

  const SCEV *visitUnknown(const SCEVUnknown *Expr) {
    if (Expr->getValue() == AllocaPtr)
      return SE.getZero(Expr->getType());
    return Expr;
  }
};

/// The CleanStack pass splits the stack of each function into the clean
/// stack, which is only accessed through unclean stack object (as
/// determined statically), and the unclean stack, which contains all
/// local variables that are accessed in ways that we can't prove to
/// be clean.
class CleanStack {
  Function &F;
  const TargetLoweringBase &TL;
  const DataLayout &DL;
  ScalarEvolution &SE;

  Type *StackPtrTy;
  Type *IntPtrTy;
  Type *Int32Ty;
  Type *Int8Ty;

  Value *UncleanStackPtr = nullptr;

  /// Unclean stack alignment. Each stack frame must ensure that the stack is
  /// aligned to this value. We need to re-align the unclean stack if the
  /// alignment of any object on the stack exceeds this value.
  ///
  /// 16 seems like a reasonable upper bound on the alignment of objects that we
  /// might expect to appear on the stack on most common targets.
  enum { StackAlignment = 16 };

  /// Return the value of the stack canary.
  Value *getStackGuard(IRBuilder<> &IRB, Function &F);

  /// Load stack guard from the frame and check if it has changed.
  void checkStackGuard(IRBuilder<> &IRB, Function &F, ReturnInst &RI,
                       AllocaInst *StackGuardSlot, Value *StackGuard);

  /// Find all static allocas, dynamic allocas, return instructions and
  /// stack restore points (exception unwind blocks and setjmp calls) in the
  /// given function and append them to the respective vectors.
  void findInsts(Function &F, SmallVectorImpl<AllocaInst *> &StaticAllocas,
                 SmallVectorImpl<AllocaInst *> &DynamicAllocas,
                 SmallVectorImpl<Argument *> &ByValArguments,
                 SmallVectorImpl<ReturnInst *> &Returns,
                 SmallVectorImpl<Instruction *> &StackRestorePoints);

  /// Calculate the allocation size of a given alloca. Returns 0 if the
  /// size can not be statically determined.
  uint64_t getStaticAllocaAllocationSize(const AllocaInst* AI);

  /// Allocate space for all static allocas in \p StaticAllocas,
  /// replace allocas with pointers into the unclean stack and generate code to
  /// restore the stack pointer before all return instructions in \p Returns.
  ///
  /// \returns A pointer to the top of the unclean stack after all unclean static
  /// allocas are allocated.
  Value *moveStaticAllocasToUncleanStack(IRBuilder<> &IRB, Function &F,
                                        ArrayRef<AllocaInst *> StaticAllocas,
                                        ArrayRef<Argument *> ByValArguments,
                                        Instruction *BasePointer,
                                        AllocaInst *StackGuardSlot);

  /// Generate code to restore the stack after all stack restore points
  /// in \p StackRestorePoints.
  ///
  /// \returns A local variable in which to maintain the dynamic top of the
  /// unclean stack if needed.
  AllocaInst *
  createStackRestorePoints(IRBuilder<> &IRB, Function &F,
                           ArrayRef<Instruction *> StackRestorePoints,
                           Value *StaticTop, bool NeedDynamicTop);

  /// Replace all allocas in \p DynamicAllocas with code to allocate
  /// space dynamically on the unclean stack and store the dynamic unclean stack
  /// top to \p DynamicTop if non-null.
  void moveDynamicAllocasToUncleanStack(Function &F, Value *UncleanStackPtr,
                                       AllocaInst *DynamicTop,
                                       ArrayRef<AllocaInst *> DynamicAllocas);

  bool ContainsProtectableArray(Type *Ty, bool InStruct) const;

  bool HasAddressTaken(const Instruction *AI, SmallPtrSet<const PHINode *, 16> &VisitedPHIs);

  bool IsUncleanStackAlloca(const Value *AllocaPtr);

  bool ShouldInlinePointerAddress(CallSite &CS);  
  void TryInlinePointerAddress();

public:
  CleanStack(Function &F, const TargetLoweringBase &TL, const DataLayout &DL,
            ScalarEvolution &SE)
      : F(F), TL(TL), DL(DL), SE(SE),
        StackPtrTy(Type::getInt8PtrTy(F.getContext())),
        IntPtrTy(DL.getIntPtrType(F.getContext())),
        Int32Ty(Type::getInt32Ty(F.getContext())),
        Int8Ty(Type::getInt8Ty(F.getContext())) {}

  // Run the transformation on the associated function.
  // Returns whether the function was changed.
  bool run();
};

uint64_t CleanStack::getStaticAllocaAllocationSize(const AllocaInst* AI) {
  uint64_t Size = DL.getTypeAllocSize(AI->getAllocatedType());
  if (AI->isArrayAllocation()) {
    auto C = dyn_cast<ConstantInt>(AI->getArraySize());
    if (!C)
      return 0;
    Size *= C->getZExtValue();
  }
  return Size;
}

bool CleanStack::ContainsProtectableArray(Type *Ty, bool InStruct) const {
  if (!Ty)
    return false;

  if (isa<ArrayType>(Ty)) {
      return true;
  }

  const StructType *ST = dyn_cast<StructType>(Ty);

  if (!ST)
    return false;

  for (Type *ET : ST->elements())
    if (ContainsProtectableArray(ET, true)) {
      return true;
    }

  return false;
}
  
bool CleanStack::HasAddressTaken(const Instruction *AI, SmallPtrSet<const PHINode *, 16> &VisitedPHIs) {
  for (const User *U : AI->users()) {
    const auto *I = cast<Instruction>(U);
    switch (I->getOpcode()) {
    case Instruction::Store:
      if (AI == cast<StoreInst>(I)->getValueOperand())
        return true;
      break;
    case Instruction::AtomicCmpXchg:
      // cmpxchg conceptually includes both a load and store from the same
      // location. So, like store, the value being stored is what matters.
      if (AI == cast<AtomicCmpXchgInst>(I)->getNewValOperand())
        return true;
      break;
    case Instruction::PtrToInt:
      if (AI == cast<PtrToIntInst>(I)->getOperand(0))
        return true;
      break;
    case Instruction::Call: {
      // Ignore intrinsics that do not become real instructions.
      // TODO: Narrow this to intrinsics that have store-like effects.
      const auto *CI = cast<CallInst>(I);
      if (!isa<DbgInfoIntrinsic>(CI) && !CI->isLifetimeStartOrEnd())
        return true;
      break;
    }
    case Instruction::Invoke:
    case Instruction::Ret:
      return true;
    case Instruction::BitCast:
    case Instruction::GetElementPtr:
    case Instruction::Select:
    case Instruction::AddrSpaceCast:
      if (HasAddressTaken(I, VisitedPHIs))
        return true;
      break;
    case Instruction::PHI: {
      // Keep track of what PHI nodes we have already visited to ensure
      // they are only visited once.
      const auto *PN = cast<PHINode>(I);
      if (VisitedPHIs.insert(PN).second)
        if (HasAddressTaken(PN, VisitedPHIs))
          return true;
      break;
    }
    case Instruction::Load:
    case Instruction::AtomicRMW:
      // These instructions take an address operand, but have load-like or
      // other innocuous behavior that should not trigger a protector.
      // atomicrmw conceptually has both load and store semantics, but the
      // value being stored must be integer; so if a pointer is being stored,
      // we'll catch it in the PtrToInt case above.
      break;
    default:
      // Conservatively return true for any instruction that takes an address
      // operand, but is not handled above.
      return true;
    }
  }
 return false;
}

/// Check whether a given allocation should be put on the unclean
/// stack or not. The function analyzes all uses of AI and checks whether it is
/// only accessed in a unclean way (as decided statically).
bool CleanStack::IsUncleanStackAlloca(const Value *AllocaPtr) {
  // Go through all uses of this alloca and check whether all accesses to the
  // allocated object are statically known to be unclean and, hence, the
  // object can be placed on the unclean stack.
      bool IsUnclean = false;
  // The set of PHI nodes visited when determining if a variable's reference has
  // been taken.  This set is maintained to ensure we don't visit the same PHI
  // node multiple times.
      SmallPtrSet<const PHINode *, 16> VisitedPHIs;
      if (const AllocaInst *AI = dyn_cast<AllocaInst>(AllocaPtr)) {
        if (AI->isArrayAllocation()) {
          IsUnclean = true;
        }
        if (ContainsProtectableArray(AI->getAllocatedType(), false)) {
          IsUnclean = true;
        }
        if (HasAddressTaken(AI, VisitedPHIs)) {
          IsUnclean = true;
        }
        // Clear any PHIs that we visited, to make sure we examine all uses of
        // any subsequent allocas that we look at.
        VisitedPHIs.clear();
      }

  return IsUnclean;
}

Value *CleanStack::getStackGuard(IRBuilder<> &IRB, Function &F) {
  Value *StackGuardVar = TL.getIRStackGuard(IRB);
  if (!StackGuardVar)
    StackGuardVar =
        F.getParent()->getOrInsertGlobal("__stack_chk_guard", StackPtrTy);
  return IRB.CreateLoad(StackPtrTy, StackGuardVar, "StackGuard");
}

void CleanStack::findInsts(Function &F,
                          SmallVectorImpl<AllocaInst *> &StaticAllocas,
                          SmallVectorImpl<AllocaInst *> &DynamicAllocas,
                          SmallVectorImpl<Argument *> &ByValArguments,
                          SmallVectorImpl<ReturnInst *> &Returns,
                          SmallVectorImpl<Instruction *> &StackRestorePoints) {
  for (Instruction &I : instructions(&F)) {
    if (auto AI = dyn_cast<AllocaInst>(&I)) {
      ++NumAllocas;

      uint64_t Size = getStaticAllocaAllocationSize(AI);
      if (!IsUncleanStackAlloca(AI))
        continue;

      if (AI->isStaticAlloca()) {
        ++NumUncleanStaticAllocas;
        StaticAllocas.push_back(AI);
      } else {
        ++NumUncleanDynamicAllocas;
        DynamicAllocas.push_back(AI);
      }
    } else if (auto RI = dyn_cast<ReturnInst>(&I)) {
      Returns.push_back(RI);
    } else if (auto CI = dyn_cast<CallInst>(&I)) {
      // setjmps require stack restore.
      if (CI->getCalledFunction() && CI->canReturnTwice())
        StackRestorePoints.push_back(CI);
    } else if (auto LP = dyn_cast<LandingPadInst>(&I)) {
      // Exception landing pads require stack restore.
      StackRestorePoints.push_back(LP);
    } else if (auto II = dyn_cast<IntrinsicInst>(&I)) {
      if (II->getIntrinsicID() == Intrinsic::gcroot)
        report_fatal_error(
            "gcroot intrinsic not compatible with cleanstack attribute");
    }
  }
  for (Argument &Arg : F.args()) {
    if (!Arg.hasByValAttr())
      continue;
    uint64_t Size =
        DL.getTypeStoreSize(Arg.getType()->getPointerElementType());
    if (!IsUncleanStackAlloca(&Arg))
      continue;

    ++NumUncleanByValArguments;
    ByValArguments.push_back(&Arg);
  }
}

AllocaInst *
CleanStack::createStackRestorePoints(IRBuilder<> &IRB, Function &F,
                                    ArrayRef<Instruction *> StackRestorePoints,
                                    Value *StaticTop, bool NeedDynamicTop) {
  assert(StaticTop && "The stack top isn't set.");

  if (StackRestorePoints.empty())
    return nullptr;

  // We need the current value of the shadow stack pointer to restore
  // after longjmp or exception catching.

  // FIXME: On some platforms this could be handled by the longjmp/exception
  // runtime itself.

  AllocaInst *DynamicTop = nullptr;
  if (NeedDynamicTop) {
    // If we also have dynamic alloca's, the stack pointer value changes
    // throughout the function. For now we store it in an alloca.
    DynamicTop = IRB.CreateAlloca(StackPtrTy, /*ArraySize=*/nullptr,
                                  "unclean_stack_dynamic_ptr");
    IRB.CreateStore(StaticTop, DynamicTop);
  }

  // Restore current stack pointer after longjmp/exception catch.
  for (Instruction *I : StackRestorePoints) {
    ++NumUncleanStackRestorePoints;

    IRB.SetInsertPoint(I->getNextNode());
    Value *CurrentTop =
        DynamicTop ? IRB.CreateLoad(StackPtrTy, DynamicTop) : StaticTop;
    IRB.CreateStore(CurrentTop, UncleanStackPtr);
  }

  return DynamicTop;
}

void CleanStack::checkStackGuard(IRBuilder<> &IRB, Function &F, ReturnInst &RI,
                                AllocaInst *StackGuardSlot, Value *StackGuard) {
  Value *V = IRB.CreateLoad(StackPtrTy, StackGuardSlot);
  Value *Cmp = IRB.CreateICmpNE(StackGuard, V);

  auto SuccessProb = BranchProbabilityInfo::getBranchProbStackProtector(true);
  auto FailureProb = BranchProbabilityInfo::getBranchProbStackProtector(false);
  MDNode *Weights = MDBuilder(F.getContext())
                        .createBranchWeights(SuccessProb.getNumerator(),
                                             FailureProb.getNumerator());
  Instruction *CheckTerm =
      SplitBlockAndInsertIfThen(Cmp, &RI,
                                /* Unreachable */ true, Weights);
  IRBuilder<> IRBFail(CheckTerm);
  // FIXME: respect -fsanitize-trap / -ftrap-function here?
  FunctionCallee StackChkFail =
      F.getParent()->getOrInsertFunction("__stack_chk_fail", IRB.getVoidTy());
  IRBFail.CreateCall(StackChkFail, {});
}

/// We explicitly compute and set the unclean stack layout for all unclean
/// static alloca instructions. We save the unclean "base pointer" in the
/// prologue into a local variable and restore it in the epilogue.
Value *CleanStack::moveStaticAllocasToUncleanStack(
    IRBuilder<> &IRB, Function &F, ArrayRef<AllocaInst *> StaticAllocas,
    ArrayRef<Argument *> ByValArguments, Instruction *BasePointer, 
    AllocaInst *StackGuardSlot) {
  if (StaticAllocas.empty() && ByValArguments.empty())
    return BasePointer;

  DIBuilder DIB(*F.getParent());

  StackColoring SSC(F, StaticAllocas);
  SSC.run();
  SSC.removeAllMarkers();

  // Unclean stack always grows down.
  StackLayout SSL(StackAlignment);
  if (StackGuardSlot) {
    Type *Ty = StackGuardSlot->getAllocatedType();
    unsigned Align =
        std::max(DL.getPrefTypeAlignment(Ty), StackGuardSlot->getAlignment());
    SSL.addObject(StackGuardSlot, getStaticAllocaAllocationSize(StackGuardSlot),
                  Align, SSC.getFullLiveRange());
  }

  for (Argument *Arg : ByValArguments) {
    Type *Ty = Arg->getType()->getPointerElementType();
    uint64_t Size = DL.getTypeStoreSize(Ty);
    if (Size == 0)
      Size = 1; // Don't create zero-sized stack objects.

    // Ensure the object is properly aligned.
    unsigned Align = std::max((unsigned)DL.getPrefTypeAlignment(Ty),
                              Arg->getParamAlignment());
    SSL.addObject(Arg, Size, Align, SSC.getFullLiveRange());
  }

  for (AllocaInst *AI : StaticAllocas) {
    Type *Ty = AI->getAllocatedType();
    uint64_t Size = getStaticAllocaAllocationSize(AI);
    if (Size == 0)
      Size = 1; // Don't create zero-sized stack objects.

    // Ensure the object is properly aligned.
    unsigned Align =
        std::max((unsigned)DL.getPrefTypeAlignment(Ty), AI->getAlignment());

    SSL.addObject(AI, Size, Align, SSC.getLiveRange(AI));
  }

  SSL.computeLayout();
  unsigned FrameAlignment = SSL.getFrameAlignment();

  // FIXME: tell SSL that we start at a less-then-MaxAlignment aligned location
  // (AlignmentSkew).
  if (FrameAlignment > StackAlignment) {
    // Re-align the base pointer according to the max requested alignment.
    assert(isPowerOf2_32(FrameAlignment));
    IRB.SetInsertPoint(BasePointer->getNextNode());
    BasePointer = cast<Instruction>(IRB.CreateIntToPtr(
        IRB.CreateAnd(IRB.CreatePtrToInt(BasePointer, IntPtrTy),
                      ConstantInt::get(IntPtrTy, ~uint64_t(FrameAlignment - 1))),
        StackPtrTy));
  }

  IRB.SetInsertPoint(BasePointer->getNextNode());

  if (StackGuardSlot) {
    unsigned Offset = SSL.getObjectOffset(StackGuardSlot);
    Value *Off = IRB.CreateGEP(Int8Ty, BasePointer, // BasePointer is i8*
                               ConstantInt::get(Int32Ty, -Offset));
    Value *NewAI =
        IRB.CreateBitCast(Off, StackGuardSlot->getType(), "StackGuardSlot");

    // Replace alloc with the new location.
    StackGuardSlot->replaceAllUsesWith(NewAI);
    StackGuardSlot->eraseFromParent();
  }

  for (Argument *Arg : ByValArguments) {
    unsigned Offset = SSL.getObjectOffset(Arg);
    MaybeAlign Align(SSL.getObjectAlignment(Arg));
    Type *Ty = Arg->getType()->getPointerElementType();

    uint64_t Size = DL.getTypeStoreSize(Ty);
    if (Size == 0)
      Size = 1; // Don't create zero-sized stack objects.

    Value *Off = IRB.CreateGEP(Int8Ty, BasePointer, // BasePointer is i8*
                               ConstantInt::get(Int32Ty, -Offset));
    Value *NewArg = IRB.CreateBitCast(Off, Arg->getType(),
                                     Arg->getName() + ".unclean-byval");

    // Replace alloc with the new location.
    replaceDbgDeclare(Arg, BasePointer, BasePointer->getNextNode(), DIB,
                      DIExpression::ApplyOffset, -Offset);
    Arg->replaceAllUsesWith(NewArg);
    IRB.SetInsertPoint(cast<Instruction>(NewArg)->getNextNode());
    IRB.CreateMemCpy(Off, Align, Arg, Arg->getParamAlign(), Size);
  }

  // Allocate space for every unclean static AllocaInst on the unclean stack.
  for (AllocaInst *AI : StaticAllocas) {
    IRB.SetInsertPoint(AI);
    unsigned Offset = SSL.getObjectOffset(AI);

    replaceDbgDeclareForAlloca(AI, BasePointer, DIB, DIExpression::ApplyOffset,
                               -Offset);
    replaceDbgValueForAlloca(AI, BasePointer, DIB, -Offset);

    // Replace uses of the alloca with the new location.
    // Insert address calculation close to each use to work around PR27844.
    std::string Name = std::string(AI->getName()) + ".unclean";
    while (!AI->use_empty()) {
      Use &U = *AI->use_begin();
      Instruction *User = cast<Instruction>(U.getUser());

      Instruction *InsertBefore;
      if (auto *PHI = dyn_cast<PHINode>(User))
        InsertBefore = PHI->getIncomingBlock(U)->getTerminator();
      else
        InsertBefore = User;

      IRBuilder<> IRBUser(InsertBefore);
      Value *Off = IRBUser.CreateGEP(Int8Ty, BasePointer, // BasePointer is i8*
                                     ConstantInt::get(Int32Ty, -Offset));
      Value *Replacement = IRBUser.CreateBitCast(Off, AI->getType(), Name);

      if (auto *PHI = dyn_cast<PHINode>(User))
        // PHI nodes may have multiple incoming edges from the same BB (why??),
        // all must be updated at once with the same incoming value.
        PHI->setIncomingValueForBlock(PHI->getIncomingBlock(U), Replacement);
      else
        U.set(Replacement);
    }

    AI->eraseFromParent();
  }

  // Re-align BasePointer so that our callees would see it aligned as
  // expected.
  // FIXME: no need to update BasePointer in leaf functions.
  unsigned FrameSize = alignTo(SSL.getFrameSize(), StackAlignment);

  // Update shadow stack pointer in the function epilogue.
  IRB.SetInsertPoint(BasePointer->getNextNode());

  Value *StaticTop =
      IRB.CreateGEP(Int8Ty, BasePointer, ConstantInt::get(Int32Ty, -FrameSize),
                    "unclean_stack_static_top");
  IRB.CreateStore(StaticTop, UncleanStackPtr);
  return StaticTop;
}

void CleanStack::moveDynamicAllocasToUncleanStack(
    Function &F, Value *UncleanStackPtr, AllocaInst *DynamicTop,
    ArrayRef<AllocaInst *> DynamicAllocas) {
  DIBuilder DIB(*F.getParent());

  for (AllocaInst *AI : DynamicAllocas) {
    IRBuilder<> IRB(AI);

    // Compute the new SP value (after AI).
    Value *ArraySize = AI->getArraySize();
    if (ArraySize->getType() != IntPtrTy)
      ArraySize = IRB.CreateIntCast(ArraySize, IntPtrTy, false);

    Type *Ty = AI->getAllocatedType();
    uint64_t TySize = DL.getTypeAllocSize(Ty);
    Value *Size = IRB.CreateMul(ArraySize, ConstantInt::get(IntPtrTy, TySize));

    Value *SP = IRB.CreatePtrToInt(IRB.CreateLoad(StackPtrTy, UncleanStackPtr),
                                   IntPtrTy);
    SP = IRB.CreateSub(SP, Size);

    // Align the SP value to satisfy the AllocaInst, type and stack alignments.
    unsigned Align = std::max(
        std::max((unsigned)DL.getPrefTypeAlignment(Ty), AI->getAlignment()),
        (unsigned)StackAlignment);

    assert(isPowerOf2_32(Align));
    Value *NewTop = IRB.CreateIntToPtr(
        IRB.CreateAnd(SP, ConstantInt::get(IntPtrTy, ~uint64_t(Align - 1))),
        StackPtrTy);

    // Save the stack pointer.
    IRB.CreateStore(NewTop, UncleanStackPtr);
    if (DynamicTop)
      IRB.CreateStore(NewTop, DynamicTop);

    Value *NewAI = IRB.CreatePointerCast(NewTop, AI->getType());
    if (AI->hasName() && isa<Instruction>(NewAI))
      NewAI->takeName(AI);

    replaceDbgDeclareForAlloca(AI, NewAI, DIB, DIExpression::ApplyOffset, 0);
    AI->replaceAllUsesWith(NewAI);
    AI->eraseFromParent();
  }

  if (!DynamicAllocas.empty()) {
    // Now go through the instructions again, replacing stacksave/stackrestore.
    for (inst_iterator It = inst_begin(&F), Ie = inst_end(&F); It != Ie;) {
      Instruction *I = &*(It++);
      auto II = dyn_cast<IntrinsicInst>(I);
      if (!II)
        continue;

      if (II->getIntrinsicID() == Intrinsic::stacksave) {
        IRBuilder<> IRB(II);
        Instruction *LI = IRB.CreateLoad(StackPtrTy, UncleanStackPtr);
        LI->takeName(II);
        II->replaceAllUsesWith(LI);
        II->eraseFromParent();
      } else if (II->getIntrinsicID() == Intrinsic::stackrestore) {
        IRBuilder<> IRB(II);
        Instruction *SI = IRB.CreateStore(II->getArgOperand(0), UncleanStackPtr);
        SI->takeName(II);
        assert(II->use_empty());
        II->eraseFromParent();
      }
    }
  }
}

bool CleanStack::ShouldInlinePointerAddress(CallSite &CS) {
  Function *Callee = CS.getCalledFunction();
  if (CS.hasFnAttr(Attribute::AlwaysInline) && isInlineViable(*Callee))
    return true;
  if (Callee->isInterposable() || Callee->hasFnAttribute(Attribute::NoInline) ||
      CS.isNoInline())
    return false;
  return true;
}

void CleanStack::TryInlinePointerAddress() {
  if (!isa<CallInst>(UncleanStackPtr))
    return;

  if(F.hasOptNone())
    return;

  CallSite CS(UncleanStackPtr);
  Function *Callee = CS.getCalledFunction();
  if (!Callee || Callee->isDeclaration())
    return;

  if (!ShouldInlinePointerAddress(CS))
    return;

  InlineFunctionInfo IFI;
  InlineFunction(CS, IFI);
}

bool CleanStack::run() {
  assert(F.hasFnAttribute(Attribute::CleanStack) &&
         "Can't run CleanStack on a function without the attribute");
  assert(!F.isDeclaration() && "Can't run CleanStack on a function declaration");

  ++NumFunctions;

  SmallVector<AllocaInst *, 16> StaticAllocas;
  SmallVector<AllocaInst *, 4> DynamicAllocas;
  SmallVector<Argument *, 4> ByValArguments;
  SmallVector<ReturnInst *, 4> Returns;

  // Collect all points where stack gets unwound and needs to be restored
  // This is only necessary because the runtime (setjmp and unwind code) is
  // not aware of the unclean stack and won't unwind/restore it properly.
  // To work around this problem without changing the runtime, we insert
  // instrumentation to restore the unclean stack pointer when necessary.
  SmallVector<Instruction *, 4> StackRestorePoints;

  // Find all static and dynamic alloca instructions that must be moved to the
  // unclean stack, all return instructions and stack restore points.
  findInsts(F, StaticAllocas, DynamicAllocas, ByValArguments, Returns,
            StackRestorePoints);

  if (StaticAllocas.empty() && DynamicAllocas.empty() &&
      ByValArguments.empty() && StackRestorePoints.empty())
    return false; // Nothing to do in this function.

  if (!StaticAllocas.empty() || !DynamicAllocas.empty() ||
      !ByValArguments.empty())
    ++NumUncleanStackFunctions; // This function has the unclean stack.

  if (!StackRestorePoints.empty())
    ++NumUncleanStackRestorePointsFunctions;

  IRBuilder<> IRB(&F.front(), F.begin()->getFirstInsertionPt());
  // Calls must always have a debug location, or else inlining breaks. So
  // we explicitly set a artificial debug location here.
  if (DISubprogram *SP = F.getSubprogram())
    IRB.SetCurrentDebugLocation(DebugLoc::get(SP->getScopeLine(), 0, SP));
  if (CleanStackUsePointerAddress) {
    FunctionCallee Fn = F.getParent()->getOrInsertFunction(
        "__cleanstack_pointer_address", StackPtrTy->getPointerTo(0));
    UncleanStackPtr = IRB.CreateCall(Fn);
  } else {
    UncleanStackPtr = TL.getCleanStackPointerLocation(IRB);
  }

  // Load the current stack pointer (we'll also use it as a base pointer).
  // FIXME: use a dedicated register for it ?
  Instruction *BasePointer =
      IRB.CreateLoad(StackPtrTy, UncleanStackPtr, false, "unclean_stack_ptr");
  assert(BasePointer->getType() == StackPtrTy);

  AllocaInst *StackGuardSlot = nullptr;
  // FIXME: implement weaker forms of stack protector.
  if (true || F.hasFnAttribute(Attribute::StackProtect) ||
      F.hasFnAttribute(Attribute::StackProtectStrong) ||
      F.hasFnAttribute(Attribute::StackProtectReq)) {
    Value *StackGuard = getStackGuard(IRB, F);
    StackGuardSlot = IRB.CreateAlloca(StackPtrTy, nullptr);
    IRB.CreateStore(StackGuard, StackGuardSlot);

    for (ReturnInst *RI : Returns) {
      IRBuilder<> IRBRet(RI);
      checkStackGuard(IRBRet, F, *RI, StackGuardSlot, StackGuard);
    }
  }

  // The top of the unclean stack after all unclean static allocas are
  // allocated.
  Value *StaticTop =
      moveStaticAllocasToUncleanStack(IRB, F, StaticAllocas, ByValArguments,
                                      BasePointer, StackGuardSlot);

  // Clean stack object that stores the current unclean stack top. It is updated
  // as unclean dynamic (non-constant-sized) allocas are allocated and freed.
  // This is only needed if we need to restore stack pointer after longjmp
  // or exceptions, and we have dynamic allocations.
  // FIXME: a better alternative might be to store the unclean stack pointer
  // before setjmp / invoke instructions.
  AllocaInst *DynamicTop = createStackRestorePoints(
      IRB, F, StackRestorePoints, StaticTop, !DynamicAllocas.empty());

  // Handle dynamic allocas.
  moveDynamicAllocasToUncleanStack(F, UncleanStackPtr, DynamicTop,
                                  DynamicAllocas);

  // Restore the unclean stack pointer before each return.
  for (ReturnInst *RI : Returns) {
    IRB.SetInsertPoint(RI);
    IRB.CreateStore(BasePointer, UncleanStackPtr);
  }

  TryInlinePointerAddress();

  LLVM_DEBUG(dbgs() << "[CleanStack]     cleanstack applied\n");
  return true;
}

class CleanStackLegacyPass : public FunctionPass {
  const TargetMachine *TM = nullptr;

public:
  static char ID; // Pass identification, replacement for typeid..

  CleanStackLegacyPass() : FunctionPass(ID) {
    initializeCleanStackLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
  }

  bool runOnFunction(Function &F) override {
    LLVM_DEBUG(dbgs() << "[CleanStack] Function: " << F.getName() << "\n");

    if (!F.hasFnAttribute(Attribute::CleanStack)) {
      LLVM_DEBUG(dbgs() << "[CleanStack]     cleanstack is not requested"
                           " for this function\n");
      return false;
    }

    if (F.isDeclaration()) {
      LLVM_DEBUG(dbgs() << "[CleanStack]     function definition"
                           " is not available\n");
      return false;
    }

    TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
    auto *TL = TM->getSubtargetImpl(F)->getTargetLowering();
    if (!TL)
      report_fatal_error("TargetLowering instance is required");

    auto *DL = &F.getParent()->getDataLayout();
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    auto &ACT = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

    // Compute DT and LI only for functions that have the attribute.
    // This is only useful because the legacy pass manager doesn't let us
    // compute analyzes lazily.
    // In the backend pipeline, nothing preserves DT before CleanStack, so we
    // would otherwise always compute it wastefully, even if there is no
    // function with the cleanstack attribute.
    DominatorTree DT(F);
    LoopInfo LI(DT);

    ScalarEvolution SE(F, TLI, ACT, DT, LI);

    return CleanStack(F, *TL, *DL, SE).run();
  }
};

} // end anonymous namespace

char CleanStackLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(CleanStackLegacyPass, DEBUG_TYPE,
                      "Clean Stack instrumentation pass", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(CleanStackLegacyPass, DEBUG_TYPE,
                    "Clean Stack instrumentation pass", false, false)

FunctionPass *llvm::createCleanStackPass() { return new CleanStackLegacyPass(); }