//===- NewShrinkWrap.cpp - GCC-style shrink-wrapping ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is a port of GCC's shrink-wrapping (gcc/shrink-wrap.cc) and is an
// alternative to the ShrinkWrap pass. It has two parts.
//
// 1. Shrink-wrapping of the prologue/epilogue (GCC's try_shrink_wrapping).
//
//    The prologue is placed before a block PRO that dominates every block
//    that needs the stack frame. The blocks that are reachable from PRO, but
//    not dominated by it, can run both with and without the frame; these are
//    duplicated: the original runs with the frame, the copy without it. This
//    makes every block reachable from the prologue dominated by it, so that
//    the prologue runs at most once on every path. If the blocks cannot be
//    duplicated, PRO is moved to one of its dominators. Finally, PRO is moved
//    up as far as possible as long as no duplication is needed.
//
//    GCC places the epilogue on every exit reached with the frame. LLVM has a
//    single restore point instead, which is the nearest common post-dominator
//    of PRO and the blocks needing the frame, moved out of loops. Only the
//    blocks between PRO and the restore point need to be duplicated.
//
// 2. Separate shrink-wrapping of callee-saved registers (GCC's
//    try_shrink_wrapping_separate).
//
//    Within the region between the prologue and the epilogue, the save and
//    restore of each callee-saved register (a "component") is placed
//    individually: a component is saved in a block if that is cheaper than
//    saving it in all dominated subtrees that need it. The placement is then
//    extended to blocks where all paths from the entry, or all paths to the
//    exit, already have the component, which minimizes the number of saves
//    and restores. Saves go at the start of a block whose predecessors all
//    lack the component, restores before the terminators of a block whose
//    successors all lack it; all other placements are on split edges.
//
//    The result is recorded in MachineFrameInfo's CSR save/restore points,
//    which prolog/epilog insertion uses for the registers the target accepts
//    (TargetFrameLowering::canShrinkWrapCSRSeparately).
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/NewShrinkWrap.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "new-shrink-wrap"

STATISTIC(NumFunc, "Number of functions");
STATISTIC(NumShrinkWrapped, "Number of functions shrink-wrapped");
STATISTIC(NumDuplicated, "Number of blocks duplicated for shrink-wrapping");
STATISTIC(NumSeparateFunc,
          "Number of functions with separately shrink-wrapped registers");
STATISTIC(NumSeparateRegs,
          "Number of callee-saved registers shrink-wrapped separately");

static cl::opt<bool>
    EnableNewShrinkWrap("enable-new-shrink-wrap", cl::Hidden, cl::init(false),
                        cl::desc("Use the GCC-style NewShrinkWrap pass instead "
                                 "of the ShrinkWrap pass"));

static cl::opt<bool> EnableSeparateShrinkWrap(
    "new-shrink-wrap-separate", cl::Hidden, cl::init(true),
    cl::desc("Shrink-wrap callee-saved registers separately in the "
             "NewShrinkWrap pass"));

static cl::opt<unsigned> MaxDuplicateSize(
    "new-shrink-wrap-max-dup-size", cl::Hidden, cl::init(8),
    cl::desc("Maximum number of instructions of a block that NewShrinkWrap "
             "duplicates"));

bool llvm::isNewShrinkWrapEnabled() { return EnableNewShrinkWrap; }

namespace {

/// The placement of the prologue before a block PRO.
struct PrologueRegion {
  /// The block before which the prologue is placed.
  MachineBasicBlock *Pro = nullptr;
  /// The block at the end of which the epilogue is placed.
  MachineBasicBlock *Restore = nullptr;
  /// The blocks reachable from Pro without passing through Restore, plus
  /// Restore. These run with the frame.
  SmallSetVector<MachineBasicBlock *, 16> Blocks;
  /// The blocks of Blocks that are not dominated by Pro, which need to be
  /// duplicated for the paths that run without the frame.
  SmallSetVector<MachineBasicBlock *, 8> Dups;
  /// Pro has predecessors that run with the frame, so a new block needs to be
  /// created for the prologue.
  bool NeedsPrologueBlock = false;
};

class NewShrinkWrapImpl {
  MachineFunction *MF = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const TargetFrameLowering *TFI = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  MachineDominatorTree *MDT = nullptr;
  MachinePostDominatorTree *MPDT = nullptr;
  MachineLoopInfo *MLI = nullptr;
  MachineBlockFrequencyInfo *MBFI = nullptr;
  const MachineBranchProbabilityInfo *MBPI = nullptr;
  MachineOptimizationRemarkEmitter *ORE = nullptr;
  RegScavenger *RS = nullptr;

  /// Current opcodes for call frame setup and destroy.
  unsigned FrameSetupOpcode = ~0u;
  unsigned FrameDestroyOpcode = ~0u;

  /// Stack pointer register, used by llvm.{savestack,restorestack}.
  Register SP;

  /// Registers that need to be saved for the current function.
  SmallSetVector<MCRegister, 16> CurrentCSRs;

  /// The blocks that need the stack frame, in reverse post-order.
  SmallVector<MachineBasicBlock *, 16> FrameBlocks;

  const SmallSetVector<MCRegister, 16> &getCurrentCSRs();

  /// Check if \p MI uses or defines a callee-saved register or a frame index,
  /// and thus needs the stack frame. This is the same check as in ShrinkWrap.
  bool useOrDefCSROrFI(const MachineInstr &MI, bool StackAddressUsed);

  /// Collect FrameBlocks. Return false if the function cannot be handled.
  bool collectFrameBlocks();

  // Shrink-wrapping of the prologue/epilogue.
  MachineBasicBlock *findRestorePoint(MachineBasicBlock *Pro);
  bool canRedirect(MachineBasicBlock &MBB) const;
  bool canDuplicate(MachineBasicBlock &MBB, const PrologueRegion &Region);
  bool computeRegion(MachineBasicBlock *Pro, PrologueRegion &Region,
                     MachineBasicBlock *&MustDominate);
  bool findPrologueRegion(PrologueRegion &Region);
  MachineBasicBlock *applyPrologueRegion(const PrologueRegion &Region);
  void recomputeAnalyses();

  // Separate shrink-wrapping of callee-saved registers.
  bool shrinkWrapSeparately(MachineBasicBlock *Save,
                            MachineBasicBlock *Restore);

public:
  NewShrinkWrapImpl(const RegisterClassInfo *RCI, MachineDominatorTree *MDT,
                    MachinePostDominatorTree *MPDT, MachineLoopInfo *MLI,
                    MachineBlockFrequencyInfo *MBFI,
                    const MachineBranchProbabilityInfo *MBPI,
                    MachineOptimizationRemarkEmitter *ORE)
      : RCI(RCI), MDT(MDT), MPDT(MPDT), MLI(MLI), MBFI(MBFI), MBPI(MBPI),
        ORE(ORE) {}

  static bool isShrinkWrapEnabled(const MachineFunction &MF);

  bool run(MachineFunction &MF);
};

class NewShrinkWrapLegacy : public MachineFunctionPass {
public:
  static char ID;

  NewShrinkWrapLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
    AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachinePostDominatorTreeWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineOptimizationRemarkEmitterPass>();
    AU.addRequired<MachineRegisterClassInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }

  StringRef getPassName() const override { return "New Shrink Wrapping"; }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char NewShrinkWrapLegacy::ID = 0;

char &llvm::NewShrinkWrapID = NewShrinkWrapLegacy::ID;

INITIALIZE_PASS_BEGIN(NewShrinkWrapLegacy, DEBUG_TYPE, "New Shrink Wrap Pass",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_DEPENDENCY(MachineRegisterClassInfoWrapperPass)
INITIALIZE_PASS_END(NewShrinkWrapLegacy, DEBUG_TYPE, "New Shrink Wrap Pass",
                    false, false)

const SmallSetVector<MCRegister, 16> &NewShrinkWrapImpl::getCurrentCSRs() {
  if (CurrentCSRs.empty()) {
    BitVector SavedRegs;
    TFI->determineCalleeSaves(*MF, SavedRegs, RS);
    for (unsigned Reg : SavedRegs.set_bits())
      CurrentCSRs.insert(MCRegister(Reg));
  }
  return CurrentCSRs;
}

bool NewShrinkWrapImpl::useOrDefCSROrFI(const MachineInstr &MI,
                                        bool StackAddressUsed) {
  // Check if Op is known to access an address not on the function's stack.
  auto IsKnownNonStackPtr = [](MachineMemOperand *Op) {
    if (Op->getValue()) {
      const Value *UO = getUnderlyingObject(Op->getValue());
      if (!UO)
        return false;
      if (auto *Arg = dyn_cast<Argument>(UO))
        return !Arg->hasPassPointeeByValueCopyAttr();
      return isa<GlobalValue>(UO);
    }
    if (const PseudoSourceValue *PSV = Op->getPseudoValue())
      return PSV->isJumpTable() || PSV->isConstantPool();
    return false;
  };
  // Load/store operations may access the stack indirectly when we previously
  // computed an address to a stack location.
  if (StackAddressUsed && MI.mayLoadOrStore() &&
      (MI.isCall() || MI.hasUnmodeledSideEffects() || MI.memoperands_empty() ||
       !all_of(MI.memoperands(), IsKnownNonStackPtr)))
    return true;

  if (MI.getOpcode() == FrameSetupOpcode ||
      MI.getOpcode() == FrameDestroyOpcode)
    return true;

  for (const MachineOperand &MO : MI.operands()) {
    bool UseOrDefCSR = false;
    if (MO.isReg()) {
      // Ignore instructions like DBG_VALUE which don't read/def the register.
      if (!MO.isDef() && !MO.readsReg())
        continue;
      Register PhysReg = MO.getReg();
      if (!PhysReg)
        continue;
      assert(PhysReg.isPhysical() && "Unallocated register?!");
      // See ShrinkWrap for the stack pointer and non-allocatable callee-saved
      // registers.
      UseOrDefCSR = (!MI.isCall() && PhysReg == SP) ||
                    RCI->getLastCalleeSavedAlias(PhysReg) ||
                    (!MI.isReturn() &&
                     TRI->isNonallocatableRegisterCalleeSave(PhysReg)) ||
                    TRI->isVirtualFrameRegister(PhysReg);
    } else if (MO.isRegMask()) {
      // Check if this regmask clobbers any of the CSRs.
      for (MCRegister Reg : getCurrentCSRs()) {
        if (MO.clobbersPhysReg(Reg)) {
          UseOrDefCSR = true;
          break;
        }
      }
    }
    // Skip FrameIndex operands in DBG_VALUE instructions.
    if (UseOrDefCSR || (MO.isFI() && !MI.isDebugValue()))
      return true;
  }
  return false;
}

bool NewShrinkWrapImpl::collectFrameBlocks() {
  FrameBlocks.clear();

  // Is true for the blocks where stack accesses or computations of
  // stack-relative addresses are possible on some path including the block.
  // Like ShrinkWrap, rely on the reverse post-order to visit predecessors
  // first, except for loops where the result is conservative.
  BitVector StackAddressUsedBlockInfo(MF->getNumBlockIDs(), true);

  // A stack address can only be computed by an instruction with a frame index
  // operand other than a spill or reload. If there is none, loads and stores
  // cannot access the stack indirectly.
  bool MayComputeStackAddress = any_of(*MF, [&](const MachineBasicBlock &MBB) {
    return any_of(MBB, [&](const MachineInstr &MI) {
      int FI;
      return !MI.isDebugInstr() &&
             any_of(MI.operands(),
                    [](const MachineOperand &MO) { return MO.isFI(); }) &&
             !TII->isLoadFromStackSlot(MI, FI) &&
             !TII->isStoreToStackSlot(MI, FI);
    });
  });
  if (!MayComputeStackAddress)
    StackAddressUsedBlockInfo.reset();

  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(&MF->front());
  for (MachineBasicBlock *MBB : RPOT) {
    if (MBB->isEHFuncletEntry())
      return false;

    // Keep the landing pads and inlineasm_br targets in the region, as a
    // block can jump to them from its middle.
    if (MBB->isEHPad() || MBB->isInlineAsmBrIndirectTarget()) {
      FrameBlocks.push_back(MBB);
      continue;
    }

    bool StackAddressUsed = any_of(MBB->predecessors(), [&](auto *Pred) {
      return StackAddressUsedBlockInfo.test(Pred->getNumber());
    });
    for (const MachineInstr &MI : *MBB) {
      if (useOrDefCSROrFI(MI, StackAddressUsed)) {
        LLVM_DEBUG(dbgs() << printMBBReference(*MBB)
                          << " needs the frame due to " << MI);
        FrameBlocks.push_back(MBB);
        StackAddressUsed = MayComputeStackAddress;
        break;
      }
    }
    StackAddressUsedBlockInfo[MBB->getNumber()] = StackAddressUsed;
  }
  return true;
}

static MachineBasicBlock *getImmediateDominator(MachineDominatorTree &DT,
                                                MachineBasicBlock *MBB) {
  MachineDomTreeNode *IDom = DT.getNode(MBB)->getIDom();
  return IDom ? IDom->getBlock() : nullptr;
}

static MachineBasicBlock *
getImmediatePostDominator(MachinePostDominatorTree &PDT,
                          MachineBasicBlock *MBB) {
  MachineDomTreeNode *IDom = PDT.getNode(MBB)->getIDom();
  return IDom ? IDom->getBlock() : nullptr;
}

/// Find the block at the end of which the epilogue goes, for the prologue
/// placed before \p Pro: the nearest common post-dominator of Pro and all
/// blocks needing the frame, which is not in a loop (the prologue is not
/// either), and whose terminators do not need the frame.
MachineBasicBlock *NewShrinkWrapImpl::findRestorePoint(MachineBasicBlock *Pro) {
  MachineBasicBlock *Restore = Pro;
  for (MachineBasicBlock *MBB : FrameBlocks) {
    Restore = MPDT->findNearestCommonDominator(Restore, MBB);
    if (!Restore)
      return nullptr;
  }

  while (Restore) {
    bool TerminatorNeedsFrame =
        any_of(Restore->terminators(), [&](const MachineInstr &Term) {
          return useOrDefCSROrFI(Term, /*StackAddressUsed=*/true);
        });
    if (!TerminatorNeedsFrame && !MLI->getLoopFor(Restore) &&
        TFI->canUseAsEpilogue(*Restore))
      return Restore;
    Restore = getImmediatePostDominator(*MPDT, Restore);
  }
  return nullptr;
}

/// Return true if the successors of \p MBB can be changed.
bool NewShrinkWrapImpl::canRedirect(MachineBasicBlock &MBB) const {
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  SmallVector<MachineOperand, 4> Cond;
  if (TII->analyzeBranch(MBB, TBB, FBB, Cond))
    return false;
  // A conditional branch with both targets the same block has two identical
  // CFG edges, which cannot be told apart.
  return !TBB || TBB != FBB;
}

/// Return true if \p MBB can be duplicated for the paths that run without
/// the frame (GCC's can_dup_for_shrink_wrapping).
bool NewShrinkWrapImpl::canDuplicate(MachineBasicBlock &MBB,
                                     const PrologueRegion &Region) {
  if (&MBB == &MF->front() || MBB.isEHPad() || MBB.hasAddressTaken() ||
      MBB.isInlineAsmBrIndirectTarget() || MBB.isEHFuncletEntry())
    return false;

  if (!MBB.succ_empty() && !canRedirect(MBB))
    return false;

  unsigned Size = 0;
  for (const MachineInstr &MI : MBB) {
    if (MI.isNotDuplicable() || MI.isLabel() || MI.getPreInstrSymbol() ||
        MI.getPostInstrSymbol())
      return false;
    if (!MI.isMetaInstruction() && ++Size > MaxDuplicateSize)
      return false;
  }

  // The predecessors that run without the frame branch to the copy.
  for (MachineBasicBlock *Pred : MBB.predecessors())
    if (!Region.Blocks.count(Pred) && !canRedirect(*Pred))
      return false;
  return true;
}

/// Compute the region running with the frame for the prologue placed before
/// \p Pro. Return false if this placement does not work. If the reason is a
/// block that cannot be duplicated, \p MustDominate is set to it.
bool NewShrinkWrapImpl::computeRegion(MachineBasicBlock *Pro,
                                      PrologueRegion &Region,
                                      MachineBasicBlock *&MustDominate) {
  MustDominate = nullptr;
  Region = PrologueRegion();
  Region.Pro = Pro;

  // GCC's can_get_prologue.
  if (Pro->isEHPad() || Pro->isInlineAsmBrIndirectTarget() ||
      Pro->hasAddressTaken() || !TFI->canUseAsPrologue(*Pro))
    return false;

  Region.Restore = findRestorePoint(Pro);
  if (!Region.Restore)
    return false;

  SmallVector<MachineBasicBlock *, 16> WorkList({Pro});
  Region.Blocks.insert(Pro);
  while (!WorkList.empty()) {
    MachineBasicBlock *MBB = WorkList.pop_back_val();
    if (MBB == Region.Restore)
      continue;
    for (MachineBasicBlock *Succ : MBB->successors())
      if (Region.Blocks.insert(Succ))
        WorkList.push_back(Succ);
  }
  // Restore post-dominates Pro, so this always holds.
  if (!Region.Blocks.count(Region.Restore))
    return false;

  for (MachineBasicBlock *MBB : Region.Blocks) {
    if (MDT->dominates(Pro, MBB))
      continue;
    if (!canDuplicate(*MBB, Region)) {
      LLVM_DEBUG(dbgs() << "Cannot duplicate " << printMBBReference(*MBB)
                        << '\n');
      MustDominate = MBB;
      return false;
    }
    Region.Dups.insert(MBB);
  }

  // The predecessors running with the frame are dominated by Pro, or the
  // originals of duplicated blocks. If there are any, the prologue goes in a
  // new block that the predecessors running without the frame branch to.
  for (MachineBasicBlock *Pred : Pro->predecessors())
    if (MDT->dominates(Pro, Pred) || Region.Dups.count(Pred))
      Region.NeedsPrologueBlock = true;
  if (Region.NeedsPrologueBlock)
    for (MachineBasicBlock *Pred : Pro->predecessors())
      if (!Region.Blocks.count(Pred) && !canRedirect(*Pred))
        return false;
  return true;
}

/// Find where to place the prologue (GCC's try_shrink_wrapping). Return false
/// if it stays in the entry block.
bool NewShrinkWrapImpl::findPrologueRegion(PrologueRegion &Region) {
  MachineBasicBlock *Entry = &MF->front();
  if (FrameBlocks.empty())
    return false;

  // The tightest placement dominates all blocks needing the frame.
  MachineBasicBlock *Pro = FrameBlocks.front();
  for (MachineBasicBlock *MBB : FrameBlocks)
    Pro = MDT->findNearestCommonDominator(Pro, MBB);
  LLVM_DEBUG(dbgs() << "After wrapping required blocks, PRO is "
                    << printMBBReference(*Pro) << '\n');

  // Move PRO up until the blocks to duplicate can be duplicated, and the
  // prologue and epilogue can be placed.
  while (Pro != Entry) {
    MachineBasicBlock *MustDominate;
    if (computeRegion(Pro, Region, MustDominate))
      break;
    Pro = MustDominate ? MDT->findNearestCommonDominator(Pro, MustDominate)
                       : getImmediateDominator(*MDT, Pro);
  }
  LLVM_DEBUG(dbgs() << "Avoiding non-duplicatable blocks, PRO is "
                    << printMBBReference(*Pro) << '\n');
  if (Pro == Entry)
    return false;

  // Move PRO up while it post-dominates its dominators, as long as no block
  // needs to be duplicated. Earlier is better for scheduling.
  MachineBasicBlock *Pre = getImmediateDominator(*MDT, Pro);
  while (Pre && MPDT->dominates(Pro, Pre)) {
    PrologueRegion PreRegion;
    MachineBasicBlock *MustDominate;
    if (computeRegion(Pre, PreRegion, MustDominate) && PreRegion.Dups.empty())
      Region = std::move(PreRegion);
    else if (Pre == Entry)
      break;
    if (Pre == Entry) {
      // The prologue may as well go in the entry block.
      LLVM_DEBUG(dbgs() << "PRO moved back to the entry block\n");
      return false;
    }
    Pre = getImmediateDominator(*MDT, Pre);
  }
  LLVM_DEBUG(dbgs() << "Bumping back to anticipatable blocks, PRO is "
                    << printMBBReference(*Region.Pro) << ", restore point is "
                    << printMBBReference(*Region.Restore) << '\n');
  return true;
}

/// Duplicate the blocks of \p Region that run both with and without the frame,
/// and create the prologue block if needed. Return the block to place the
/// prologue in.
MachineBasicBlock *
NewShrinkWrapImpl::applyPrologueRegion(const PrologueRegion &Region) {
  MachineBasicBlock *Pro = Region.Pro;

  // The blocks running without the frame, before any change.
  SmallVector<MachineBasicBlock *, 16> NoFrameBlocks;
  // The block each block falls through to, if any, before any change.
  DenseMap<MachineBasicBlock *, MachineBasicBlock *> FallThrough;
  for (MachineBasicBlock &MBB : *MF) {
    if (!Region.Blocks.count(&MBB))
      NoFrameBlocks.push_back(&MBB);
    FallThrough[&MBB] = MBB.getLogicalFallThrough();
  }

  // Copy the blocks; the copies are added at the end of the function, and
  // run without the frame.
  DenseMap<MachineBasicBlock *, MachineBasicBlock *> Copies;
  SmallPtrSet<MachineBasicBlock *, 8> IsCopy;
  for (MachineBasicBlock *MBB : Region.Dups) {
    MachineBasicBlock *Copy = MF->CreateMachineBasicBlock(MBB->getBasicBlock());
    MF->push_back(Copy);
    for (MachineInstr &MI : *MBB)
      TII->duplicate(*Copy, Copy->end(), MI);
    for (const auto &LI : MBB->liveins())
      Copy->addLiveIn(LI);
    for (auto SI = MBB->succ_begin(), SE = MBB->succ_end(); SI != SE; ++SI)
      Copy->copySuccessor(MBB, SI);
    Copy->setCallFrameSize(MBB->getCallFrameSize());
    Copies[MBB] = Copy;
    IsCopy.insert(Copy);
    FallThrough[Copy] = FallThrough[MBB];
    NoFrameBlocks.push_back(Copy);
    ++NumDuplicated;
    LLVM_DEBUG(dbgs() << "Duplicated " << printMBBReference(*MBB) << " to "
                      << printMBBReference(*Copy) << '\n');
  }

  MachineBasicBlock *PrologueBlock = nullptr;
  if (Region.NeedsPrologueBlock) {
    PrologueBlock = MF->CreateMachineBasicBlock(Pro->getBasicBlock());
    MF->push_back(PrologueBlock);
    PrologueBlock->addSuccessor(Pro, BranchProbability::getOne());
    for (const auto &LI : Pro->liveins())
      PrologueBlock->addLiveIn(LI);
    PrologueBlock->setCallFrameSize(Pro->getCallFrameSize());
    PrologueBlock->updateTerminator(Pro);
    LLVM_DEBUG(dbgs() << "Made prologue block "
                      << printMBBReference(*PrologueBlock) << '\n');
  }

  // Redirect the edges from the blocks running without the frame: to the
  // copies, and to the prologue block instead of Pro.
  auto MapSucc = [&](MachineBasicBlock *Succ) {
    if (MachineBasicBlock *Copy = Copies.lookup(Succ))
      return Copy;
    if (Succ == Pro && PrologueBlock)
      return PrologueBlock;
    return Succ;
  };
  for (MachineBasicBlock *MBB : NoFrameBlocks) {
    bool Changed = false;
    SmallSetVector<MachineBasicBlock *, 4> Succs(MBB->succ_begin(),
                                                 MBB->succ_end());
    for (MachineBasicBlock *Succ : Succs) {
      MachineBasicBlock *NewSucc = MapSucc(Succ);
      if (NewSucc != Succ) {
        LLVM_DEBUG(dbgs() << "Redirecting edge " << printMBBReference(*MBB)
                          << " -> " << printMBBReference(*Succ) << " to "
                          << printMBBReference(*NewSucc) << '\n');
        MBB->ReplaceUsesOfBlockWith(Succ, NewSucc);
        Changed = true;
      }
    }
    if ((Changed || IsCopy.count(MBB)) && !MBB->succ_empty()) {
      MachineBasicBlock *FT = FallThrough.lookup(MBB);
      MBB->updateTerminator(FT ? MapSucc(FT) : nullptr);
    }
  }

  return PrologueBlock ? PrologueBlock : Pro;
}

void NewShrinkWrapImpl::recomputeAnalyses() {
  MDT->recalculate(*MF);
  MPDT->recalculate(*MF);
  MLI->calculate(*MDT);
  MachineCycleInfo MCI;
  MCI.compute(*MF);
  MBFI->calculate(*MF, *MBPI, MCI);
}

/// Separately shrink-wrap the callee-saved registers in the region between
/// \p Save and \p Restore (GCC's try_shrink_wrapping_separate). If \p Restore
/// is null, the region extends to the return blocks. Return true if the
/// function changed.
bool NewShrinkWrapImpl::shrinkWrapSeparately(MachineBasicBlock *Save,
                                             MachineBasicBlock *Restore) {
  if (!EnableSeparateShrinkWrap || !TFI->enableSeparateCSRShrinkWrapping(*MF))
    return false;

  // GCC does not handle these "strange" functions either.
  const MachineFrameInfo &MFI = MF->getFrameInfo();
  if (MFI.hasVarSizedObjects() || MF->exposesReturnsTwice() ||
      MF->callsEHReturn() || MF->callsUnwindInit() || MF->hasEHFunclets() ||
      any_of(*MF, [](const MachineBasicBlock &MBB) {
        return MBB.isEHPad() || MBB.isInlineAsmBrIndirectTarget();
      }))
    return false;

  // The components are the callee-saved registers that need to be saved, and
  // that the target can handle separately.
  SmallVector<MCRegister, 16> Components;
  for (const MCPhysReg *CSR = MF->getRegInfo().getCalleeSavedRegs(); *CSR;
       ++CSR)
    if (getCurrentCSRs().count(*CSR) &&
        TFI->canShrinkWrapCSRSeparately(*MF, *CSR))
      Components.push_back(*CSR);
  if (Components.empty())
    return false;
  unsigned NumComponents = Components.size();

  // The region between the prologue and the epilogue.
  unsigned NumBlocks = MF->getNumBlockIDs();
  BitVector InRegion(NumBlocks);
  SmallVector<MachineBasicBlock *, 16> RegionBlocks;
  for (MachineBasicBlock &MBB : *MF)
    if (MDT->isReachableFromEntry(&MBB) && MDT->dominates(Save, &MBB) &&
        (!Restore || MPDT->dominates(Restore, &MBB))) {
      InRegion.set(MBB.getNumber());
      RegionBlocks.push_back(&MBB);
    }
  auto IsInRegion = [&](const MachineBasicBlock *MBB) {
    return InRegion.test(MBB->getNumber());
  };
  // Whether the edges out of MBB leave the region, i.e. go to the exit.
  auto IsRegionExit = [&](const MachineBasicBlock *MBB) {
    return Restore ? MBB == Restore : MBB->isReturnBlock();
  };

  // Only the prologue enters the region, and only the epilogue leaves it.
  for (MachineBasicBlock *MBB : RegionBlocks) {
    if (MBB != Save && !all_of(MBB->predecessors(), IsInRegion))
      return false;
    if (MBB == Save && any_of(MBB->predecessors(), IsInRegion))
      return false;
    if (!IsRegionExit(MBB) && !all_of(MBB->successors(), IsInRegion))
      return false;
  }

  auto UsesComponent = [&](const MachineInstr &MI, MCRegister Reg) {
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.getReg() && (MO.isDef() || MO.readsReg()) &&
          TRI->regsOverlap(MO.getReg(), Reg))
        return true;
      if (MO.isRegMask() && MO.clobbersPhysReg(Reg))
        return true;
    }
    return false;
  };

  // The components each block needs: those live-in, used or defined in it
  // (GCC's components_for_bb).
  std::vector<BitVector> Needs(NumBlocks, BitVector(NumComponents));
  for (MachineBasicBlock *MBB : RegionBlocks) {
    BitVector &N = Needs[MBB->getNumber()];
    for (unsigned C = 0; C != NumComponents; ++C) {
      MCRegister Reg = Components[C];
      if (any_of(MBB->liveins(),
                 [&](const MachineBasicBlock::RegisterMaskPair &LI) {
                   return TRI->regsOverlap(LI.PhysReg, Reg);
                 }) ||
          any_of(*MBB, [&](const MachineInstr &MI) {
            return !MI.isDebugInstr() && UsesComponent(MI, Reg);
          }))
        N.set(C);
    }
    // Blocks without successors that are not returns need all components, to
    // keep the unwind information the same on all paths to such blocks.
    if (MBB->succ_empty() && !MBB->isReturnBlock())
      N.set();
  }

  // The cost of placing a save at the start of a block, which is its
  // frequency without the frequency from back edges.
  DenseMap<const MachineBasicBlock *, uint64_t> OwnCost;
  for (MachineBasicBlock *MBB : RegionBlocks) {
    uint64_t Cost = MBFI->getBlockFreq(MBB).getFrequency();
    for (MachineBasicBlock *Pred : MBB->predecessors()) {
      if (!MDT->dominates(MBB, Pred))
        continue;
      uint64_t EdgeFreq = (MBFI->getBlockFreq(Pred) *
                           MBPI->getEdgeProbability(Pred, MBB))
                              .getFrequency();
      Cost = Cost > EdgeFreq ? Cost - EdgeFreq : 0;
    }
    OwnCost[MBB] = Cost;
  }

  // Place the saves of each component (GCC's place_prologue_for_one_component):
  // walk the dominator tree from Save, and mark a block as having the
  // component if it needs it, or if this is cheaper than placing the saves in
  // all its dominator subtrees.
  std::vector<BitVector> Has(NumBlocks, BitVector(NumComponents));
  struct Frame {
    MachineDomTreeNode *Node;
    MachineDomTreeNode::iterator NextChild;
    uint64_t TotalCost;
  };
  for (unsigned C = 0; C != NumComponents; ++C) {
    MachineDomTreeNode *SaveNode = MDT->getNode(Save);
    SmallVector<Frame, 16> Stack({{SaveNode, SaveNode->begin(), 0}});
    while (true) {
      Frame &F = Stack.back();
      MachineBasicBlock *MBB = F.Node->getBlock();
      uint64_t Own = OwnCost.lookup(MBB);
      bool NeedsC = Needs[MBB->getNumber()].test(C);
      // Visit the children unless the block needs the component itself, or
      // they already cost more than this block.
      if (!NeedsC && F.NextChild != F.Node->end() && F.TotalCost <= Own) {
        MachineDomTreeNode *Child = *F.NextChild++;
        if (IsInRegion(Child->getBlock()))
          Stack.push_back({Child, Child->begin(), 0});
        continue;
      }

      bool HasC = NeedsC || F.TotalCost > Own;
      if (!HasC) {
        // If this block's immediate post-dominator is dominated by it, and
        // has the component, place it here as well: earlier is better.
        MachineBasicBlock *Kid = getImmediatePostDominator(*MPDT, MBB);
        HasC = Kid && IsInRegion(Kid) && MDT->dominates(MBB, Kid) &&
               Has[Kid->getNumber()].test(C);
      }
      if (HasC) {
        F.TotalCost = Own;
        Has[MBB->getNumber()].set(C);
      }

      uint64_t Total = F.TotalCost;
      Stack.pop_back();
      if (Stack.empty())
        break;
      uint64_t &ParentTotal = Stack.back().TotalCost;
      ParentTotal = ParentTotal + Total < ParentTotal ? UINT64_MAX
                                                      : ParentTotal + Total;
    }
  }

  // Extend the components to every block where they are already present on
  // all paths from the region entry, or on all paths to the region exit
  // (GCC's spread_components). This does not add saves or restores on any
  // path, but may remove some. Repeat until nothing changes.
  BitVector AllComponents(NumComponents, true);
  std::vector<BitVector> Head(NumBlocks, BitVector(NumComponents));
  std::vector<BitVector> Tail(NumBlocks, BitVector(NumComponents));
  bool Changed;
  do {
    // Head: the components missing on some path from the region entry.
    for (MachineBasicBlock *MBB : RegionBlocks)
      Head[MBB->getNumber()].reset();
    SmallSetVector<MachineBasicBlock *, 16> WorkList;
    WorkList.insert(Save);
    while (!WorkList.empty()) {
      MachineBasicBlock *MBB = WorkList.pop_back_val();
      BitVector &H = Head[MBB->getNumber()];
      BitVector Old = H;
      if (MBB == Save)
        H |= AllComponents;
      for (MachineBasicBlock *Pred : MBB->predecessors())
        if (IsInRegion(Pred))
          H |= Head[Pred->getNumber()];
      H.reset(Has[MBB->getNumber()]);
      if (H != Old || MBB == Save)
        for (MachineBasicBlock *Succ : MBB->successors())
          if (IsInRegion(Succ))
            WorkList.insert(Succ);
    }

    // Tail: the components missing on some path to the region exit. Blocks
    // that cannot reach the exit only consider the paths from the entry.
    for (MachineBasicBlock *MBB : RegionBlocks)
      Tail[MBB->getNumber()] = AllComponents;
    SmallVector<MachineBasicBlock *, 16> Exits;
    for (MachineBasicBlock *MBB : RegionBlocks)
      if (IsRegionExit(MBB))
        Exits.push_back(MBB);
    {
      SmallPtrSet<MachineBasicBlock *, 16> ReachesExit;
      SmallVector<MachineBasicBlock *, 16> Stack(Exits);
      while (!Stack.empty()) {
        MachineBasicBlock *MBB = Stack.pop_back_val();
        if (!ReachesExit.insert(MBB).second)
          continue;
        Tail[MBB->getNumber()].reset();
        for (MachineBasicBlock *Pred : MBB->predecessors())
          if (IsInRegion(Pred))
            Stack.push_back(Pred);
      }
    }
    for (MachineBasicBlock *MBB : Exits)
      WorkList.insert(MBB);
    while (!WorkList.empty()) {
      MachineBasicBlock *MBB = WorkList.pop_back_val();
      BitVector &T = Tail[MBB->getNumber()];
      BitVector Old = T;
      if (IsRegionExit(MBB))
        T |= AllComponents;
      else
        for (MachineBasicBlock *Succ : MBB->successors())
          T |= Tail[Succ->getNumber()];
      T.reset(Has[MBB->getNumber()]);
      if (T != Old || IsRegionExit(MBB))
        for (MachineBasicBlock *Pred : MBB->predecessors())
          if (IsInRegion(Pred))
            WorkList.insert(Pred);
    }

    // A block has a component unless it is missing both on some path from
    // the entry and on some path to the exit.
    Changed = false;
    for (MachineBasicBlock *MBB : RegionBlocks) {
      BitVector NewHas = Head[MBB->getNumber()];
      NewHas &= Tail[MBB->getNumber()];
      NewHas.flip();
      if (NewHas != Has[MBB->getNumber()]) {
        Has[MBB->getNumber()] = NewHas;
        Changed = true;
      }
    }
  } while (Changed);

  LLVM_DEBUG({
    for (MachineBasicBlock *MBB : RegionBlocks) {
      dbgs() << printMBBReference(*MBB) << " has";
      for (unsigned C : Has[MBB->getNumber()].set_bits())
        dbgs() << ' ' << printReg(Components[C], TRI);
      dbgs() << '\n';
    }
  });

  // Don't wrap the components present in the prologue block separately; the
  // prologue saves them (GCC does the same).
  BitVector Active = AllComponents;
  Active.reset(Has[Save->getNumber()]);

  // Saves go at the start of a block if all its predecessors lack the
  // component, restores before the terminators of a block if all its
  // successors lack it and the terminators do not use it. All others go on
  // split edges.
  auto TerminatorsUse = [&](MachineBasicBlock *MBB, unsigned C) {
    return any_of(MBB->terminators(), [&](const MachineInstr &MI) {
      return UsesComponent(MI, Components[C]);
    });
  };
  auto ComputeHeadTail = [&](std::vector<BitVector> &ProHead,
                             std::vector<BitVector> &EpiTail) {
    for (MachineBasicBlock *MBB : RegionBlocks) {
      const BitVector &H = Has[MBB->getNumber()];
      BitVector &Pro = ProHead[MBB->getNumber()];
      Pro = H;
      Pro &= Active;
      if (MBB == Save)
        Pro.reset();
      for (MachineBasicBlock *Pred : MBB->predecessors())
        Pro.reset(Has[Pred->getNumber()]);

      BitVector &Epi = EpiTail[MBB->getNumber()];
      Epi = H;
      Epi &= Active;
      if (MBB->succ_empty() && !IsRegionExit(MBB))
        Epi.reset();
      if (!IsRegionExit(MBB))
        for (MachineBasicBlock *Succ : MBB->successors())
          Epi.reset(Has[Succ->getNumber()]);
      for (unsigned C : Epi.set_bits())
        if (TerminatorsUse(MBB, C))
          Epi.reset(C);
    }
  };

  // Disqualify the components needing code on edges that cannot be split
  // (GCC's disqualify_problematic_components).
  std::vector<BitVector> ProHead(NumBlocks, BitVector(NumComponents));
  std::vector<BitVector> EpiTail(NumBlocks, BitVector(NumComponents));
  ComputeHeadTail(ProHead, EpiTail);
  for (MachineBasicBlock *MBB : RegionBlocks) {
    const BitVector &H = Has[MBB->getNumber()];
    if (IsRegionExit(MBB)) {
      // The restores must go before the epilogue.
      BitVector Epi = H;
      Epi &= Active;
      Epi.reset(EpiTail[MBB->getNumber()]);
      Active.reset(Epi);
      continue;
    }
    for (MachineBasicBlock *Succ : MBB->successors()) {
      const BitVector &SH = Has[Succ->getNumber()];
      BitVector Pro = SH, Epi = H;
      Pro.reset(H);
      Pro.reset(ProHead[Succ->getNumber()]);
      Epi.reset(SH);
      Epi.reset(EpiTail[MBB->getNumber()]);
      Pro |= Epi;
      Pro &= Active;
      if (Pro.any() && !MBB->canSplitCriticalEdge(Succ)) {
        LLVM_DEBUG(dbgs() << "Cannot split edge " << printMBBReference(*MBB)
                          << " -> " << printMBBReference(*Succ) << '\n');
        Active.reset(Pro);
      }
    }
  }
  if (Active.none())
    return false;
  ComputeHeadTail(ProHead, EpiTail);

  SaveRestorePoints CSRSaves, CSRRestores;
  auto AddPoints = [&](SaveRestorePoints &Points, MachineBasicBlock *MBB,
                       const BitVector &Comps) {
    for (unsigned C : Comps.set_bits())
      Points[MBB].push_back(CalleeSavedInfo(Components[C]));
  };
  struct EdgeSplit {
    MachineBasicBlock *From, *To;
    BitVector Pro, Epi;
  };
  SmallVector<EdgeSplit, 4> EdgeSplits;
  for (MachineBasicBlock *MBB : RegionBlocks) {
    AddPoints(CSRSaves, MBB, ProHead[MBB->getNumber()]);
    AddPoints(CSRRestores, MBB, EpiTail[MBB->getNumber()]);
    if (IsRegionExit(MBB))
      continue;
    const BitVector &H = Has[MBB->getNumber()];
    for (MachineBasicBlock *Succ : MBB->successors()) {
      const BitVector &SH = Has[Succ->getNumber()];
      BitVector Pro = SH, Epi = H;
      Pro.reset(H);
      Pro.reset(ProHead[Succ->getNumber()]);
      Pro &= Active;
      Epi.reset(SH);
      Epi.reset(EpiTail[MBB->getNumber()]);
      Epi &= Active;
      if (Pro.any() || Epi.any())
        EdgeSplits.push_back({MBB, Succ, Pro, Epi});
    }
  }
  for (EdgeSplit &E : EdgeSplits) {
    MachineBasicBlock *NMBB = E.From->SplitCriticalEdge(
        E.To, MachineBasicBlock::SplitCriticalEdgeAnalyses{nullptr, nullptr,
                                                           nullptr, nullptr});
    assert(NMBB && "Checked that the edge can be split");
    LLVM_DEBUG(dbgs() << "Split edge " << printMBBReference(*E.From) << " -> "
                      << printMBBReference(*E.To) << " with "
                      << printMBBReference(*NMBB) << '\n');
    AddPoints(CSRSaves, NMBB, E.Pro);
    AddPoints(CSRRestores, NMBB, E.Epi);
  }

  LLVM_DEBUG({
    dbgs() << "Separately shrink-wrapped:";
    for (unsigned C : Active.set_bits())
      dbgs() << ' ' << printReg(Components[C], TRI);
    dbgs() << '\n';
  });
  MachineFrameInfo &MutableMFI = MF->getFrameInfo();
  MutableMFI.setCSRSavePoints(std::move(CSRSaves));
  MutableMFI.setCSRRestorePoints(std::move(CSRRestores));
  ++NumSeparateFunc;
  NumSeparateRegs += Active.count();
  return !EdgeSplits.empty();
}

bool NewShrinkWrapImpl::run(MachineFunction &Fn) {
  LLVM_DEBUG(dbgs() << "**** Analysing " << Fn.getName() << '\n');
  MF = &Fn;
  const TargetSubtargetInfo &ST = MF->getSubtarget();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  TFI = ST.getFrameLowering();
  FrameSetupOpcode = TII->getCallFrameSetupOpcode();
  FrameDestroyOpcode = TII->getCallFrameDestroyOpcode();
  SP = ST.getTargetLowering()->getStackPointerRegisterToSaveRestore();
  CurrentCSRs.clear();
  ++NumFunc;

  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(&MF->front());
  if (containsIrreducibleCFG<MachineBasicBlock *>(RPOT, *MLI)) {
    ORE->emit([&]() {
      return MachineOptimizationRemarkMissed(DEBUG_TYPE,
                                             "UnsupportedIrreducibleCFG",
                                             MF->getFunction().getSubprogram(),
                                             &MF->front())
             << "Irreducible CFGs are not supported yet.";
    });
    return false;
  }

  std::unique_ptr<RegScavenger> OwnedRS(
      TRI->requiresRegisterScavenging(*MF) ? new RegScavenger() : nullptr);
  RS = OwnedRS.get();

  if (!collectFrameBlocks())
    return false;

  bool Changed = false;
  MachineBasicBlock *Save = &MF->front();
  MachineBasicBlock *Restore = nullptr;
  PrologueRegion Region;
  if (findPrologueRegion(Region)) {
    Restore = Region.Restore;
    Save = applyPrologueRegion(Region);
    if (!Region.Dups.empty() || Region.NeedsPrologueBlock) {
      Changed = true;
      recomputeAnalyses();
    }
  } else if (!FrameBlocks.empty() && Save->pred_empty()) {
    // The prologue stays in the entry block. GCC places the epilogue on all
    // exits, but the epilogue may still be placed earlier, before the code
    // after the last use of the frame.
    Restore = findRestorePoint(Save);
    if (Restore && Restore->isReturnBlock())
      Restore = nullptr;
  }

  if (Restore) {
    assert(MDT->dominates(Save, Restore) && MPDT->dominates(Restore, Save) &&
           "Invalid save/restore points");
    LLVM_DEBUG(dbgs() << "Shrink-wrapped: save point "
                      << printMBBReference(*Save) << ", restore point "
                      << printMBBReference(*Restore) << '\n');
    MachineFrameInfo &MFI = MF->getFrameInfo();
    MFI.setSavePoints(SaveRestorePoints({{Save, {}}}));
    MFI.setRestorePoints(SaveRestorePoints({{Restore, {}}}));
    ++NumShrinkWrapped;
  }

  Changed |= shrinkWrapSeparately(Save, Restore);
  return Changed;
}

bool NewShrinkWrapImpl::isShrinkWrapEnabled(const MachineFunction &MF) {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const Function &F = MF.getFunction();
  // See ShrinkWrap for these restrictions.
  return TFI->enableShrinkWrapping(MF) &&
         !MF.getTarget().getMCAsmInfo().usesWindowsCFI() &&
         !(F.hasFnAttribute(Attribute::SanitizeAddress) ||
           F.hasFnAttribute(Attribute::SanitizeThread) ||
           F.hasFnAttribute(Attribute::SanitizeMemory) ||
           F.hasFnAttribute(Attribute::SanitizeType) ||
           F.hasFnAttribute(Attribute::SanitizeHWAddress));
}

bool NewShrinkWrapLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || MF.empty() ||
      !NewShrinkWrapImpl::isShrinkWrapEnabled(MF))
    return false;

  return NewShrinkWrapImpl(
             &getAnalysis<MachineRegisterClassInfoWrapperPass>().getRCI(),
             &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree(),
             &getAnalysis<MachinePostDominatorTreeWrapperPass>()
                  .getPostDomTree(),
             &getAnalysis<MachineLoopInfoWrapperPass>().getLI(),
             &getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI(),
             &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI(),
             &getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE())
      .run(MF);
}

PreservedAnalyses
NewShrinkWrapPass::run(MachineFunction &MF,
                       MachineFunctionAnalysisManager &MFAM) {
  MFPropsModifier _(*this, MF);
  if (MF.empty() || !NewShrinkWrapImpl::isShrinkWrapEnabled(MF))
    return PreservedAnalyses::all();

  bool Changed =
      NewShrinkWrapImpl(&MFAM.getResult<MachineRegisterClassAnalysis>(MF),
                        &MFAM.getResult<MachineDominatorTreeAnalysis>(MF),
                        &MFAM.getResult<MachinePostDominatorTreeAnalysis>(MF),
                        &MFAM.getResult<MachineLoopAnalysis>(MF),
                        &MFAM.getResult<MachineBlockFrequencyAnalysis>(MF),
                        &MFAM.getResult<MachineBranchProbabilityAnalysis>(MF),
                        &MFAM.getResult<MachineOptimizationRemarkEmitterAnalysis>(
                            MF))
          .run(MF);
  if (!Changed)
    return PreservedAnalyses::all();
  return getMachineFunctionPassPreservedAnalyses();
}
