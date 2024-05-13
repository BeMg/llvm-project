//===- RISCVInsertVSETVLI.cpp - Insert VSETVLI instructions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function pass that inserts VSETVLI instructions where
// needed and expands the vl outputs of VLEFF/VLSEGFF to PseudoReadVL
// instructions.
//
// This pass consists of 3 phases:
//
// Phase 1 collects how each basic block affects VL/VTYPE.
//
// Phase 2 uses the information from phase 1 to do a data flow analysis to
// propagate the VL/VTYPE changes through the function. This gives us the
// VL/VTYPE at the start of each basic block.
//
// Phase 3 inserts VSETVLI instructions in each basic block. Information from
// phase 2 is used to prevent inserting a VSETVLI before the first vector
// instruction in the block if possible.
//
//===----------------------------------------------------------------------===//

#include "RISCVInsertVSETVLI.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveDebugVariables.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <queue>
using namespace llvm;

#define DEBUG_TYPE "riscv-insert-vsetvli"
#define RISCV_INSERT_VSETVLI_NAME "RISC-V Insert VSETVLI pass"
#define RISCV_COALESCE_VSETVLI_NAME "RISC-V Coalesce VSETVLI pass"

STATISTIC(NumInsertedVSETVL, "Number of VSETVL inst inserted");
STATISTIC(NumCoalescedVSETVL, "Number of VSETVL inst coalesced");

static cl::opt<bool> DisableInsertVSETVLPHIOpt(
    "riscv-disable-insert-vsetvl-phi-opt", cl::init(false), cl::Hidden,
    cl::desc("Disable looking through phis when inserting vsetvlis."));

namespace {

/// Return true if this is 'vsetvli x0, x0, vtype' which preserves
/// VL and only sets VTYPE.
static bool isVLPreservingConfig(const MachineInstr &MI) {
  if (MI.getOpcode() != RISCV::PseudoVSETVLIX0)
    return false;
  assert(RISCV::X0 == MI.getOperand(1).getReg());
  return RISCV::X0 == MI.getOperand(0).getReg();
}

static bool isFloatScalarMoveOrScalarSplatInstr(const MachineInstr &MI) {
  switch (RISCV::getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case RISCV::VFMV_S_F:
  case RISCV::VFMV_V_F:
    return true;
  }
}

static bool isScalarInsertInstr(const MachineInstr &MI) {
  switch (RISCV::getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case RISCV::VMV_S_X:
  case RISCV::VFMV_S_F:
    return true;
  }
}

static bool isScalarSplatInstr(const MachineInstr &MI) {
  switch (RISCV::getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case RISCV::VMV_V_I:
  case RISCV::VMV_V_X:
  case RISCV::VFMV_V_F:
    return true;
  }
}

static bool isVSlideInstr(const MachineInstr &MI) {
  switch (RISCV::getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case RISCV::VSLIDEDOWN_VX:
  case RISCV::VSLIDEDOWN_VI:
  case RISCV::VSLIDEUP_VX:
  case RISCV::VSLIDEUP_VI:
    return true;
  }
}

/// Return true if this is an operation on mask registers.  Note that
/// this includes both arithmetic/logical ops and load/store (vlm/vsm).
static bool isMaskRegOp(const MachineInstr &MI) {
  if (!RISCVII::hasSEWOp(MI.getDesc().TSFlags))
    return false;
  const unsigned Log2SEW = MI.getOperand(getSEWOpNum(MI)).getImm();
  // A Log2SEW of 0 is an operation on mask registers only.
  return Log2SEW == 0;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const DemandedFields &DF) {
  DF.print(OS);
  return OS;
}
#endif

/// Return the fields and properties demanded by the provided instruction.
DemandedFields getDemanded(const MachineInstr &MI, const RISCVSubtarget *ST) {
  // This function works in coalesceVSETVLI too. We can still use the value of a
  // SEW, VL, or Policy operand even though it might not be the exact value in
  // the VL or VTYPE, since we only care about what the instruction originally
  // demanded.

  // Most instructions don't use any of these subfeilds.
  DemandedFields Res;
  // Start conservative if registers are used
  if (MI.isCall() || MI.isInlineAsm() ||
      MI.readsRegister(RISCV::VL, /*TRI=*/nullptr))
    Res.demandVL();
  if (MI.isCall() || MI.isInlineAsm() ||
      MI.readsRegister(RISCV::VTYPE, /*TRI=*/nullptr))
    Res.demandVTYPE();
  // Start conservative on the unlowered form too
  uint64_t TSFlags = MI.getDesc().TSFlags;
  if (RISCVII::hasSEWOp(TSFlags)) {
    Res.demandVTYPE();
    if (RISCVII::hasVLOp(TSFlags))
      if (const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
          !VLOp.isReg() || !VLOp.isUndef())
        Res.demandVL();

    // Behavior is independent of mask policy.
    if (!RISCVII::usesMaskPolicy(TSFlags))
      Res.MaskPolicy = false;
  }

  // Loads and stores with implicit EEW do not demand SEW or LMUL directly.
  // They instead demand the ratio of the two which is used in computing
  // EMUL, but which allows us the flexibility to change SEW and LMUL
  // provided we don't change the ratio.
  // Note: We assume that the instructions initial SEW is the EEW encoded
  // in the opcode.  This is asserted when constructing the VSETVLIInfo.
  if (getEEWForLoadStore(MI)) {
    Res.SEW = DemandedFields::SEWNone;
    Res.LMUL = DemandedFields::LMULNone;
  }

  // Store instructions don't use the policy fields.
  if (RISCVII::hasSEWOp(TSFlags) && MI.getNumExplicitDefs() == 0) {
    Res.TailPolicy = false;
    Res.MaskPolicy = false;
  }

  // If this is a mask reg operation, it only cares about VLMAX.
  // TODO: Possible extensions to this logic
  // * Probably ok if available VLMax is larger than demanded
  // * The policy bits can probably be ignored..
  if (isMaskRegOp(MI)) {
    Res.SEW = DemandedFields::SEWNone;
    Res.LMUL = DemandedFields::LMULNone;
  }

  // For vmv.s.x and vfmv.s.f, there are only two behaviors, VL = 0 and VL > 0.
  if (isScalarInsertInstr(MI)) {
    Res.LMUL = DemandedFields::LMULNone;
    Res.SEWLMULRatio = false;
    Res.VLAny = false;
    // For vmv.s.x and vfmv.s.f, if the merge operand is *undefined*, we don't
    // need to preserve any other bits and are thus compatible with any larger,
    // etype and can disregard policy bits.  Warning: It's tempting to try doing
    // this for any tail agnostic operation, but we can't as TA requires
    // tail lanes to either be the original value or -1.  We are writing
    // unknown bits to the lanes here.
    if (hasUndefinedMergeOp(MI)) {
      if (isFloatScalarMoveOrScalarSplatInstr(MI) && !ST->hasVInstructionsF64())
        Res.SEW = DemandedFields::SEWGreaterThanOrEqualAndLessThan64;
      else
        Res.SEW = DemandedFields::SEWGreaterThanOrEqual;
      Res.TailPolicy = false;
    }
  }

  // vmv.x.s, and vmv.f.s are unconditional and ignore everything except SEW.
  if (isScalarExtractInstr(MI)) {
    assert(!RISCVII::hasVLOp(TSFlags));
    Res.LMUL = DemandedFields::LMULNone;
    Res.SEWLMULRatio = false;
    Res.TailPolicy = false;
    Res.MaskPolicy = false;
  }

  if (RISCVII::hasVLOp(MI.getDesc().TSFlags)) {
    const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
    // A slidedown/slideup with an *undefined* merge op can freely clobber
    // elements not copied from the source vector (e.g. masked off, tail, or
    // slideup's prefix). Notes:
    // * We can't modify SEW here since the slide amount is in units of SEW.
    // * VL=1 is special only because we have existing support for zero vs
    //   non-zero VL.  We could generalize this if we had a VL > C predicate.
    // * The LMUL1 restriction is for machines whose latency may depend on VL.
    // * As above, this is only legal for tail "undefined" not "agnostic".
    if (isVSlideInstr(MI) && VLOp.isImm() && VLOp.getImm() == 1 &&
        hasUndefinedMergeOp(MI)) {
      Res.VLAny = false;
      Res.VLZeroness = true;
      Res.LMUL = DemandedFields::LMULLessThanOrEqualToM1;
      Res.TailPolicy = false;
    }

    // A tail undefined vmv.v.i/x or vfmv.v.f with VL=1 can be treated in the
    // same semantically as vmv.s.x.  This is particularly useful since we don't
    // have an immediate form of vmv.s.x, and thus frequently use vmv.v.i in
    // it's place. Since a splat is non-constant time in LMUL, we do need to be
    // careful to not increase the number of active vector registers (unlike for
    // vmv.s.x.)
    if (isScalarSplatInstr(MI) && VLOp.isImm() && VLOp.getImm() == 1 &&
        hasUndefinedMergeOp(MI)) {
      Res.LMUL = DemandedFields::LMULLessThanOrEqualToM1;
      Res.SEWLMULRatio = false;
      Res.VLAny = false;
      if (isFloatScalarMoveOrScalarSplatInstr(MI) && !ST->hasVInstructionsF64())
        Res.SEW = DemandedFields::SEWGreaterThanOrEqualAndLessThan64;
      else
        Res.SEW = DemandedFields::SEWGreaterThanOrEqual;
      Res.TailPolicy = false;
    }
  }

  return Res;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_ATTRIBUTE_USED
inline raw_ostream &operator<<(raw_ostream &OS, const VSETVLIInfo &V) {
  V.print(OS);
  return OS;
}
#endif

struct BlockData {
  // The VSETVLIInfo that represents the VL/VTYPE settings on exit from this
  // block. Calculated in Phase 2.
  VSETVLIInfo Exit;

  // The VSETVLIInfo that represents the VL/VTYPE settings from all predecessor
  // blocks. Calculated in Phase 2, and used by Phase 3.
  VSETVLIInfo Pred;

  // Keeps track of whether the block is already in the queue.
  bool InQueue = false;

  BlockData() = default;
};

class RISCVInsertVSETVLI : public MachineFunctionPass {
  const RISCVSubtarget *ST;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  // Possibly null!
  LiveIntervals *LIS;

  std::vector<BlockData> BlockInfo;
  std::queue<const MachineBasicBlock *> WorkList;

public:
  static char ID;

  RISCVInsertVSETVLI() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();

    AU.addUsedIfAvailable<LiveIntervals>();
    AU.addPreserved<LiveIntervals>();
    AU.addPreserved<SlotIndexes>();
    AU.addPreserved<LiveDebugVariables>();
    AU.addPreserved<LiveStacks>();

    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_INSERT_VSETVLI_NAME; }

private:
  bool needVSETVLI(const DemandedFields &Used, const VSETVLIInfo &Require,
                   const VSETVLIInfo &CurInfo) const;
  bool needVSETVLIPHI(const VSETVLIInfo &Require,
                      const MachineBasicBlock &MBB) const;
  void insertVSETVLI(MachineBasicBlock &MBB, MachineInstr &MI,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo);
  void insertVSETVLI(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, DebugLoc DL,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo);

  void transferBefore(VSETVLIInfo &Info, const MachineInstr &MI) const;
  void transferAfter(VSETVLIInfo &Info, const MachineInstr &MI) const;
  bool computeVLVTYPEChanges(const MachineBasicBlock &MBB,
                             VSETVLIInfo &Info) const;
  void computeIncomingVLVTYPE(const MachineBasicBlock &MBB);
  void emitVSETVLIs(MachineBasicBlock &MBB);
  void doPRE(MachineBasicBlock &MBB);
  void insertReadVL(MachineBasicBlock &MBB);

  bool canMutatePriorConfig(const MachineInstr &PrevMI, const MachineInstr &MI,
                            const DemandedFields &Used) const;
  void coalesceVSETVLIs(MachineBasicBlock &MBB) const;

  VSETVLIInfo getInfoForVSETVLI(const MachineInstr &MI) const;
  VSETVLIInfo computeInfoForInstr(const MachineInstr &MI) const;
};

} // end anonymous namespace

char RISCVInsertVSETVLI::ID = 0;
char &llvm::RISCVInsertVSETVLIID = RISCVInsertVSETVLI::ID;

INITIALIZE_PASS(RISCVInsertVSETVLI, DEBUG_TYPE, RISCV_INSERT_VSETVLI_NAME,
                false, false)

// Return a VSETVLIInfo representing the changes made by this VSETVLI or
// VSETIVLI instruction.
VSETVLIInfo
RISCVInsertVSETVLI::getInfoForVSETVLI(const MachineInstr &MI) const {
  VSETVLIInfo NewInfo;
  if (MI.getOpcode() == RISCV::PseudoVSETIVLI) {
    NewInfo.setAVLImm(MI.getOperand(1).getImm());
  } else {
    assert(MI.getOpcode() == RISCV::PseudoVSETVLI ||
           MI.getOpcode() == RISCV::PseudoVSETVLIX0);
    Register AVLReg = MI.getOperand(1).getReg();
    assert((AVLReg != RISCV::X0 || MI.getOperand(0).getReg() != RISCV::X0) &&
           "Can't handle X0, X0 vsetvli yet");
    if (AVLReg == RISCV::X0)
      NewInfo.setAVLVLMAX();
    else if (MI.getOperand(1).isUndef())
      // Otherwise use an AVL of 1 to avoid depending on previous vl.
      NewInfo.setAVLImm(1);
    else {
      VNInfo *VNI = getVNInfoFromReg(AVLReg, MI, LIS);
      NewInfo.setAVLRegDef(VNI, AVLReg);
    }
  }
  NewInfo.setVTYPE(MI.getOperand(2).getImm());

  return NewInfo;
}

static unsigned computeVLMAX(unsigned VLEN, unsigned SEW,
                             RISCVII::VLMUL VLMul) {
  auto [LMul, Fractional] = RISCVVType::decodeVLMUL(VLMul);
  if (Fractional)
    VLEN = VLEN / LMul;
  else
    VLEN = VLEN * LMul;
  return VLEN/SEW;
}

VSETVLIInfo
RISCVInsertVSETVLI::computeInfoForInstr(const MachineInstr &MI) const {
  VSETVLIInfo InstrInfo;
  const uint64_t TSFlags = MI.getDesc().TSFlags;

  bool TailAgnostic = true;
  bool MaskAgnostic = true;
  if (!hasUndefinedMergeOp(MI)) {
    // Start with undisturbed.
    TailAgnostic = false;
    MaskAgnostic = false;

    // If there is a policy operand, use it.
    if (RISCVII::hasVecPolicyOp(TSFlags)) {
      const MachineOperand &Op = MI.getOperand(MI.getNumExplicitOperands() - 1);
      uint64_t Policy = Op.getImm();
      assert(Policy <= (RISCVII::TAIL_AGNOSTIC | RISCVII::MASK_AGNOSTIC) &&
             "Invalid Policy Value");
      TailAgnostic = Policy & RISCVII::TAIL_AGNOSTIC;
      MaskAgnostic = Policy & RISCVII::MASK_AGNOSTIC;
    }

    // Some pseudo instructions force a tail agnostic policy despite having a
    // tied def.
    if (RISCVII::doesForceTailAgnostic(TSFlags))
      TailAgnostic = true;

    if (!RISCVII::usesMaskPolicy(TSFlags))
      MaskAgnostic = true;
  }

  RISCVII::VLMUL VLMul = RISCVII::getLMul(TSFlags);

  unsigned Log2SEW = MI.getOperand(getSEWOpNum(MI)).getImm();
  // A Log2SEW of 0 is an operation on mask registers only.
  unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
  assert(RISCVVType::isValidSEW(SEW) && "Unexpected SEW");

  if (RISCVII::hasVLOp(TSFlags)) {
    const MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
    if (VLOp.isImm()) {
      int64_t Imm = VLOp.getImm();
      // Conver the VLMax sentintel to X0 register.
      if (Imm == RISCV::VLMaxSentinel) {
        // If we know the exact VLEN, see if we can use the constant encoding
        // for the VLMAX instead.  This reduces register pressure slightly.
        const unsigned VLMAX = computeVLMAX(ST->getRealMaxVLen(), SEW, VLMul);
        if (ST->getRealMinVLen() == ST->getRealMaxVLen() && VLMAX <= 31)
          InstrInfo.setAVLImm(VLMAX);
        else
          InstrInfo.setAVLVLMAX();
      }
      else
        InstrInfo.setAVLImm(Imm);
    } else if (VLOp.isUndef()) {
      // Otherwise use an AVL of 1 to avoid depending on previous vl.
      InstrInfo.setAVLImm(1);
    } else {
      VNInfo *VNI = getVNInfoFromReg(VLOp.getReg(), MI, LIS);
      InstrInfo.setAVLRegDef(VNI, VLOp.getReg());
    }
  } else {
    assert(isScalarExtractInstr(MI));
    // Pick a random value for state tracking purposes, will be ignored via
    // the demanded fields mechanism
    InstrInfo.setAVLImm(1);
  }
#ifndef NDEBUG
  if (std::optional<unsigned> EEW = getEEWForLoadStore(MI)) {
    assert(SEW == EEW && "Initial SEW doesn't match expected EEW");
  }
#endif
  InstrInfo.setVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic);

  // If AVL is defined by a vsetvli with the same VLMAX, we can replace the
  // AVL operand with the AVL of the defining vsetvli.  We avoid general
  // register AVLs to avoid extending live ranges without being sure we can
  // kill the original source reg entirely.
  if (InstrInfo.hasAVLReg()) {
    if (const MachineInstr *DefMI = InstrInfo.getAVLDefMI(LIS);
        DefMI && isVectorConfigInstr(*DefMI)) {
      VSETVLIInfo DefInstrInfo = getInfoForVSETVLI(*DefMI);
      if (DefInstrInfo.hasSameVLMAX(InstrInfo) &&
          (DefInstrInfo.hasAVLImm() || DefInstrInfo.hasAVLVLMAX()))
        InstrInfo.setAVL(DefInstrInfo);
    }
  }

  return InstrInfo;
}

void RISCVInsertVSETVLI::insertVSETVLI(MachineBasicBlock &MBB, MachineInstr &MI,
                                       const VSETVLIInfo &Info,
                                       const VSETVLIInfo &PrevInfo) {
  DebugLoc DL = MI.getDebugLoc();
  insertVSETVLI(MBB, MachineBasicBlock::iterator(&MI), DL, Info, PrevInfo);
}

void RISCVInsertVSETVLI::insertVSETVLI(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, DebugLoc DL,
                     const VSETVLIInfo &Info, const VSETVLIInfo &PrevInfo) {

  ++NumInsertedVSETVL;
  if (PrevInfo.isValid() && !PrevInfo.isUnknown()) {
    // Use X0, X0 form if the AVL is the same and the SEW+LMUL gives the same
    // VLMAX.
    if (Info.hasSameAVL(PrevInfo) && Info.hasSameVLMAX(PrevInfo)) {
      auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0))
                    .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                    .addReg(RISCV::X0, RegState::Kill)
                    .addImm(Info.encodeVTYPE())
                    .addReg(RISCV::VL, RegState::Implicit);
      if (LIS)
        LIS->InsertMachineInstrInMaps(*MI);
      return;
    }

    // If our AVL is a virtual register, it might be defined by a VSET(I)VLI. If
    // it has the same VLMAX we want and the last VL/VTYPE we observed is the
    // same, we can use the X0, X0 form.
    if (Info.hasSameVLMAX(PrevInfo) && Info.hasAVLReg()) {
      if (const MachineInstr *DefMI = Info.getAVLDefMI(LIS);
          DefMI && isVectorConfigInstr(*DefMI)) {
        VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
        if (DefInfo.hasSameAVL(PrevInfo) && DefInfo.hasSameVLMAX(PrevInfo)) {
          auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0))
                        .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                        .addReg(RISCV::X0, RegState::Kill)
                        .addImm(Info.encodeVTYPE())
                        .addReg(RISCV::VL, RegState::Implicit);
          if (LIS)
            LIS->InsertMachineInstrInMaps(*MI);
          return;
        }
      }
    }
  }

  if (Info.hasAVLImm()) {
    auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETIVLI))
                  .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                  .addImm(Info.getAVLImm())
                  .addImm(Info.encodeVTYPE());
    if (LIS)
      LIS->InsertMachineInstrInMaps(*MI);
    return;
  }

  if (Info.hasAVLVLMAX()) {
    Register DestReg = MRI->createVirtualRegister(&RISCV::GPRRegClass);
    auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLIX0))
                  .addReg(DestReg, RegState::Define | RegState::Dead)
                  .addReg(RISCV::X0, RegState::Kill)
                  .addImm(Info.encodeVTYPE());
    if (LIS) {
      LIS->InsertMachineInstrInMaps(*MI);
      LIS->createAndComputeVirtRegInterval(DestReg);
    }
    return;
  }

  Register AVLReg = Info.getAVLReg();
  MRI->constrainRegClass(AVLReg, &RISCV::GPRNoX0RegClass);
  auto MI = BuildMI(MBB, InsertPt, DL, TII->get(RISCV::PseudoVSETVLI))
                .addReg(RISCV::X0, RegState::Define | RegState::Dead)
                .addReg(AVLReg)
                .addImm(Info.encodeVTYPE());
  if (LIS) {
    LIS->InsertMachineInstrInMaps(*MI);
    // Normally the AVL's live range will already extend past the inserted
    // vsetvli because the pseudos below will already use the AVL. But this
    // isn't always the case, e.g. PseudoVMV_X_S doesn't have an AVL operand.
    LIS->getInterval(AVLReg).extendInBlock(
        LIS->getMBBStartIdx(&MBB), LIS->getInstructionIndex(*MI).getRegSlot());
  }
}

/// Return true if a VSETVLI is required to transition from CurInfo to Require
/// given a set of DemandedFields \p Used.
bool RISCVInsertVSETVLI::needVSETVLI(const DemandedFields &Used,
                                     const VSETVLIInfo &Require,
                                     const VSETVLIInfo &CurInfo) const {
  if (!CurInfo.isValid() || CurInfo.isUnknown() || CurInfo.hasSEWLMULRatioOnly())
    return true;

  if (CurInfo.isCompatible(Used, Require, LIS))
    return false;

  // We didn't find a compatible value. If our AVL is a virtual register,
  // it might be defined by a VSET(I)VLI. If it has the same VLMAX we need
  // and the last VL/VTYPE we observed is the same, we don't need a
  // VSETVLI here.
  if (Require.hasAVLReg() && CurInfo.hasCompatibleVTYPE(Used, Require)) {
    if (const MachineInstr *DefMI = Require.getAVLDefMI(LIS);
        DefMI && isVectorConfigInstr(*DefMI)) {
      VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
      if (DefInfo.hasSameAVL(CurInfo) && DefInfo.hasSameVLMAX(CurInfo))
        return false;
    }
  }

  return true;
}

// If we don't use LMUL or the SEW/LMUL ratio, then adjust LMUL so that we
// maintain the SEW/LMUL ratio. This allows us to eliminate VL toggles in more
// places.
static VSETVLIInfo adjustIncoming(VSETVLIInfo PrevInfo, VSETVLIInfo NewInfo,
                                  DemandedFields &Demanded) {
  VSETVLIInfo Info = NewInfo;

  if (!Demanded.LMUL && !Demanded.SEWLMULRatio && PrevInfo.isValid() &&
      !PrevInfo.isUnknown()) {
    if (auto NewVLMul = RISCVVType::getSameRatioLMUL(
            PrevInfo.getSEW(), PrevInfo.getVLMUL(), Info.getSEW()))
      Info.setVLMul(*NewVLMul);
    Demanded.LMUL = DemandedFields::LMULEqual;
  }

  return Info;
}

// Given an incoming state reaching MI, minimally modifies that state so that it
// is compatible with MI. The resulting state is guaranteed to be semantically
// legal for MI, but may not be the state requested by MI.
void RISCVInsertVSETVLI::transferBefore(VSETVLIInfo &Info,
                                        const MachineInstr &MI) const {
  if (!RISCVII::hasSEWOp(MI.getDesc().TSFlags))
    return;

  DemandedFields Demanded = getDemanded(MI, ST);

  const VSETVLIInfo NewInfo = computeInfoForInstr(MI);
  assert(NewInfo.isValid() && !NewInfo.isUnknown());
  if (Info.isValid() && !needVSETVLI(Demanded, NewInfo, Info))
    return;

  const VSETVLIInfo PrevInfo = Info;
  if (!Info.isValid() || Info.isUnknown())
    Info = NewInfo;

  const VSETVLIInfo IncomingInfo = adjustIncoming(PrevInfo, NewInfo, Demanded);

  // If MI only demands that VL has the same zeroness, we only need to set the
  // AVL if the zeroness differs.  This removes a vsetvli entirely if the types
  // match or allows use of cheaper avl preserving variant if VLMAX doesn't
  // change. If VLMAX might change, we couldn't use the 'vsetvli x0, x0, vtype"
  // variant, so we avoid the transform to prevent extending live range of an
  // avl register operand.
  // TODO: We can probably relax this for immediates.
  bool EquallyZero = IncomingInfo.hasEquallyZeroAVL(PrevInfo, LIS) &&
                     IncomingInfo.hasSameVLMAX(PrevInfo);
  if (Demanded.VLAny || (Demanded.VLZeroness && !EquallyZero))
    Info.setAVL(IncomingInfo);

  Info.setVTYPE(
      ((Demanded.LMUL || Demanded.SEWLMULRatio) ? IncomingInfo : Info)
          .getVLMUL(),
      ((Demanded.SEW || Demanded.SEWLMULRatio) ? IncomingInfo : Info).getSEW(),
      // Prefer tail/mask agnostic since it can be relaxed to undisturbed later
      // if needed.
      (Demanded.TailPolicy ? IncomingInfo : Info).getTailAgnostic() ||
          IncomingInfo.getTailAgnostic(),
      (Demanded.MaskPolicy ? IncomingInfo : Info).getMaskAgnostic() ||
          IncomingInfo.getMaskAgnostic());

  // If we only knew the sew/lmul ratio previously, replace the VTYPE but keep
  // the AVL.
  if (Info.hasSEWLMULRatioOnly()) {
    VSETVLIInfo RatiolessInfo = IncomingInfo;
    RatiolessInfo.setAVL(Info);
    Info = RatiolessInfo;
  }
}

// Given a state with which we evaluated MI (see transferBefore above for why
// this might be different that the state MI requested), modify the state to
// reflect the changes MI might make.
void RISCVInsertVSETVLI::transferAfter(VSETVLIInfo &Info,
                                       const MachineInstr &MI) const {
  if (isVectorConfigInstr(MI)) {
    Info = getInfoForVSETVLI(MI);
    return;
  }

  if (RISCV::isFaultFirstLoad(MI)) {
    // Update AVL to vl-output of the fault first load.
    assert(MI.getOperand(1).getReg().isVirtual());
    if (LIS) {
      auto &LI = LIS->getInterval(MI.getOperand(1).getReg());
      SlotIndex SI =
          LIS->getSlotIndexes()->getInstructionIndex(MI).getRegSlot();
      VNInfo *VNI = LI.getVNInfoAt(SI);
      Info.setAVLRegDef(VNI, MI.getOperand(1).getReg());
    } else
      Info.setAVLRegDef(nullptr, MI.getOperand(1).getReg());
    return;
  }

  // If this is something that updates VL/VTYPE that we don't know about, set
  // the state to unknown.
  if (MI.isCall() || MI.isInlineAsm() ||
      MI.modifiesRegister(RISCV::VL, /*TRI=*/nullptr) ||
      MI.modifiesRegister(RISCV::VTYPE, /*TRI=*/nullptr))
    Info = VSETVLIInfo::getUnknown();
}

bool RISCVInsertVSETVLI::computeVLVTYPEChanges(const MachineBasicBlock &MBB,
                                               VSETVLIInfo &Info) const {
  bool HadVectorOp = false;

  Info = BlockInfo[MBB.getNumber()].Pred;
  for (const MachineInstr &MI : MBB) {
    transferBefore(Info, MI);

    if (isVectorConfigInstr(MI) || RISCVII::hasSEWOp(MI.getDesc().TSFlags))
      HadVectorOp = true;

    transferAfter(Info, MI);
  }

  return HadVectorOp;
}

void RISCVInsertVSETVLI::computeIncomingVLVTYPE(const MachineBasicBlock &MBB) {

  BlockData &BBInfo = BlockInfo[MBB.getNumber()];

  BBInfo.InQueue = false;

  // Start with the previous entry so that we keep the most conservative state
  // we have ever found.
  VSETVLIInfo InInfo = BBInfo.Pred;
  if (MBB.pred_empty()) {
    // There are no predecessors, so use the default starting status.
    InInfo.setUnknown();
  } else {
    for (MachineBasicBlock *P : MBB.predecessors())
      InInfo = InInfo.intersect(BlockInfo[P->getNumber()].Exit);
  }

  // If we don't have any valid predecessor value, wait until we do.
  if (!InInfo.isValid())
    return;

  // If no change, no need to rerun block
  if (InInfo == BBInfo.Pred)
    return;

  BBInfo.Pred = InInfo;
  LLVM_DEBUG(dbgs() << "Entry state of " << printMBBReference(MBB)
                    << " changed to " << BBInfo.Pred << "\n");

  // Note: It's tempting to cache the state changes here, but due to the
  // compatibility checks performed a blocks output state can change based on
  // the input state.  To cache, we'd have to add logic for finding
  // never-compatible state changes.
  VSETVLIInfo TmpStatus;
  computeVLVTYPEChanges(MBB, TmpStatus);

  // If the new exit value matches the old exit value, we don't need to revisit
  // any blocks.
  if (BBInfo.Exit == TmpStatus)
    return;

  BBInfo.Exit = TmpStatus;
  LLVM_DEBUG(dbgs() << "Exit state of " << printMBBReference(MBB)
                    << " changed to " << BBInfo.Exit << "\n");

  // Add the successors to the work list so we can propagate the changed exit
  // status.
  for (MachineBasicBlock *S : MBB.successors())
    if (!BlockInfo[S->getNumber()].InQueue) {
      BlockInfo[S->getNumber()].InQueue = true;
      WorkList.push(S);
    }
}

// If we weren't able to prove a vsetvli was directly unneeded, it might still
// be unneeded if the AVL was a phi node where all incoming values are VL
// outputs from the last VSETVLI in their respective basic blocks.
bool RISCVInsertVSETVLI::needVSETVLIPHI(const VSETVLIInfo &Require,
                                        const MachineBasicBlock &MBB) const {
  if (DisableInsertVSETVLPHIOpt)
    return true;

  if (!Require.hasAVLReg())
    return true;

  if (!LIS)
    return true;

  // We need the AVL to have been produced by a PHI node in this basic block.
  const VNInfo *Valno = Require.getAVLVNInfo();
  if (!Valno->isPHIDef() || LIS->getMBBFromIndex(Valno->def) != &MBB)
    return true;

  const LiveRange &LR = LIS->getInterval(Require.getAVLReg());

  for (auto *PBB : MBB.predecessors()) {
    const VSETVLIInfo &PBBExit = BlockInfo[PBB->getNumber()].Exit;

    // We need the PHI input to the be the output of a VSET(I)VLI.
    const VNInfo *Value = LR.getVNInfoBefore(LIS->getMBBEndIdx(PBB));
    if (!Value)
      return true;
    MachineInstr *DefMI = LIS->getInstructionFromIndex(Value->def);
    if (!DefMI || !isVectorConfigInstr(*DefMI))
      return true;

    // We found a VSET(I)VLI make sure it matches the output of the
    // predecessor block.
    VSETVLIInfo DefInfo = getInfoForVSETVLI(*DefMI);
    if (DefInfo != PBBExit)
      return true;

    // Require has the same VL as PBBExit, so if the exit from the
    // predecessor has the VTYPE we are looking for we might be able
    // to avoid a VSETVLI.
    if (PBBExit.isUnknown() || !PBBExit.hasSameVTYPE(Require))
      return true;
  }

  // If all the incoming values to the PHI checked out, we don't need
  // to insert a VSETVLI.
  return false;
}

void RISCVInsertVSETVLI::emitVSETVLIs(MachineBasicBlock &MBB) {
  VSETVLIInfo CurInfo = BlockInfo[MBB.getNumber()].Pred;
  // Track whether the prefix of the block we've scanned is transparent
  // (meaning has not yet changed the abstract state).
  bool PrefixTransparent = true;
  for (MachineInstr &MI : MBB) {
    const VSETVLIInfo PrevInfo = CurInfo;
    transferBefore(CurInfo, MI);

    // If this is an explicit VSETVLI or VSETIVLI, update our state.
    if (isVectorConfigInstr(MI)) {
      // Conservatively, mark the VL and VTYPE as live.
      assert(MI.getOperand(3).getReg() == RISCV::VL &&
             MI.getOperand(4).getReg() == RISCV::VTYPE &&
             "Unexpected operands where VL and VTYPE should be");
      MI.getOperand(3).setIsDead(false);
      MI.getOperand(4).setIsDead(false);
      PrefixTransparent = false;
    }

    uint64_t TSFlags = MI.getDesc().TSFlags;
    if (RISCVII::hasSEWOp(TSFlags)) {
      if (!PrevInfo.isCompatible(DemandedFields::all(), CurInfo, LIS)) {
        // If this is the first implicit state change, and the state change
        // requested can be proven to produce the same register contents, we
        // can skip emitting the actual state change and continue as if we
        // had since we know the GPR result of the implicit state change
        // wouldn't be used and VL/VTYPE registers are correct.  Note that
        // we *do* need to model the state as if it changed as while the
        // register contents are unchanged, the abstract model can change.
        if (!PrefixTransparent || needVSETVLIPHI(CurInfo, MBB))
          insertVSETVLI(MBB, MI, CurInfo, PrevInfo);
        PrefixTransparent = false;
      }

      if (RISCVII::hasVLOp(TSFlags)) {
        MachineOperand &VLOp = MI.getOperand(getVLOpNum(MI));
        if (VLOp.isReg()) {
          Register Reg = VLOp.getReg();

          // Erase the AVL operand from the instruction.
          VLOp.setReg(RISCV::NoRegister);
          VLOp.setIsKill(false);
          if (LIS) {
            LiveInterval &LI = LIS->getInterval(Reg);
            SmallVector<MachineInstr *> DeadMIs;
            LIS->shrinkToUses(&LI, &DeadMIs);
            // We might have separate components that need split due to
            // needVSETVLIPHI causing us to skip inserting a new VL def.
            SmallVector<LiveInterval *> SplitLIs;
            LIS->splitSeparateComponents(LI, SplitLIs);

            // If the AVL was an immediate > 31, then it would have been emitted
            // as an ADDI. However, the ADDI might not have been used in the
            // vsetvli, or a vsetvli might not have been emitted, so it may be
            // dead now.
            for (MachineInstr *DeadMI : DeadMIs) {
              if (!TII->isAddImmediate(*DeadMI, Reg))
                continue;
              LIS->RemoveMachineInstrFromMaps(*DeadMI);
              DeadMI->eraseFromParent();
            }
          }
        }
        MI.addOperand(MachineOperand::CreateReg(RISCV::VL, /*isDef*/ false,
                                                /*isImp*/ true));
      }
      MI.addOperand(MachineOperand::CreateReg(RISCV::VTYPE, /*isDef*/ false,
                                              /*isImp*/ true));
    }

    if (MI.isCall() || MI.isInlineAsm() ||
        MI.modifiesRegister(RISCV::VL, /*TRI=*/nullptr) ||
        MI.modifiesRegister(RISCV::VTYPE, /*TRI=*/nullptr))
      PrefixTransparent = false;

    transferAfter(CurInfo, MI);
  }

  const auto &Info = BlockInfo[MBB.getNumber()];
  if (CurInfo != Info.Exit) {
    LLVM_DEBUG(dbgs() << "in block " << printMBBReference(MBB) << "\n");
    LLVM_DEBUG(dbgs() << "  begin        state: " << Info.Pred << "\n");
    LLVM_DEBUG(dbgs() << "  expected end state: " << Info.Exit << "\n");
    LLVM_DEBUG(dbgs() << "  actual   end state: " << CurInfo << "\n");
  }
  assert(CurInfo == Info.Exit && "InsertVSETVLI dataflow invariant violated");
}

/// Perform simple partial redundancy elimination of the VSETVLI instructions
/// we're about to insert by looking for cases where we can PRE from the
/// beginning of one block to the end of one of its predecessors.  Specifically,
/// this is geared to catch the common case of a fixed length vsetvl in a single
/// block loop when it could execute once in the preheader instead.
void RISCVInsertVSETVLI::doPRE(MachineBasicBlock &MBB) {
  if (!BlockInfo[MBB.getNumber()].Pred.isUnknown())
    return;

  MachineBasicBlock *UnavailablePred = nullptr;
  VSETVLIInfo AvailableInfo;
  for (MachineBasicBlock *P : MBB.predecessors()) {
    const VSETVLIInfo &PredInfo = BlockInfo[P->getNumber()].Exit;
    if (PredInfo.isUnknown()) {
      if (UnavailablePred)
        return;
      UnavailablePred = P;
    } else if (!AvailableInfo.isValid()) {
      AvailableInfo = PredInfo;
    } else if (AvailableInfo != PredInfo) {
      return;
    }
  }

  // Unreachable, single pred, or full redundancy. Note that FRE is handled by
  // phase 3.
  if (!UnavailablePred || !AvailableInfo.isValid())
    return;

  if (!LIS)
    return;

  // If we don't know the exact VTYPE, we can't copy the vsetvli to the exit of
  // the unavailable pred.
  if (AvailableInfo.hasSEWLMULRatioOnly())
    return;

  // Critical edge - TODO: consider splitting?
  if (UnavailablePred->succ_size() != 1)
    return;

  // If the AVL value is a register (other than our VLMAX sentinel),
  // we need to prove the value is available at the point we're going
  // to insert the vsetvli at.
  if (AvailableInfo.hasAVLReg()) {
    SlotIndex SI = AvailableInfo.getAVLVNInfo()->def;
    // This is an inline dominance check which covers the case of
    // UnavailablePred being the preheader of a loop.
    if (LIS->getMBBFromIndex(SI) != UnavailablePred)
      return;
    if (!UnavailablePred->terminators().empty() &&
        SI >= LIS->getInstructionIndex(*UnavailablePred->getFirstTerminator()))
      return;
  }

  // Model the effect of changing the input state of the block MBB to
  // AvailableInfo.  We're looking for two issues here; one legality,
  // one profitability.
  // 1) If the block doesn't use some of the fields from VL or VTYPE, we
  //    may hit the end of the block with a different end state.  We can
  //    not make this change without reflowing later blocks as well.
  // 2) If we don't actually remove a transition, inserting a vsetvli
  //    into the predecessor block would be correct, but unprofitable.
  VSETVLIInfo OldInfo = BlockInfo[MBB.getNumber()].Pred;
  VSETVLIInfo CurInfo = AvailableInfo;
  int TransitionsRemoved = 0;
  for (const MachineInstr &MI : MBB) {
    const VSETVLIInfo LastInfo = CurInfo;
    const VSETVLIInfo LastOldInfo = OldInfo;
    transferBefore(CurInfo, MI);
    transferBefore(OldInfo, MI);
    if (CurInfo == LastInfo)
      TransitionsRemoved++;
    if (LastOldInfo == OldInfo)
      TransitionsRemoved--;
    transferAfter(CurInfo, MI);
    transferAfter(OldInfo, MI);
    if (CurInfo == OldInfo)
      // Convergence.  All transitions after this must match by construction.
      break;
  }
  if (CurInfo != OldInfo || TransitionsRemoved <= 0)
    // Issues 1 and 2 above
    return;

  // Finally, update both data flow state and insert the actual vsetvli.
  // Doing both keeps the code in sync with the dataflow results, which
  // is critical for correctness of phase 3.
  auto OldExit = BlockInfo[UnavailablePred->getNumber()].Exit;
  LLVM_DEBUG(dbgs() << "PRE VSETVLI from " << MBB.getName() << " to "
                    << UnavailablePred->getName() << " with state "
                    << AvailableInfo << "\n");
  BlockInfo[UnavailablePred->getNumber()].Exit = AvailableInfo;
  BlockInfo[MBB.getNumber()].Pred = AvailableInfo;

  // Note there's an implicit assumption here that terminators never use
  // or modify VL or VTYPE.  Also, fallthrough will return end().
  auto InsertPt = UnavailablePred->getFirstInstrTerminator();
  insertVSETVLI(*UnavailablePred, InsertPt,
                UnavailablePred->findDebugLoc(InsertPt),
                AvailableInfo, OldExit);
}

// Return true if we can mutate PrevMI to match MI without changing any the
// fields which would be observed.
bool RISCVInsertVSETVLI::canMutatePriorConfig(
    const MachineInstr &PrevMI, const MachineInstr &MI,
    const DemandedFields &Used) const {
  // If the VL values aren't equal, return false if either a) the former is
  // demanded, or b) we can't rewrite the former to be the later for
  // implementation reasons.
  if (!isVLPreservingConfig(MI)) {
    if (Used.VLAny)
      return false;

    if (Used.VLZeroness) {
      if (isVLPreservingConfig(PrevMI))
        return false;
      if (!getInfoForVSETVLI(PrevMI).hasEquallyZeroAVL(getInfoForVSETVLI(MI),
                                                       LIS))
        return false;
    }

    auto &AVL = MI.getOperand(1);
    auto &PrevAVL = PrevMI.getOperand(1);

    // If the AVL is a register, we need to make sure MI's AVL dominates PrevMI.
    // For now just check that PrevMI uses the same virtual register.
    if (AVL.isReg() && AVL.getReg() != RISCV::X0 &&
        (!MRI->hasOneDef(AVL.getReg()) || !PrevAVL.isReg() ||
         PrevAVL.getReg() != AVL.getReg()))
      return false;
  }

  assert(PrevMI.getOperand(2).isImm() && MI.getOperand(2).isImm());
  auto PriorVType = PrevMI.getOperand(2).getImm();
  auto VType = MI.getOperand(2).getImm();
  return areCompatibleVTYPEs(PriorVType, VType, Used);
}

void RISCVInsertVSETVLI::coalesceVSETVLIs(MachineBasicBlock &MBB) const {
  MachineInstr *NextMI = nullptr;
  // We can have arbitrary code in successors, so VL and VTYPE
  // must be considered demanded.
  DemandedFields Used;
  Used.demandVL();
  Used.demandVTYPE();
  SmallVector<MachineInstr*> ToDelete;
  for (MachineInstr &MI : make_range(MBB.rbegin(), MBB.rend())) {

    if (!isVectorConfigInstr(MI)) {
      Used.doUnion(getDemanded(MI, ST));
      if (MI.isCall() || MI.isInlineAsm() ||
          MI.modifiesRegister(RISCV::VL, /*TRI=*/nullptr) ||
          MI.modifiesRegister(RISCV::VTYPE, /*TRI=*/nullptr))
        NextMI = nullptr;
      continue;
    }

    if (!MI.getOperand(0).isDead())
      Used.demandVL();

    if (NextMI) {
      if (!Used.usedVL() && !Used.usedVTYPE()) {
        ToDelete.push_back(&MI);
        // Leave NextMI unchanged
        continue;
      }

      if (canMutatePriorConfig(MI, *NextMI, Used)) {
        if (!isVLPreservingConfig(*NextMI)) {
          Register DefReg = NextMI->getOperand(0).getReg();

          MI.getOperand(0).setReg(DefReg);
          MI.getOperand(0).setIsDead(false);

          // The def of DefReg moved to MI, so extend the LiveInterval up to
          // it.
          if (DefReg.isVirtual() && LIS) {
            LiveInterval &DefLI = LIS->getInterval(DefReg);
            SlotIndex MISlot = LIS->getInstructionIndex(MI).getRegSlot();
            VNInfo *DefVNI = DefLI.getVNInfoAt(DefLI.beginIndex());
            LiveInterval::Segment S(MISlot, DefLI.beginIndex(), DefVNI);
            DefLI.addSegment(S);
            DefVNI->def = MISlot;
            // Mark DefLI as spillable if it was previously unspillable
            DefLI.setWeight(0);

            // DefReg may have had no uses, in which case we need to shrink
            // the LiveInterval up to MI.
            LIS->shrinkToUses(&DefLI);
          }

          Register OldVLReg;
          if (MI.getOperand(1).isReg())
            OldVLReg = MI.getOperand(1).getReg();
          if (NextMI->getOperand(1).isImm())
            MI.getOperand(1).ChangeToImmediate(NextMI->getOperand(1).getImm());
          else
            MI.getOperand(1).ChangeToRegister(NextMI->getOperand(1).getReg(), false);

          // Clear NextMI's AVL early so we're not counting it as a use.
          if (NextMI->getOperand(1).isReg())
            NextMI->getOperand(1).setReg(RISCV::NoRegister);

          if (OldVLReg && OldVLReg.isVirtual()) {
            // NextMI no longer uses OldVLReg so shrink its LiveInterval.
            if (LIS)
              LIS->shrinkToUses(&LIS->getInterval(OldVLReg));

            MachineInstr *VLOpDef = MRI->getUniqueVRegDef(OldVLReg);
            if (VLOpDef && TII->isAddImmediate(*VLOpDef, OldVLReg) &&
                MRI->use_nodbg_empty(OldVLReg)) {
              VLOpDef->eraseFromParent();
              if (LIS)
                LIS->removeInterval(OldVLReg);
            }
          }
          MI.setDesc(NextMI->getDesc());
        }
        MI.getOperand(2).setImm(NextMI->getOperand(2).getImm());
        ToDelete.push_back(NextMI);
        // fallthrough
      }
    }
    NextMI = &MI;
    Used = getDemanded(MI, ST);
  }

  NumCoalescedVSETVL += ToDelete.size();
  for (auto *MI : ToDelete) {
    if (LIS)
      LIS->RemoveMachineInstrFromMaps(*MI);
    MI->eraseFromParent();
  }
}

void RISCVInsertVSETVLI::insertReadVL(MachineBasicBlock &MBB) {
  for (auto I = MBB.begin(), E = MBB.end(); I != E;) {
    MachineInstr &MI = *I++;
    if (RISCV::isFaultFirstLoad(MI)) {
      Register VLOutput = MI.getOperand(1).getReg();
      assert(VLOutput.isVirtual());
      if (!MI.getOperand(1).isDead()) {
        auto ReadVLMI = BuildMI(MBB, I, MI.getDebugLoc(),
                                TII->get(RISCV::PseudoReadVL), VLOutput);
        // Move the LiveInterval's definition down to PseudoReadVL.
        if (LIS) {
          SlotIndex NewDefSI =
              LIS->InsertMachineInstrInMaps(*ReadVLMI).getRegSlot();
          LiveInterval &DefLI = LIS->getInterval(VLOutput);
          VNInfo *DefVNI = DefLI.getVNInfoAt(DefLI.beginIndex());
          DefLI.removeSegment(DefLI.beginIndex(), NewDefSI);
          DefVNI->def = NewDefSI;
        }
      }
      // We don't use the vl output of the VLEFF/VLSEGFF anymore.
      MI.getOperand(1).setReg(RISCV::X0);
    }
  }
}

bool RISCVInsertVSETVLI::runOnMachineFunction(MachineFunction &MF) {
  // Skip if the vector extension is not enabled.
  ST = &MF.getSubtarget<RISCVSubtarget>();
  if (!ST->hasVInstructions())
    return false;

  LLVM_DEBUG(dbgs() << "Entering InsertVSETVLI for " << MF.getName() << "\n");

  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  LIS = getAnalysisIfAvailable<LiveIntervals>();

  assert(BlockInfo.empty() && "Expect empty block infos");
  BlockInfo.resize(MF.getNumBlockIDs());

  bool HaveVectorOp = false;

  // Phase 1 - determine how VL/VTYPE are affected by the each block.
  for (const MachineBasicBlock &MBB : MF) {
    VSETVLIInfo TmpStatus;
    HaveVectorOp |= computeVLVTYPEChanges(MBB, TmpStatus);
    // Initial exit state is whatever change we found in the block.
    BlockData &BBInfo = BlockInfo[MBB.getNumber()];
    BBInfo.Exit = TmpStatus;
    LLVM_DEBUG(dbgs() << "Initial exit state of " << printMBBReference(MBB)
                      << " is " << BBInfo.Exit << "\n");

  }

  // If we didn't find any instructions that need VSETVLI, we're done.
  if (!HaveVectorOp) {
    BlockInfo.clear();
    return false;
  }

  // Phase 2 - determine the exit VL/VTYPE from each block. We add all
  // blocks to the list here, but will also add any that need to be revisited
  // during Phase 2 processing.
  for (const MachineBasicBlock &MBB : MF) {
    WorkList.push(&MBB);
    BlockInfo[MBB.getNumber()].InQueue = true;
  }
  while (!WorkList.empty()) {
    const MachineBasicBlock &MBB = *WorkList.front();
    WorkList.pop();
    computeIncomingVLVTYPE(MBB);
  }

  // Perform partial redundancy elimination of vsetvli transitions.
  for (MachineBasicBlock &MBB : MF)
    doPRE(MBB);

  // Phase 3 - add any vsetvli instructions needed in the block. Use the
  // Phase 2 information to avoid adding vsetvlis before the first vector
  // instruction in the block if the VL/VTYPE is satisfied by its
  // predecessors.
  for (MachineBasicBlock &MBB : MF)
    emitVSETVLIs(MBB);

  // Now that all vsetvlis are explicit, go through and do block local
  // DSE and peephole based demanded fields based transforms.  Note that
  // this *must* be done outside the main dataflow so long as we allow
  // any cross block analysis within the dataflow.  We can't have both
  // demanded fields based mutation and non-local analysis in the
  // dataflow at the same time without introducing inconsistencies.
  for (MachineBasicBlock &MBB : MF)
    coalesceVSETVLIs(MBB);

  // Insert PseudoReadVL after VLEFF/VLSEGFF and replace it with the vl output
  // of VLEFF/VLSEGFF.
  for (MachineBasicBlock &MBB : MF)
    insertReadVL(MBB);

  BlockInfo.clear();
  return HaveVectorOp;
}

/// Returns an instance of the Insert VSETVLI pass.
FunctionPass *llvm::createRISCVInsertVSETVLIPass() {
  return new RISCVInsertVSETVLI();
}
