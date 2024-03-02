//===-- RISCVMachineScheduler.h - Create RISC-V Own Scheduler --*- C++ -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVMACHINESCHEDULER_H
#define LLVM_LIB_TARGET_RISCV_RISCVMACHINESCHEDULER_H

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineScheduler.h"

using namespace llvm;

// FIXME: Should reuse the VSETVLInfo from VSETVL insertion pass

// For the SSA form, we could just use the getVRegDef to take Reaching
// definition. For the non-SSA, we retrieve reaching definition for specific
// register from LiveInterval/VNInfo.
template <typename T>
static T *getReachingDefMI(Register Reg, T *MI, const MachineRegisterInfo *MRI,
                           const LiveIntervals *LIS) {
  if (MRI->isSSA())
    return MRI->getVRegDef(Reg);

  return nullptr;
}

/// Which subfields of VL or VTYPE have values we need to preserve?
struct DemandedFields {
  // Some unknown property of VL is used.  If demanded, must preserve entire
  // value.
  bool VLAny = false;
  // Only zero vs non-zero is used. If demanded, can change non-zero values.
  bool VLZeroness = false;
  // What properties of SEW we need to preserve.
  enum : uint8_t {
    SEWEqual = 3,              // The exact value of SEW needs to be preserved.
    SEWGreaterThanOrEqual = 2, // SEW can be changed as long as it's greater
                               // than or equal to the original value.
    SEWGreaterThanOrEqualAndLessThan64 =
        1,      // SEW can be changed as long as it's greater
                // than or equal to the original value, but must be less
                // than 64.
    SEWNone = 0 // We don't need to preserve SEW at all.
  } SEW = SEWNone;
  bool LMUL = false;
  bool SEWLMULRatio = false;
  bool TailPolicy = false;
  bool MaskPolicy = false;

  // Return true if any part of VTYPE was used
  bool usedVTYPE() const {
    return SEW || LMUL || SEWLMULRatio || TailPolicy || MaskPolicy;
  }

  // Return true if any property of VL was used
  bool usedVL() { return VLAny || VLZeroness; }

  // Mark all VTYPE subfields and properties as demanded
  void demandVTYPE() {
    SEW = SEWEqual;
    LMUL = true;
    SEWLMULRatio = true;
    TailPolicy = true;
    MaskPolicy = true;
  }

  // Mark all VL properties as demanded
  void demandVL() {
    VLAny = true;
    VLZeroness = true;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  /// Implement operator<<.
  void print(raw_ostream &OS) const {
    OS << "{";
    OS << "VLAny=" << VLAny << ", ";
    OS << "VLZeroness=" << VLZeroness << ", ";
    OS << "SEW=";
    switch (SEW) {
    case SEWEqual:
      OS << "SEWEqual";
      break;
    case SEWGreaterThanOrEqual:
      OS << "SEWGreaterThanOrEqual";
      break;
    case SEWGreaterThanOrEqualAndLessThan64:
      OS << "SEWGreaterThanOrEqualAndLessThan64";
      break;
    case SEWNone:
      OS << "SEWNone";
      break;
    };
    OS << ", ";
    OS << "LMUL=" << LMUL << ", ";
    OS << "SEWLMULRatio=" << SEWLMULRatio << ", ";
    OS << "TailPolicy=" << TailPolicy << ", ";
    OS << "MaskPolicy=" << MaskPolicy;
    OS << "}";
  }
#endif
};

static bool isNonZeroLoadImmediate(MachineInstr &MI) {
  return MI.getOpcode() == RISCV::ADDI && MI.getOperand(1).isReg() &&
         MI.getOperand(2).isImm() && MI.getOperand(1).getReg() == RISCV::X0 &&
         MI.getOperand(2).getImm() != 0;
}

/// Return true if moving from CurVType to NewVType is
/// indistinguishable from the perspective of an instruction (or set
/// of instructions) which use only the Used subfields and properties.
static bool areCompatibleVTYPEs(uint64_t CurVType, uint64_t NewVType,
                                const DemandedFields &Used) {
  switch (Used.SEW) {
  case DemandedFields::SEWNone:
    break;
  case DemandedFields::SEWEqual:
    if (RISCVVType::getSEW(CurVType) != RISCVVType::getSEW(NewVType))
      return false;
    break;
  case DemandedFields::SEWGreaterThanOrEqual:
    if (RISCVVType::getSEW(NewVType) < RISCVVType::getSEW(CurVType))
      return false;
    break;
  case DemandedFields::SEWGreaterThanOrEqualAndLessThan64:
    if (RISCVVType::getSEW(NewVType) < RISCVVType::getSEW(CurVType) ||
        RISCVVType::getSEW(NewVType) >= 64)
      return false;
    break;
  }

  if (Used.LMUL &&
      RISCVVType::getVLMUL(CurVType) != RISCVVType::getVLMUL(NewVType))
    return false;

  if (Used.SEWLMULRatio) {
    auto Ratio1 = RISCVVType::getSEWLMULRatio(RISCVVType::getSEW(CurVType),
                                              RISCVVType::getVLMUL(CurVType));
    auto Ratio2 = RISCVVType::getSEWLMULRatio(RISCVVType::getSEW(NewVType),
                                              RISCVVType::getVLMUL(NewVType));
    if (Ratio1 != Ratio2)
      return false;
  }

  if (Used.TailPolicy && RISCVVType::isTailAgnostic(CurVType) !=
                             RISCVVType::isTailAgnostic(NewVType))
    return false;
  if (Used.MaskPolicy && RISCVVType::isMaskAgnostic(CurVType) !=
                             RISCVVType::isMaskAgnostic(NewVType))
    return false;
  return true;
}

class VSETVLIInfo {
  union {
    Register AVLReg;
    unsigned AVLImm;
  };

  enum : uint8_t {
    Uninitialized,
    AVLIsReg,
    AVLIsImm,
    Unknown,
  } State = Uninitialized;

  // Fields from VTYPE.
  RISCVII::VLMUL VLMul = RISCVII::LMUL_1;
  uint8_t SEW = 0;
  uint8_t TailAgnostic : 1;
  uint8_t MaskAgnostic : 1;
  uint8_t SEWLMULRatioOnly : 1;

public:
  VSETVLIInfo()
      : AVLImm(0), TailAgnostic(false), MaskAgnostic(false),
        SEWLMULRatioOnly(false) {}

  static VSETVLIInfo getUnknown() {
    VSETVLIInfo Info;
    Info.setUnknown();
    return Info;
  }

  bool isValid() const { return State != Uninitialized; }
  void setUnknown() { State = Unknown; }
  bool isUnknown() const { return State == Unknown; }

  void setAVLReg(Register Reg) {
    AVLReg = Reg;
    State = AVLIsReg;
  }

  void setAVLImm(unsigned Imm) {
    AVLImm = Imm;
    State = AVLIsImm;
  }

  bool hasAVLImm() const { return State == AVLIsImm; }
  bool hasAVLReg() const { return State == AVLIsReg; }
  Register getAVLReg() const {
    assert(hasAVLReg());
    return AVLReg;
  }
  unsigned getAVLImm() const {
    assert(hasAVLImm());
    return AVLImm;
  }

  void setAVL(VSETVLIInfo Info) {
    assert(Info.isValid());
    if (Info.isUnknown())
      setUnknown();
    else if (Info.hasAVLReg())
      setAVLReg(Info.getAVLReg());
    else {
      assert(Info.hasAVLImm());
      setAVLImm(Info.getAVLImm());
    }
  }

  unsigned getSEW() const { return SEW; }
  RISCVII::VLMUL getVLMUL() const { return VLMul; }
  bool getTailAgnostic() const { return TailAgnostic; }
  bool getMaskAgnostic() const { return MaskAgnostic; }

  bool hasNonZeroAVL(const MachineRegisterInfo &MRI,
                     const LiveIntervals *LIS) const {
    if (hasAVLImm())
      return getAVLImm() > 0;
    if (hasAVLReg()) {
      if (getAVLReg() == RISCV::X0)
        return true;
      if (MachineInstr *MI =
              getReachingDefMI(getAVLReg(), (MachineInstr *)nullptr, &MRI, LIS);
          MI && isNonZeroLoadImmediate(*MI))
        return true;
      return false;
    }
    return false;
  }

  bool hasEquallyZeroAVL(const VSETVLIInfo &Other,
                         const MachineRegisterInfo &MRI,
                         const LiveIntervals *LIS) const {
    if (hasSameAVL(Other))
      return true;
    return (hasNonZeroAVL(MRI, LIS) && Other.hasNonZeroAVL(MRI, LIS));
  }

  bool hasSameAVL(const VSETVLIInfo &Other) const {
    if (hasAVLReg() && Other.hasAVLReg())
      return getAVLReg() == Other.getAVLReg();

    if (hasAVLImm() && Other.hasAVLImm())
      return getAVLImm() == Other.getAVLImm();

    return false;
  }

  void setVTYPE(unsigned VType) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = RISCVVType::getVLMUL(VType);
    SEW = RISCVVType::getSEW(VType);
    TailAgnostic = RISCVVType::isTailAgnostic(VType);
    MaskAgnostic = RISCVVType::isMaskAgnostic(VType);
  }
  void setVTYPE(RISCVII::VLMUL L, unsigned S, bool TA, bool MA) {
    assert(isValid() && !isUnknown() &&
           "Can't set VTYPE for uninitialized or unknown");
    VLMul = L;
    SEW = S;
    TailAgnostic = TA;
    MaskAgnostic = MA;
  }

  void setVLMul(RISCVII::VLMUL VLMul) { this->VLMul = VLMul; }

  unsigned encodeVTYPE() const {
    assert(isValid() && !isUnknown() && !SEWLMULRatioOnly &&
           "Can't encode VTYPE for uninitialized or unknown");
    return RISCVVType::encodeVTYPE(VLMul, SEW, TailAgnostic, MaskAgnostic);
  }

  bool hasSEWLMULRatioOnly() const { return SEWLMULRatioOnly; }

  bool hasSameVTYPE(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    assert(!SEWLMULRatioOnly && !Other.SEWLMULRatioOnly &&
           "Can't compare when only LMUL/SEW ratio is valid.");
    return std::tie(VLMul, SEW, TailAgnostic, MaskAgnostic) ==
           std::tie(Other.VLMul, Other.SEW, Other.TailAgnostic,
                    Other.MaskAgnostic);
  }

  unsigned getSEWLMULRatio() const {
    assert(isValid() && !isUnknown() &&
           "Can't use VTYPE for uninitialized or unknown");
    return RISCVVType::getSEWLMULRatio(SEW, VLMul);
  }

  // Check if the VTYPE for these two VSETVLIInfos produce the same VLMAX.
  // Note that having the same VLMAX ensures that both share the same
  // function from AVL to VL; that is, they must produce the same VL value
  // for any given AVL value.
  bool hasSameVLMAX(const VSETVLIInfo &Other) const {
    assert(isValid() && Other.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!isUnknown() && !Other.isUnknown() &&
           "Can't compare VTYPE in unknown state");
    return getSEWLMULRatio() == Other.getSEWLMULRatio();
  }

  bool hasCompatibleVTYPE(const DemandedFields &Used,
                          const VSETVLIInfo &Require) const {
    return areCompatibleVTYPEs(Require.encodeVTYPE(), encodeVTYPE(), Used);
  }

  // Determine whether the vector instructions requirements represented by
  // Require are compatible with the previous vsetvli instruction represented
  // by this.  MI is the instruction whose requirements we're considering.
  bool isCompatible(const DemandedFields &Used, const VSETVLIInfo &Require,
                    const MachineRegisterInfo &MRI,
                    const LiveIntervals *LIS) const {
    assert(isValid() && Require.isValid() &&
           "Can't compare invalid VSETVLIInfos");
    assert(!Require.SEWLMULRatioOnly &&
           "Expected a valid VTYPE for instruction!");
    // Nothing is compatible with Unknown.
    if (isUnknown() || Require.isUnknown())
      return false;

    // If only our VLMAX ratio is valid, then this isn't compatible.
    if (SEWLMULRatioOnly)
      return false;

    if (Used.VLAny && !hasSameAVL(Require))
      return false;

    if (Used.VLZeroness && !hasEquallyZeroAVL(Require, MRI, LIS))
      return false;

    return hasCompatibleVTYPE(Used, Require);
  }

  bool operator==(const VSETVLIInfo &Other) const {
    // Uninitialized is only equal to another Uninitialized.
    if (!isValid())
      return !Other.isValid();
    if (!Other.isValid())
      return !isValid();

    // Unknown is only equal to another Unknown.
    if (isUnknown())
      return Other.isUnknown();
    if (Other.isUnknown())
      return isUnknown();

    if (!hasSameAVL(Other))
      return false;

    // If the SEWLMULRatioOnly bits are different, then they aren't equal.
    if (SEWLMULRatioOnly != Other.SEWLMULRatioOnly)
      return false;

    // If only the VLMAX is valid, check that it is the same.
    if (SEWLMULRatioOnly)
      return hasSameVLMAX(Other);

    // If the full VTYPE is valid, check that it is the same.
    return hasSameVTYPE(Other);
  }

  bool operator!=(const VSETVLIInfo &Other) const { return !(*this == Other); }

  // Calculate the VSETVLIInfo visible to a block assuming this and Other are
  // both predecessors.
  VSETVLIInfo intersect(const VSETVLIInfo &Other) const {
    // If the new value isn't valid, ignore it.
    if (!Other.isValid())
      return *this;

    // If this value isn't valid, this must be the first predecessor, use it.
    if (!isValid())
      return Other;

    // If either is unknown, the result is unknown.
    if (isUnknown() || Other.isUnknown())
      return VSETVLIInfo::getUnknown();

    // If we have an exact, match return this.
    if (*this == Other)
      return *this;

    // Not an exact match, but maybe the AVL and VLMAX are the same. If so,
    // return an SEW/LMUL ratio only value.
    if (hasSameAVL(Other) && hasSameVLMAX(Other)) {
      VSETVLIInfo MergeInfo = *this;
      MergeInfo.SEWLMULRatioOnly = true;
      return MergeInfo;
    }

    // Otherwise the result is unknown.
    return VSETVLIInfo::getUnknown();
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// Support for debugging, callable in GDB: V->dump()
  LLVM_DUMP_METHOD void dump() const {
    print(dbgs());
    dbgs() << "\n";
  }

  /// Implement operator<<.
  /// @{
  void print(raw_ostream &OS) const {
    OS << "{";
    if (!isValid())
      OS << "Uninitialized";
    if (isUnknown())
      OS << "unknown";
    if (hasAVLReg())
      OS << "AVLReg=" << (unsigned)AVLReg;
    if (hasAVLImm())
      OS << "AVLImm=" << (unsigned)AVLImm;
    OS << ", " << "VLMul=" << (unsigned)VLMul << ", " << "SEW=" << (unsigned)SEW
       << ", " << "TailAgnostic=" << (bool)TailAgnostic << ", "
       << "MaskAgnostic=" << (bool)MaskAgnostic << ", "
       << "SEWLMULRatioOnly=" << (bool)SEWLMULRatioOnly << "}";
  }
#endif
};

static unsigned getSEWOpNum(const MachineInstr &MI) {
  return RISCVII::getSEWOpNum(MI.getDesc());
}

static unsigned getVLOpNum(const MachineInstr &MI) {
  return RISCVII::getVLOpNum(MI.getDesc());
}

static unsigned computeVLMAX(unsigned VLEN, unsigned SEW,
                             RISCVII::VLMUL VLMul) {
  auto [LMul, Fractional] = RISCVVType::decodeVLMUL(VLMul);
  if (Fractional)
    VLEN = VLEN / LMul;
  else
    VLEN = VLEN * LMul;
  return VLEN / SEW;
}

static bool isScalarExtractInstr(const MachineInstr &MI) {
  switch (RISCV::getRVVMCOpcode(MI.getOpcode())) {
  default:
    return false;
  case RISCV::VMV_X_S:
  case RISCV::VFMV_F_S:
    return true;
  }
}

/// Get the EEW for a load or store instruction.  Return std::nullopt if MI is
/// not a load or store which ignores SEW.
static std::optional<unsigned> getEEWForLoadStore(const MachineInstr &MI) {
  switch (RISCV::getRVVMCOpcode(MI.getOpcode())) {
  default:
    return std::nullopt;
  case RISCV::VLE8_V:
  case RISCV::VLSE8_V:
  case RISCV::VSE8_V:
  case RISCV::VSSE8_V:
    return 8;
  case RISCV::VLE16_V:
  case RISCV::VLSE16_V:
  case RISCV::VSE16_V:
  case RISCV::VSSE16_V:
    return 16;
  case RISCV::VLE32_V:
  case RISCV::VLSE32_V:
  case RISCV::VSE32_V:
  case RISCV::VSSE32_V:
    return 32;
  case RISCV::VLE64_V:
  case RISCV::VLSE64_V:
  case RISCV::VSE64_V:
  case RISCV::VSSE64_V:
    return 64;
  }
}

/// Return true if the inactive elements in the result are entirely undefined.
/// Note that this is different from "agnostic" as defined by the vector
/// specification.  Agnostic requires each lane to either be undisturbed, or
/// take the value -1; no other value is allowed.
static bool hasUndefinedMergeOp(const MachineInstr &MI,
                                const MachineRegisterInfo &MRI,
                                const LiveIntervals *LIS) {

  unsigned UseOpIdx;
  if (!MI.isRegTiedToUseOperand(0, &UseOpIdx))
    // If there is no passthrough operand, then the pass through
    // lanes are undefined.
    return true;

  // If the tied operand is NoReg, an IMPLICIT_DEF, or a REG_SEQEUENCE whose
  // operands are solely IMPLICIT_DEFS, then the pass through lanes are
  // undefined.
  const MachineOperand &UseMO = MI.getOperand(UseOpIdx);
  if (UseMO.getReg() == RISCV::NoRegister)
    return true;

  if (!MRI.isSSA() && UseMO.isUndef())
    return true;

  if (const MachineInstr *UseMI =
          getReachingDefMI(UseMO.getReg(), &MI, &MRI, LIS)) {
    if (UseMI->isImplicitDef())
      return true;

    if (UseMI->isRegSequence()) {
      for (unsigned i = 1, e = UseMI->getNumOperands(); i < e; i += 2) {
        const MachineInstr *SourceMI =
            getReachingDefMI(UseMI->getOperand(i).getReg(), UseMI, &MRI, LIS);
        if (!SourceMI || !SourceMI->isImplicitDef())
          return false;
      }
      return true;
    }
  }
  return false;
}

static bool isVectorConfigInstr(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::PseudoVSETVLI ||
         MI.getOpcode() == RISCV::PseudoVSETVLIX0 ||
         MI.getOpcode() == RISCV::PseudoVSETIVLI;
}

// Return a VSETVLIInfo representing the changes made by this VSETVLI or
// VSETIVLI instruction.
static VSETVLIInfo getInfoForVSETVLI(const MachineInstr &MI) {
  VSETVLIInfo NewInfo;
  if (MI.getOpcode() == RISCV::PseudoVSETIVLI) {
    NewInfo.setAVLImm(MI.getOperand(1).getImm());
  } else {
    assert(MI.getOpcode() == RISCV::PseudoVSETVLI ||
           MI.getOpcode() == RISCV::PseudoVSETVLIX0);
    Register AVLReg = MI.getOperand(1).getReg();
    assert((AVLReg != RISCV::X0 || MI.getOperand(0).getReg() != RISCV::X0) &&
           "Can't handle X0, X0 vsetvli yet");
    NewInfo.setAVLReg(AVLReg);
  }
  NewInfo.setVTYPE(MI.getOperand(2).getImm());

  return NewInfo;
}

static VSETVLIInfo computeInfoForInstrImpl(const MachineInstr &MI,
                                           uint64_t TSFlags,
                                           const RISCVSubtarget &ST,
                                           const MachineRegisterInfo *MRI,
                                           const LiveIntervals *LIS) {
  VSETVLIInfo InstrInfo;

  bool TailAgnostic = true;
  bool MaskAgnostic = true;
  if (!hasUndefinedMergeOp(MI, *MRI, LIS)) {
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
        const unsigned VLMAX = computeVLMAX(ST.getRealMaxVLen(), SEW, VLMul);
        if (ST.getRealMinVLen() == ST.getRealMaxVLen() && VLMAX <= 31)
          InstrInfo.setAVLImm(VLMAX);
        else
          InstrInfo.setAVLReg(RISCV::X0);
      } else
        InstrInfo.setAVLImm(Imm);
    } else {
      InstrInfo.setAVLReg(VLOp.getReg());
    }
  } else {
    assert(isScalarExtractInstr(MI));
    InstrInfo.setAVLReg(RISCV::NoRegister);
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
  if (InstrInfo.hasAVLReg() && InstrInfo.getAVLReg().isVirtual()) {
    const MachineInstr *DefMI =
        getReachingDefMI(InstrInfo.getAVLReg(), &MI, MRI, LIS);
    if (DefMI && isVectorConfigInstr(*DefMI)) {
      VSETVLIInfo DefInstrInfo = getInfoForVSETVLI(*DefMI);
      if (DefInstrInfo.hasSameVLMAX(InstrInfo) &&
          (DefInstrInfo.hasAVLImm() || DefInstrInfo.getAVLReg() == RISCV::X0)) {
        InstrInfo.setAVL(DefInstrInfo);
      }
    }
  }

  return InstrInfo;
}

namespace llvm {

/// A MachineSchedStrategy implementation for PowerPC pre RA scheduling.
class RISCVSchedStrategy : public GenericScheduler {
public:
  RISCVSchedStrategy(const MachineSchedContext *C) : GenericScheduler(C) {}

protected:
  void schedNode(SUnit *SU, bool IsTopNode) override;
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

private:
  VSETVLIInfo CurrVSETVLInfo = VSETVLIInfo::getUnknown();

  VSETVLIInfo computeInfoForInstr(MachineInstr *MI) const {
    uint64_t TSFlags = MI->getDesc().TSFlags;
    MachineFunction *MF = MI->getMF();
    MachineRegisterInfo &MRI = MF->getRegInfo();
    const RISCVSubtarget &ST = MF->getSubtarget<RISCVSubtarget>();
    return computeInfoForInstrImpl(*MI, TSFlags, ST, &MRI, nullptr);
  }

  bool isRVVInst(MachineInstr *MI) const {
    uint64_t TSFlags = MI->getDesc().TSFlags;
    if (!RISCVII::hasSEWOp(TSFlags))
      return false;
    return true;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVMACHINESCHEDULER_H
