//===--- RISCVVSETVLClusting.cpp - RISCV VSETVL Clustering  -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to cluster VSETVL
///       instruction.
//
//===----------------------------------------------------------------------===//

#include "RISCVVSETVLCluster.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"
#include <utility>

using namespace llvm;

namespace {

class VSETVLClustering : public ScheduleDAGMutation {
public:
  VSETVLClustering() = default;
  void apply(ScheduleDAGInstrs *DAG) override;
};

static void
collectCandidate(ScheduleDAGInstrs *DAG,
                 SmallVector<std::pair<SUnit *, VSETVLIInfo>> &Candidate) {
  for (SUnit &SU : DAG->SUnits) {
    if (SU.isInstr()) {
      MachineInstr *MI = SU.getInstr();
      const RISCVSubtarget &STI = MI->getMF()->getSubtarget<RISCVSubtarget>();
      uint64_t TSFlags = MI->getDesc().TSFlags;
      if (!RISCVII::hasSEWOp(TSFlags))
        continue;

      VSETVLIInfo Info = computeInfoForInstr(*MI, &STI, nullptr);
      Candidate.push_back(std::make_pair(&SU, Info));
    }
  }
}

void VSETVLClustering::apply(ScheduleDAGInstrs *DAG) {
  SmallVector<std::pair<SUnit *, VSETVLIInfo>> Candidate;
  collectCandidate(DAG, Candidate);

  SmallVector<SmallVector<std::pair<SUnit *, VSETVLIInfo>>> VSETVLGroups;

  for (auto &Tmp : Candidate) {
    SUnit *SU = Tmp.first;
    VSETVLIInfo Info = Tmp.second;
    bool Found = false;
    for (auto &Group : VSETVLGroups) {
      if (Group[0].second == Info &&
          all_of(Group, [SU, DAG](std::pair<SUnit *, VSETVLIInfo> Member) {
            return !DAG->IsReachable(SU, Member.first) &&
                   !DAG->IsReachable(Member.first, SU);
          })) {
        Found = true;
        Group.push_back(Tmp);
        break;
      }
    }

    if (!Found) {
      SmallVector<std::pair<SUnit *, VSETVLIInfo>> NewGroup;
      NewGroup.push_back(Tmp);
      VSETVLGroups.push_back(NewGroup);
    }
  }

  for (auto &Group : VSETVLGroups) {
    if (Group.size() < 2)
      continue;

    SUnit *CurrSU = Group[0].first;
    for (auto *I = Group.begin() + 1; I != Group.end(); I++) {
      DAG->addEdge(CurrSU, SDep(I->first, SDep::Cluster));
    }
  }
}

} // end namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createRISCVVSETVLClusteringDAGMutation() {
  return std::make_unique<VSETVLClustering>();
}

} // end namespace llvm
