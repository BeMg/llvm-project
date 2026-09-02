; RUN: llc -mtriple=riscv64 -mattr=+zba -O3 < %s | FileCheck %s

; RISCVInstrInfo::getJumpTableIndex lets MachineBasicBlock::canSplitCriticalEdge
; recognize a jump-table block. Without it findJumpTableIndex always returns -1
; on RISC-V, canSplitCriticalEdge falls through to analyzeBranch, which fails on
; the indirect PseudoBRIND, and no edge out of a jump table can ever be split.
;
; MachineSink's BreakPHIEdge path depends on that split: %v below is computed in
; the dispatch block but its only use is a PHI operand on the single %disp -> %m
; edge, and %m has another predecessor, so the edge is critical.
;
; This also covers the cross-block part of the walk: the jump-table address is
; loop invariant, so MachineLICM hoists the PseudoMovAddr that carries the
; %jump-table.0 operand out of the dispatch block entirely.

declare void @use(i32)

define void @sink_through_jt(ptr %p) {
; The dispatch block must hold nothing but the table lookup and the branch.
; CHECK-LABEL: sink_through_jt:
; CHECK:         sh2add
; CHECK-NEXT:    lw
; CHECK-NEXT:    jr
;
; ... and the sunk computation must still exist, in a block of its own.
; CHECK:         addiw {{.*}}, 7
entry:
  br label %disp

disp:
  %ip = phi ptr [ %p, %entry ], [ %ipn, %c0 ], [ %ipn, %c1 ], [ %ipn, %c2 ],
                [ %ipn, %c3 ], [ %ipn, %c4 ], [ %ipn, %c5 ], [ %ipn, %c6 ],
                [ %ipn, %c7 ], [ %ipn, %c8 ], [ %ipn, %c9 ], [ %ipn, %c10 ],
                [ %ipn, %c11 ], [ %ipn, %m ]
  %op = load i8, ptr %ip
  %ipn = getelementptr i8, ptr %ip, i64 1
  %x = load i32, ptr %ipn
  %v = add i32 %x, 7
  switch i8 %op, label %c0 [
    i8 1, label %m
    i8 2, label %other
    i8 3, label %c1
    i8 4, label %c2
    i8 5, label %c3
    i8 6, label %c4
    i8 7, label %c5
    i8 8, label %c6
    i8 9, label %c7
    i8 10, label %c8
    i8 11, label %c9
    i8 12, label %c10
    i8 13, label %c11
  ]

; Second predecessor of %m, so that %disp -> %m is a critical edge.
other:
  call void @use(i32 55)
  br label %m

m:
  %phi = phi i32 [ %v, %disp ], [ 0, %other ]
  call void @use(i32 %phi)
  br label %disp

c0:
  call void @use(i32 100)
  br label %disp
c1:
  call void @use(i32 101)
  br label %disp
c2:
  call void @use(i32 102)
  br label %disp
c3:
  call void @use(i32 103)
  br label %disp
c4:
  call void @use(i32 104)
  br label %disp
c5:
  call void @use(i32 105)
  br label %disp
c6:
  call void @use(i32 106)
  br label %disp
c7:
  call void @use(i32 107)
  br label %disp
c8:
  call void @use(i32 108)
  br label %disp
c9:
  call void @use(i32 109)
  br label %disp
c10:
  call void @use(i32 110)
  br label %disp
c11:
  call void @use(i32 111)
  br label %disp
}
