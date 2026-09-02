; RUN: opt -passes=reassociate -S < %s | FileCheck %s
; RUN: opt -passes=reassociate -reassociate-max-negate-relocation-freq-ratio=0 \
; RUN:     -S < %s | FileCheck %s --check-prefix=NOCHECK

; When NegateValue breaks up a subtract it looks for an existing negation of the
; operand and relocates that one negation to just after the operand's def, so
; every user can share it. If the operand is defined in a block that is much
; hotter than the blocks that actually need the negation -- the canonical case
; being the shared dispatch block of a large switch -- sharing costs far more
; than it saves.

declare void @use(i32)

; %oparg is defined in the dispatch block, which runs once per case, so it is
; roughly 12x hotter than the two blocks that need a negation of it. The
; negations must stay in those blocks.
define void @high_fanout(ptr %ip0, i32 %k) {
; CHECK-LABEL: @high_fanout(
; CHECK:       disp:
; CHECK-NOT:     sub i32 0, %oparg
; CHECK:         switch i8
;
; With the check disabled, the old behaviour returns: one negation is hoisted
; into the dispatch block and shared.
; NOCHECK-LABEL: @high_fanout(
; NOCHECK:       disp:
; NOCHECK:         sub i32 0, %oparg
; NOCHECK:         switch i8
entry:
  br label %disp

disp:
  %ip = phi ptr [ %ip0, %entry ], [ %ipn, %c0 ], [ %ipn, %c1 ], [ %ipn, %c2 ],
                [ %ipn, %c3 ], [ %ipn, %c4 ], [ %ipn, %c5 ], [ %ipn, %c6 ],
                [ %ipn, %c7 ], [ %ipn, %c8 ], [ %ipn, %c9 ], [ %ipn, %c10 ],
                [ %ipn, %c11 ]
  %op = load i8, ptr %ip
  %argb = load i8, ptr %ip
  %oparg = zext i8 %argb to i32
  %ipn = getelementptr i8, ptr %ip, i64 2
  switch i8 %op, label %c0 [
    i8 1, label %c1
    i8 2, label %c2
    i8 3, label %c3
    i8 4, label %c4
    i8 5, label %c5
    i8 6, label %c6
    i8 7, label %c7
    i8 8, label %c8
    i8 9, label %c9
    i8 10, label %c10
    i8 11, label %c11
  ]

; A reassociable subtract of %oparg: this is what calls NegateValue.
c0:
  %a = add i32 %k, 7
  %s = sub i32 %a, %oparg
  call void @use(i32 %s)
  br label %disp

; Two existing negations, the relocation candidates.
c1:
  %n1 = sub i32 0, %oparg
  call void @use(i32 %n1)
  br label %disp
c2:
  %n2 = sub i32 0, %oparg
  call void @use(i32 %n2)
  br label %disp

c3:
  call void @use(i32 3)
  br label %disp
c4:
  call void @use(i32 4)
  br label %disp
c5:
  call void @use(i32 5)
  br label %disp
c6:
  call void @use(i32 6)
  br label %disp
c7:
  call void @use(i32 7)
  br label %disp
c8:
  call void @use(i32 8)
  br label %disp
c9:
  call void @use(i32 9)
  br label %disp
c10:
  call void @use(i32 10)
  br label %disp
c11:
  call void @use(i32 11)
  br label %disp
}

; Same shape, but only three cases, so the dispatch block is not meaningfully
; hotter than the blocks that need the negation. Sharing is still worth it and
; the existing behaviour must be preserved.
define void @small_switch(ptr %ip0, i32 %k) {
; CHECK-LABEL: @small_switch(
; CHECK:       disp:
; CHECK:         sub i32 0, %oparg
; CHECK:         switch i8
entry:
  br label %disp

disp:
  %ip = phi ptr [ %ip0, %entry ], [ %ipn, %c0 ], [ %ipn, %c1 ], [ %ipn, %c2 ]
  %op = load i8, ptr %ip
  %argb = load i8, ptr %ip
  %oparg = zext i8 %argb to i32
  %ipn = getelementptr i8, ptr %ip, i64 2
  switch i8 %op, label %c0 [
    i8 1, label %c1
    i8 2, label %c2
  ]

c0:
  %a = add i32 %k, 7
  %s = sub i32 %a, %oparg
  call void @use(i32 %s)
  br label %disp

c1:
  %n1 = sub i32 0, %oparg
  call void @use(i32 %n1)
  br label %disp
c2:
  %n2 = sub i32 0, %oparg
  call void @use(i32 %n2)
  br label %disp
}
