# Перенос runtests Cranelift

Проверены 378 файлов, включая вложенный каталог x64, и 2 482 определения функций. Исходная ревизия закреплена для сверки входов и ожидаемых результатов. [Wasmtime v45.0.3, commit 84f1cbe](https://github.com/bytecodealliance/wasmtime/tree/84f1cbebd7e68ad6385f3438b12cf8b8f10d2f08/cranelift/filetests/filetests/runtests)

Перенос охватывает 263 исходные функции из 46 файлов. Для них записаны 268 программ Lancy и 926 директив `@run`. Разница в числе программ возникает при проверке компонентов множественного результата и дальнего перехода. Ссылки в таблице открывают перенесённые тесты. Имена функций указаны в самих файлах.

Проверки используют векторы Cranelift. Сравнения Lancy возвращают маску 0/−1. Там, где CLIF потребляет 0/1, тест выделяет младший бит. Беззнаковые сравнения выражены через смену знакового бита, select — через параметры блока. У отмеченных побитовых и арифметических выражений i64 перенесено младшее слово. Лицензия перенесённых материалов — [Apache-2.0 WITH LLVM-exception](LICENSE).

| Исходный файл | Определений | Исходных функций в переносе | Особенности переноса и причины пропуска остальных вариантов |
| --- | ---: | ---: | --- |
| aarch64-apple-stack-return-abi.clif | 1 | 0 | SIMD |
| [alias.clif](alias.lancy) | 3 | 3 |  |
| arithmetic-extends.clif | 20 | 0 | другая разрядность или ABI, другой тип результата |
| [arithmetic.clif](arithmetic.lancy) | 83 | 53 | другая разрядность или ABI, инструкция udiv |
| atomic-128.clif | 14 | 0 | атомарные операции |
| atomic-cas-little.clif | 2 | 0 | атомарные операции |
| atomic-cas-subword-big.clif | 2 | 0 | атомарные операции |
| atomic-cas-subword-little.clif | 2 | 0 | атомарные операции |
| atomic-cas.clif | 2 | 0 | атомарные операции |
| atomic-load-store.clif | 5 | 0 | атомарные операции |
| atomic-rmw-little.clif | 44 | 0 | атомарные операции |
| atomic-rmw-subword-big.clif | 44 | 0 | атомарные операции |
| atomic-rmw-subword-little.clif | 44 | 0 | атомарные операции |
| bb-padding.clif | 2 | 0 | SIMD |
| [bitcast-same-type.clif](bitcast-same-type.lancy) | 7 | 1 | Тождественное преобразование представлено copy. другая разрядность или ABI, i128, вещественные типы |
| bitcast.clif | 4 | 0 | вещественные типы |
| [bitops.clif](bitops.lancy) | 65 | 58 | другая разрядность или ABI, вспомогательные тела без своего вектора |
| bitrev.clif | 4 | 0 | другая разрядность или ABI, инструкция bitrev |
| bitselect-float.clif | 2 | 0 | вещественные типы |
| [bitselect.clif](bitselect.lancy) | 5 | 1 | другая разрядность или ABI |
| [bmask.clif](bmask.lancy) | 18 | 1 | другая разрядность или ABI, другой тип результата, узкая маска |
| [bnot.clif](bnot.lancy) | 7 | 1 | другая разрядность или ABI |
| [br.clif](br.lancy) | 12 | 4 | другая разрядность или ABI |
| [br_table.clif](br_table.lancy) | 5 | 3 | Диспетчеризация и параметры рёбер выражены через br. раскладка таблицы переходов RISC-V, инструкция rotr |
| [brif.clif](brif.lancy) | 12 | 8 | Переполнение i8 в удвоении перенесено на границу i32. другая разрядность или ABI, несколько результатов |
| bswap.clif | 5 | 0 | другая разрядность или ABI, инструкция bswap, несколько результатов, вещественные типы |
| [call.clif](call.lancy) | 9 | 1 | Прямой внутренний вызов адаптирован к i32. вспомогательные тела без своего вектора, другая разрядность или ABI, вещественные типы |
| call_indirect.clif | 3 | 0 | func_addr и call_indirect |
| call_libcall.clif | 2 | 0 | вещественные типы |
| ceil.clif | 4 | 0 | вещественные типы |
| cls.clif | 5 | 0 | другая разрядность или ABI, инструкция cls, i128 |
| clz.clif | 4 | 0 | другая разрядность или ABI, инструкция clz |
| [const.clif](const.lancy) | 20 | 5 | узкие константы, другой тип результата |
| conversion-i8-i16.clif | 16 | 0 | вещественные типы |
| conversion.clif | 16 | 0 | вещественные типы |
| ctz.clif | 4 | 0 | другая разрядность или ABI, инструкция ctz |
| [div-checks.clif](div-checks.lancy) | 16 | 2 | Для MIN % −1 ожидается ошибка Lancy. другая разрядность или ABI, инструкция urem |
| dynamic-simd-arithmetic.clif | 23 | 0 | SIMD |
| dynamic-simd-narrow-widen.clif | 15 | 0 | SIMD |
| extend.clif | 26 | 0 | другая разрядность или ABI, другой тип результата |
| f128-bitcast.clif | 2 | 0 | вещественные типы |
| f128-select.clif | 4 | 0 | вещественные типы |
| f128const.clif | 3 | 0 | вещественные типы |
| f16-arithmetic.clif | 8 | 0 | вещественные типы |
| f16-bitcast.clif | 2 | 0 | вещественные типы |
| f16-fma.clif | 4 | 0 | вещественные типы |
| f16-memory.clif | 3 | 0 | вещественные типы |
| f16-select.clif | 4 | 0 | вещественные типы |
| f16const.clif | 3 | 0 | вещественные типы |
| f32const.clif | 2 | 0 | вещественные типы |
| f64const.clif | 2 | 0 | вещественные типы |
| fabs.clif | 2 | 0 | вещественные типы |
| fadd.clif | 4 | 0 | вещественные типы |
| fcmp-eq.clif | 2 | 0 | вещественные типы |
| fcmp-ge.clif | 2 | 0 | вещественные типы |
| fcmp-gt.clif | 2 | 0 | вещественные типы |
| fcmp-le.clif | 2 | 0 | вещественные типы |
| fcmp-lt.clif | 2 | 0 | вещественные типы |
| fcmp-ne.clif | 2 | 0 | вещественные типы |
| fcmp-one.clif | 2 | 0 | вещественные типы |
| fcmp-ord.clif | 2 | 0 | вещественные типы |
| fcmp-ueq.clif | 2 | 0 | вещественные типы |
| fcmp-uge.clif | 2 | 0 | вещественные типы |
| fcmp-ugt.clif | 2 | 0 | вещественные типы |
| fcmp-ule.clif | 2 | 0 | вещественные типы |
| fcmp-ult.clif | 2 | 0 | вещественные типы |
| fcmp-uno.clif | 2 | 0 | вещественные типы |
| fcopysign.clif | 2 | 0 | вещественные типы |
| fcvt-from-int.clif | 16 | 0 | вещественные типы |
| fcvt-sat-small.clif | 8 | 0 | вещественные типы |
| fdemote.clif | 2 | 0 | вещественные типы |
| fdemote_32.clif | 1 | 0 | вещественные типы |
| fdemote_64.clif | 1 | 0 | вещественные типы |
| fdiv.clif | 4 | 0 | вещественные типы |
| fence.clif | 1 | 0 | атомарная память |
| [fibonacci.clif](fibonacci.lancy) | 2 | 1 | Итеративная версия. рекурсивный вызов |
| float-bitops.clif | 4 | 0 | вещественные типы |
| floor.clif | 4 | 0 | вещественные типы |
| fma-neg-chain.clif | 1 | 0 | SIMD |
| fma.clif | 11 | 0 | вещественные типы |
| fmax-pseudo.clif | 4 | 0 | вещественные типы |
| fmax-vector.clif | 2 | 0 | SIMD |
| fmax.clif | 4 | 0 | вещественные типы |
| fmin-pseudo.clif | 4 | 0 | вещественные типы |
| fmin-vector.clif | 2 | 0 | SIMD |
| fmin.clif | 4 | 0 | вещественные типы |
| fmul.clif | 4 | 0 | вещественные типы |
| fneg.clif | 2 | 0 | вещественные типы |
| [fold-bitops.clif](fold-bitops.lancy) | 22 | 22 |  |
| fp-bitops.clif | 8 | 0 | вещественные типы |
| fpromote.clif | 2 | 0 | вещественные типы |
| fpromote_32.clif | 1 | 0 | вещественные типы |
| fpromote_64.clif | 1 | 0 | вещественные типы |
| fsub.clif | 4 | 0 | вещественные типы |
| global_value.clif | 3 | 0 | другая разрядность или ABI, явные стековые слоты, вспомогательные тела без своего вектора |
| i128-arithmetic-extends.clif | 2 | 0 | i128 |
| i128-arithmetic.clif | 10 | 0 | i128 |
| i128-bandnot.clif | 1 | 0 | i128 |
| i128-bitcast.clif | 1 | 0 | i128 |
| i128-bitops-count.clif | 3 | 0 | i128 |
| i128-bitops.clif | 4 | 0 | i128 |
| i128-bitrev.clif | 4 | 0 | i128 |
| i128-bitselect.clif | 2 | 0 | i128 |
| i128-bmask.clif | 9 | 0 | i128 |
| i128-bnot.clif | 1 | 0 | i128 |
| i128-bornot.clif | 1 | 0 | i128 |
| i128-br.clif | 2 | 0 | i128 |
| i128-bswap.clif | 1 | 0 | i128 |
| i128-bxornot.clif | 1 | 0 | i128 |
| i128-call.clif | 2 | 0 | i128 |
| i128-concat-split.clif | 1 | 0 | несколько результатов |
| i128-conversion.clif | 4 | 0 | вещественные типы |
| i128-extend.clif | 8 | 0 | i128 |
| i128-iabs.clif | 1 | 0 | i128 |
| i128-icmp.clif | 12 | 0 | i128 |
| i128-ineg.clif | 1 | 0 | i128 |
| i128-ireduce.clif | 4 | 0 | i128 |
| i128-load-store.clif | 5 | 0 | i128 |
| i128-min-max.clif | 4 | 0 | i128 |
| i128-rotate.clif | 12 | 0 | i128 |
| i128-select-float.clif | 2 | 0 | вещественные типы |
| i128-select.clif | 4 | 0 | i128, вещественные типы |
| i128-shifts.clif | 27 | 0 | i128, другая разрядность или ABI |
| i128-srem.clif | 2 | 0 | i128 |
| i128-urem.clif | 2 | 0 | i128 |
| [iabs.clif](iabs.lancy) | 5 | 1 | другая разрядность или ABI |
| iaddcarry.clif | 8 | 0 | несколько результатов |
| [icmp-eq-imm.clif](icmp-eq-imm.lancy) | 8 | 2 | другая разрядность или ABI |
| [icmp-eq.clif](icmp-eq.lancy) | 8 | 2 | другая разрядность или ABI |
| [icmp-ne.clif](icmp-ne.lancy) | 8 | 2 | другая разрядность или ABI |
| [icmp-of-icmp.clif](icmp-of-icmp.lancy) | 4 | 4 |  |
| [icmp-sge.clif](icmp-sge.lancy) | 8 | 2 | другая разрядность или ABI |
| [icmp-sgt.clif](icmp-sgt.lancy) | 8 | 2 | другая разрядность или ABI |
| [icmp-sle.clif](icmp-sle.lancy) | 8 | 2 | другая разрядность или ABI |
| [icmp-slt.clif](icmp-slt.lancy) | 8 | 2 | другая разрядность или ABI |
| [icmp-uge.clif](icmp-uge.lancy) | 9 | 3 | другая разрядность или ABI |
| [icmp-ugt.clif](icmp-ugt.lancy) | 9 | 3 | другая разрядность или ABI |
| [icmp-ule.clif](icmp-ule.lancy) | 8 | 2 | другая разрядность или ABI |
| [icmp-ult.clif](icmp-ult.lancy) | 8 | 2 | другая разрядность или ABI |
| [icmp.clif](icmp.lancy) | 14 | 12 | другая разрядность или ABI |
| [ineg.clif](ineg.lancy) | 4 | 1 | другая разрядность или ABI |
| inline-probestack.clif | 2 | 0 | явные стековые слоты |
| [integer-minmax.clif](integer-minmax.lancy) | 10 | 3 | Компоненты результата проверяются отдельными программами. несколько результатов |
| ireduce.clif | 6 | 0 | другая разрядность или ABI, инструкция ireduce, другой тип результата |
| issue-10906.clif | 2 | 0 | SIMD |
| issue-12696.clif | 3 | 0 | SIMD |
| [issue-5498.clif](issue-5498.lancy) | 1 | 1 |  |
| issue-5690.clif | 1 | 0 | вещественные типы |
| issue-5992.clif | 4 | 0 | вещественные типы |
| issue-5993.clif | 2 | 0 | вещественные типы |
| issue-6581.clif | 1 | 0 | вещественные типы |
| issue-6582.clif | 3 | 0 | несколько результатов, вызов через специальный ABI |
| issue-6635.clif | 2 | 0 | вспомогательные тела без своего вектора, несколько результатов |
| issue-6640.clif | 1 | 0 | i128 |
| issue-6859.clif | 1 | 0 | вещественные типы |
| issue-6916.clif | 4 | 0 | SIMD |
| issue-6954.clif | 1 | 0 | SIMD |
| issue5497.clif | 1 | 0 | i128 |
| issue5523.clif | 1 | 0 | i128 |
| issue5524.clif | 1 | 0 | i128 |
| issue5525.clif | 1 | 0 | i128 |
| issue5526.clif | 1 | 0 | i128 |
| issue5528.clif | 2 | 0 | вещественные типы |
| issue5569.clif | 1 | 0 | вещественные типы |
| issue5714.clif | 1 | 0 | вещественные типы |
| [issue5839.clif](issue5839.lancy) | 1 | 1 |  |
| issue5884.clif | 1 | 0 | атомарная память |
| issue5901.clif | 1 | 0 | атомарная память |
| issue5952.clif | 1 | 0 | вещественные типы |
| isubborrow.clif | 8 | 0 | несколько результатов |
| [long-jump.clif](long-jump.lancy) | 1 | 1 | Дистанцию rel32 создаёт пропускаемый блок с живыми инструкциями. |
| nearest.clif | 4 | 0 | вещественные типы |
| [or-and-y-with-not-y.clif](or-and-y-with-not-y.lancy) | 2 | 2 |  |
| pinned-reg.clif | 1 | 0 | другая разрядность или ABI |
| popcnt-interpret.clif | 4 | 0 | SIMD |
| popcnt.clif | 8 | 0 | другая разрядность или ABI, инструкция popcnt |
| return-call-indirect.clif | 9 | 0 | косвенный хвостовой вызов |
| return-call-loop.clif | 7 | 0 | хвостовая рекурсия и регистр стека |
| return-call.clif | 12 | 0 | хвостовые вызовы |
| riscv64-vstate.clif | 2 | 0 | SIMD |
| riscv64_issue_4996.clif | 1 | 0 | i128 |
| rotl.clif | 19 | 0 | другая разрядность или ABI, инструкция rotl |
| rotr.clif | 21 | 0 | другая разрядность или ABI, инструкция rotr |
| s390x-lxa.clif | 2 | 0 | другая разрядность или ABI |
| s390x-veval.clif | 50 | 0 | SIMD |
| sadd_overflow.clif | 5 | 0 | i128, несколько результатов |
| scalar-to-vector-upper-bits-zero.clif | 2 | 0 | SIMD |
| sdiv-i128.clif | 1 | 0 | i128 |
| [sdiv.clif](sdiv.lancy) | 12 | 6 | другая разрядность или ABI |
| select-float.clif | 8 | 0 | вещественные типы |
| [select.clif](select.lancy) | 48 | 18 | вещественные типы, другая разрядность или ABI |
| selectif-spectre-guard.clif | 26 | 0 | другая разрядность или ABI, i128 |
| [shift-right-left.clif](shift-right-left.lancy) | 6 | 2 | другая разрядность или ABI |
| [shifts.clif](shifts.lancy) | 61 | 5 | другая разрядность или ABI, инструкция ushr |
| simd-arithmetic.clif | 11 | 0 | SIMD |
| simd-avg-round-small.clif | 5 | 0 | SIMD |
| simd-avg-round.clif | 2 | 0 | SIMD |
| simd-band-64bit.clif | 3 | 0 | SIMD |
| simd-band-splat.clif | 10 | 0 | SIMD |
| simd-band.clif | 6 | 0 | SIMD |
| simd-bandnot-64bit.clif | 3 | 0 | SIMD |
| simd-bitcast-128.clif | 4 | 0 | SIMD |
| simd-bitcast-64bit.clif | 4 | 0 | SIMD |
| simd-bitcast-aarch64.clif | 2 | 0 | SIMD |
| simd-bitcast.clif | 6 | 0 | SIMD |
| simd-bitselect-to-vselect.clif | 4 | 0 | SIMD |
| simd-bitselect.clif | 9 | 0 | SIMD |
| simd-bnot-64bit.clif | 3 | 0 | SIMD |
| simd-bnot.clif | 6 | 0 | SIMD |
| simd-bor-64bit.clif | 3 | 0 | SIMD |
| simd-bor-splat.clif | 10 | 0 | SIMD |
| simd-bor.clif | 6 | 0 | SIMD |
| simd-bornot-64bit.clif | 3 | 0 | SIMD |
| simd-bxor-64bit.clif | 3 | 0 | SIMD |
| simd-bxor-splat.clif | 10 | 0 | SIMD |
| simd-bxor.clif | 6 | 0 | SIMD |
| simd-ceil.clif | 2 | 0 | SIMD |
| simd-extractlane.clif | 5 | 0 | SIMD |
| simd-extractlane_32.clif | 6 | 0 | SIMD |
| simd-extractlane_64.clif | 6 | 0 | SIMD |
| simd-fabs.clif | 2 | 0 | SIMD |
| simd-fadd-splat.clif | 2 | 0 | SIMD |
| simd-fadd.clif | 2 | 0 | SIMD |
| simd-fcmp-eq.clif | 7 | 0 | SIMD |
| simd-fcmp-ge.clif | 7 | 0 | SIMD |
| simd-fcmp-gt.clif | 7 | 0 | SIMD |
| simd-fcmp-le.clif | 6 | 0 | SIMD |
| simd-fcmp-lt.clif | 7 | 0 | SIMD |
| simd-fcmp-ne.clif | 6 | 0 | SIMD |
| simd-fcmp-one.clif | 6 | 0 | SIMD |
| simd-fcmp-ord.clif | 6 | 0 | SIMD |
| simd-fcmp-ueq.clif | 6 | 0 | SIMD |
| simd-fcmp-uge.clif | 6 | 0 | SIMD |
| simd-fcmp-ugt.clif | 6 | 0 | SIMD |
| simd-fcmp-ule.clif | 6 | 0 | SIMD |
| simd-fcmp-ult.clif | 6 | 0 | SIMD |
| simd-fcmp-uno.clif | 7 | 0 | SIMD |
| simd-fcopysign-64bit.clif | 1 | 0 | SIMD |
| simd-fcopysign.clif | 2 | 0 | SIMD |
| simd-fcvt-from-sint.clif | 2 | 0 | SIMD |
| simd-fcvt-from-uint.clif | 2 | 0 | SIMD |
| simd-fcvt-to-sint-sat.clif | 1 | 0 | SIMD |
| simd-fcvt-to-uint-sat.clif | 2 | 0 | SIMD |
| simd-fdiv.clif | 2 | 0 | SIMD |
| simd-floor.clif | 2 | 0 | SIMD |
| simd-fma-64bit.clif | 2 | 0 | SIMD |
| simd-fma-neg.clif | 6 | 0 | SIMD |
| simd-fma.clif | 8 | 0 | SIMD |
| simd-fmax-fmin-nondeterministic-aarch64.clif | 2 | 0 | SIMD |
| simd-fmax-fmin-nondeterministic-riscv64.clif | 2 | 0 | SIMD |
| simd-fmax-fmin-nondeterministic-x86_64.clif | 2 | 0 | SIMD |
| simd-fmax-fmin.clif | 2 | 0 | SIMD |
| simd-fmin-max-pseudo.clif | 4 | 0 | SIMD |
| simd-fmul.clif | 2 | 0 | SIMD |
| simd-fneg.clif | 2 | 0 | SIMD |
| simd-fsub.clif | 2 | 0 | SIMD |
| simd-fvdemote.clif | 1 | 0 | SIMD |
| simd-fvpromote-low.clif | 1 | 0 | SIMD |
| simd-iabs.clif | 4 | 0 | SIMD |
| simd-iadd-small.clif | 3 | 0 | SIMD |
| simd-iadd-splat.clif | 14 | 0 | SIMD |
| simd-iadd-swiden-high.clif | 9 | 0 | SIMD |
| simd-iadd-swiden-low.clif | 9 | 0 | SIMD |
| simd-iadd-swiden-mix.clif | 6 | 0 | SIMD |
| simd-iadd-uwiden-high.clif | 9 | 0 | SIMD |
| simd-iadd-uwiden-low.clif | 9 | 0 | SIMD |
| simd-iadd-uwiden-mix.clif | 6 | 0 | SIMD |
| simd-iadd.clif | 4 | 0 | SIMD |
| simd-iaddpairwise-64bit.clif | 7 | 0 | SIMD |
| simd-iaddpairwise.clif | 7 | 0 | SIMD |
| simd-icmp-eq.clif | 10 | 0 | SIMD |
| simd-icmp-ne.clif | 10 | 0 | SIMD |
| simd-icmp-sge.clif | 9 | 0 | SIMD |
| simd-icmp-sgt.clif | 10 | 0 | SIMD |
| simd-icmp-sle.clif | 9 | 0 | SIMD |
| simd-icmp-slt.clif | 9 | 0 | SIMD |
| simd-icmp-uge.clif | 9 | 0 | SIMD |
| simd-icmp-ugt.clif | 9 | 0 | SIMD |
| simd-icmp-ule.clif | 9 | 0 | SIMD |
| simd-icmp-ult.clif | 10 | 0 | SIMD |
| simd-ifma.clif | 6 | 0 | SIMD |
| simd-imul-i8x16.clif | 1 | 0 | SIMD |
| simd-imul.clif | 3 | 0 | SIMD |
| simd-ineg.clif | 4 | 0 | SIMD |
| simd-insert-extract-lane.clif | 1 | 0 | SIMD |
| simd-insertlane.clif | 14 | 0 | SIMD |
| simd-insertlane_32.clif | 8 | 0 | SIMD |
| simd-insertlane_64.clif | 8 | 0 | SIMD |
| simd-ishl.clif | 9 | 0 | SIMD |
| simd-isub-splat.clif | 22 | 0 | SIMD |
| simd-isub-swiden-high.clif | 9 | 0 | SIMD |
| simd-isub-swiden-low.clif | 9 | 0 | SIMD |
| simd-isub-uwiden-high.clif | 9 | 0 | SIMD |
| simd-isub-uwiden-low.clif | 9 | 0 | SIMD |
| simd-isub.clif | 4 | 0 | SIMD |
| simd-lane-access.clif | 20 | 0 | SIMD |
| simd-logical.clif | 11 | 0 | SIMD |
| simd-make-vectors.clif | 3 | 0 | SIMD |
| simd-min-max.clif | 16 | 0 | SIMD |
| simd-nearest.clif | 2 | 0 | SIMD |
| simd-popcnt-large.clif | 3 | 0 | SIMD |
| simd-popcnt.clif | 1 | 0 | SIMD |
| simd-saddsat-aarch64.clif | 2 | 0 | SIMD |
| simd-saddsat.clif | 2 | 0 | SIMD |
| simd-scalartovector-aarch64.clif | 2 | 0 | SIMD |
| simd-scalartovector.clif | 6 | 0 | SIMD |
| simd-select.clif | 6 | 0 | SIMD |
| simd-shuffle.clif | 60 | 0 | SIMD |
| simd-small.clif | 19 | 0 | SIMD |
| simd-smulhi.clif | 4 | 0 | SIMD |
| simd-snarrow-aarch64.clif | 1 | 0 | SIMD |
| simd-snarrow.clif | 2 | 0 | SIMD |
| simd-splat.clif | 20 | 0 | SIMD |
| simd-sqmulroundsat-aarch64.clif | 2 | 0 | SIMD |
| simd-sqmulroundsat.clif | 2 | 0 | SIMD |
| simd-sqrt.clif | 2 | 0 | SIMD |
| simd-sshr.clif | 12 | 0 | SIMD |
| simd-ssubsat-aarch64.clif | 2 | 0 | SIMD |
| simd-ssubsat.clif | 2 | 0 | SIMD |
| simd-swidenhigh.clif | 6 | 0 | SIMD |
| simd-swidenlow.clif | 6 | 0 | SIMD |
| simd-swizzle.clif | 3 | 0 | SIMD |
| simd-trunc.clif | 2 | 0 | SIMD |
| simd-uaddsat-aarch64.clif | 2 | 0 | SIMD |
| simd-uaddsat.clif | 2 | 0 | SIMD |
| simd-umulhi.clif | 5 | 0 | другая разрядность или ABI, SIMD |
| simd-unarrow-aarch64.clif | 1 | 0 | SIMD |
| simd-unarrow.clif | 2 | 0 | SIMD |
| simd-ushr.clif | 9 | 0 | SIMD |
| simd-usubsat-aarch64.clif | 2 | 0 | SIMD |
| simd-usubsat.clif | 2 | 0 | SIMD |
| simd-uunarrow.clif | 3 | 0 | SIMD |
| simd-uwidenhigh.clif | 6 | 0 | SIMD |
| simd-uwidenlow.clif | 6 | 0 | SIMD |
| simd-valltrue-64bit.clif | 6 | 0 | SIMD |
| simd-valltrue.clif | 6 | 0 | SIMD |
| simd-vanytrue-64bit.clif | 6 | 0 | SIMD |
| simd-vanytrue.clif | 6 | 0 | SIMD |
| simd-vconst-64bit.clif | 5 | 0 | SIMD |
| simd-vconst-large.clif | 1 | 0 | SIMD |
| simd-vconst.clif | 6 | 0 | SIMD |
| simd-vhighbits-float.clif | 2 | 0 | SIMD |
| simd-vhighbits.clif | 4 | 0 | SIMD |
| simd-wideningpairwisedotproducts.clif | 1 | 0 | SIMD |
| simd_compare_zero.clif | 24 | 0 | SIMD |
| smul_overflow.clif | 4 | 0 | несколько результатов |
| smulhi-aarch64.clif | 1 | 0 | другая разрядность или ABI |
| smulhi.clif | 4 | 0 | другая разрядность или ABI, инструкция smulhi |
| [spill-reload.clif](spill-reload.lancy) | 1 | 1 | Проверяются spill и младшее слово результата. |
| sqrt.clif | 4 | 0 | вещественные типы |
| [srem.clif](srem.lancy) | 9 | 2 | другая разрядность или ABI |
| [srem_opts.clif](srem_opts.lancy) | 33 | 5 | другая разрядность или ABI, другой тип результата |
| ssub_overflow.clif | 5 | 0 | i128, несколько результатов |
| stack-addr-32.clif | 4 | 0 | явные стековые слоты |
| [stack-addr-64.clif](stack-addr-64.lancy) | 2 | 2 | Два способа адресации одного буфера. |
| [stack.clif](stack.lancy) | 7 | 4 | Выделение памяти задаёт вызывающая сторона. явные стековые слоты |
| tail-call-conv.clif | 8 | 0 | вещественные типы, вспомогательные тела без своего вектора, другой тип результата |
| tail-stack-limit.clif | 2 | 0 | i128 |
| throw.clif | 3 | 0 | другой тип результата, вспомогательные тела без своего вектора |
| trapnz.clif | 4 | 0 | другая разрядность или ABI, i128, вещественные типы |
| trapz.clif | 4 | 0 | другая разрядность или ABI, i128, вещественные типы |
| trunc.clif | 4 | 0 | вещественные типы |
| try_call.clif | 3 | 0 | вспомогательные тела без своего вектора, другая разрядность или ABI |
| uadd_overflow.clif | 2 | 0 | несколько результатов |
| uadd_overflow_128.clif | 1 | 0 | i128 |
| uadd_overflow_narrow.clif | 2 | 0 | несколько результатов |
| uadd_overflow_trap.clif | 6 | 0 | инструкция uadd_overflow_trap, вспомогательные тела без своего вектора, другая разрядность или ABI |
| udiv.clif | 6 | 0 | инструкция udiv, другая разрядность или ABI |
| umul_overflow.clif | 4 | 0 | несколько результатов |
| umulhi.clif | 4 | 0 | другая разрядность или ABI, инструкция umulhi |
| urem.clif | 9 | 0 | другая разрядность или ABI, инструкция urem |
| usub_overflow.clif | 5 | 0 | i128, несколько результатов |
| vec-cmp-true-opts.clif | 8 | 0 | SIMD |
| winch.clif | 5 | 0 | ABI Winch |
| [x64-bmi1.clif](x64-bmi1.lancy) | 6 | 3 | другая разрядность или ABI |
| [x64-bmi2.clif](x64-bmi2.lancy) | 2 | 1 | другая разрядность или ABI |
| x64-xmm-mem-align-bug.clif | 1 | 0 | вещественные типы |
| x64/fmsub.clif | 6 | 0 | вещественные типы, SIMD |
