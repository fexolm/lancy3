# Прогресс рефакторинга

Ветка: `refactor/review-fixes`. План: `review/parts/09-plan.md` (фазы Ф0–Ф9). Владелец принял все рекомендации раздела «Решения для владельца» (Р1–Р18). Р3: LLVM 19 на всех ОС (LLVM 21 недоступен в окружении сборки; на Windows уже 19).

Правило: один агент за раз, одна фаза (или подфаза) на агента. После каждого шага — коммит и запись сюда. Продолжать с первой незакрытой фазы.

## Фазы
- [ ] Ф0 — подтверждённые ошибки и быстрые исправления
- [ ] Ф1 — страховочная сетка: явный путь понижения, харнессы, CI
- [ ] Ф2 — типизированные ошибки и общий `Trap`
- [ ] Ф3 — цель как значение
- [ ] Ф4 — splice-core: типизированные скаляры и единый контракт бэкендов
- [ ] Ф5 — lancy HIR
- [ ] Ф6 — lancy LIR и бэкенды
- [ ] Ф7 — splice-macros
- [ ] Ф8 — производительность
- [ ] Ф9 — публичный API, unsafe, документация, crates.io

## Журнал

### Ф0.1 — CI после переименования и LLVM 19 (коммит 3b7d815)
- Закрыто: SPL-01, SPL-16(п.4), SPL-21; Р3 (LLVM 19 на всех ОС, часть BE-03).
- `-p splice` → `-p splice-kernel` в CI и AGENTS.md; шаг профилей теперь гоняет и `splice-cranelift`.
- Linux x86-64/AArch64 переведены на `llvm19-1-force-dynamic`; CI ставит `llvm-19-dev` из Ubuntu (без apt.llvm.org), `LLVM_SYS_191_PREFIX`. Удалена развилка `capture_attribute_kind` (LLVM 19 — всегда `nocapture`). `Llvm::version` ждёт 19 везде. README описывает одну версию.
- `.gitignore`: `target/` на любой глубине, `tests/fixtures/**/Cargo.lock`.
- Проверки: fmt, clippy `-D warnings`, `cargo test --workspace --release` — всё зелёное (контейнер, LLVM 19.1.1).

### Ф0.2–0.3 — Cranelift BE-01, BE-02 (коммит e8b2967)
- BE-01: причина подтверждена в сгенерированном `assembler.isle` Cranelift 0.132 — только `x64_cmpps_a`/`x64_cmppd_a` принимают `XmmMem128` вместо `XmmMemAligned128`. Без AVX сравниваемые векторные загрузки проходят через пару `bitcast` (регистровый no-op, который x64-лоуеринг не вливает): `keep_compared_loads_in_registers` в `splice-cranelift/src/lower/simd.rs`. Тест `splice-cranelift/tests/sse_unaligned.rs` перезапускает себя с `SPLICE_SIMD=sse` (профиль глобален до Ф3 — после Ф3 перевести на профиль-значение в одном процессе). Апстрим-репро: CLIF `fcmp eq` с `load.f32x4` по адресу +4 без `has_avx` — отправить при наличии доступа (из контейнера нет).
- BE-02: `fcvt_to_sint_sat.i64x2` на x64 — поэлементно. Тест `splice/tests/conversions.rs`: все допустимые `Cast/Widen/Narrow` × Lancy/Cranelift/Llvm/LlvmManualSimd × build/compile против Rust `as` (NaN, ±inf, −0, границы насыщения). Прогнан при `SPLICE_SIMD=auto|avx2|sse`. Это зачаток харнесса Ф1.5.

### Ф0.4 — HIR-01, HIR-13: scratch и отладочная область при инлайнинге (коммит 5af261a)
- По Р10(а): `Op::CallScratch { offset, bytes }` (смещение кратно 8), у Rust packet call — собственное смещение пакета в scratch (пакет пишется в ту же память, иначе инлайненный вызов затирал бы scratch вызывающей). Верификатор: `IrError::InvalidScratch`.
- Инлайнер сдвигает scratch/packet/`DebugStore` вызываемой функции выше собственных областей вызывающей (`Function::scratch_bytes/debug_bytes`); `check_expansion` ограничивает вложенные суммы `u32` (`FrameOverflow`). LIR x64/AArch64 адресуют `scratch_base + offset`.
- Интерпретатор получил scratch на кадр (только raw-режим), filetest-прогон без буфера идёт через `call_raw`. Синтаксис `.lancy`: `call_scratch OFF BYTES`, `debug_store OFF v`; листинг пакета — `... scratch N`.
- Тест `lancy/filetests/passes/inline-call-scratch.lancy` (Always/Never → 719, пакетный вызов, debug-область); падает при отключённом сдвиге. Прогнан на x64 и под qemu AArch64 (neon). `memory.rs` (must-alias по смещению) не трогал — корректность не требует, это оптимизация для Ф5.

### Ф0.5 — MAC-02 быстрый фикс
- `assignment()` применяет то же правило, что `binary()`: составное присваивание runtime-`usize` отвергается. Из-за MAC-01 ядро пока молча откатывается в Rust-импорт, поэтому результат совпадает с Rust; trybuild-сообщение появится, когда Ф7 (Р4) сделает откат ошибкой.
- **Ф4 обязана заменить** эту заглушку настоящим 64-битным `Index` (Р6) с беззнаковыми checked-операциями и убрать запрет из `assignment()`/`binary()`; тест `language.rs::compound_index_assignment_matches_rust_usize` при этом должен остаться зелёным уже на JIT-пути.
- Тест сравнивает Lancy/Cranelift/LLVM с Rust-функцией (паника ⇒ ошибка JIT) в debug и release; до фикса падал (`Ok(0)` против `Ok(1)` в release).

### Ф0.6 — MAC-03 (коммит 778b4e2)
- В `splice-core` один `HostFault` (деление на ноль / переполнение деления / переполнение) с i32-операцией ядра, которая ловит так же, и guard-трейты `HostArithmetic`/`HostShift`/`HostNegation` для всех целых примитивов (autoref: float и пользовательские типы — своими операторами). Макрос ставит guard перед host-бинарными, составными и унарным минусом; staged-код ловит в точке достижения, `comptime!` возвращает сообщение как ошибку сборки. `source::division_fault` → `source::fault_panic`.
- Тест `overflow.rs::host_arithmetic_faults_are_deferred_for_every_integer_type`: build без паники (хук паники на поток — эквивалент проверки `panic=abort`, отдельный бинарь не нужен), вызовы совпадают с Rust; Lancy/LLVM/Cranelift × build/compile × Release/Debug, cargo debug и release. До фикса падал.

### Ф0.7 — гигиена: MAC-04, MAC-05(б), MAC-22, MAC-23(часть) (коммит 673f35e)
- Все пути в `emit!/quote!` полностью квалифицированы (скрипт по телам макросов + ручные правки UFCS для `Into/ToOwned/Default`); `break #end (::path)` в скобках. Внутренние элементы — `__SpliceEnvironment`, `__SpliceOpaqueIr`.
- MAC-04: переименование `self` и в токенах макросов. MAC-22: `RebasePaths` переписывает корень `use self::/super::`. MAC-23: lint-атрибуты параметров переносятся на функцию; `#[cfg]` на параметре и единая детекция маркеров — в Ф7.1 (там же MAC-23 по плану).
- Тесты `splice/tests/hygiene.rs` (+ юнит `rebased_use_trees_keep_their_targets`); каждое ядро проверяет, что оно понижено (`print`), а не откатилось.
- **Отложено в Ф7.1**: пользовательский тип с именем публичного сгенерированного элемента (`Signature`, `BuildArgs`, `CallArgs`) и `mod foo` рядом с `#[kernel] fn foo`. Причина: оба требуют не переносить сигнатуры в сгенерированный `mod` (алиасы/генерация на уровне элемента) — это и есть дизайн эмиттера Ф7.1; точечный фикс в Ф0 означал бы переписать эмиссию дважды. Тест-кейсы добавить в `hygiene.rs` в Ф7.1.
