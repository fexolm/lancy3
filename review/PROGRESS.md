# Прогресс рефакторинга

Ветка: `refactor/review-fixes`. План: `review/parts/09-plan.md` (фазы Ф0–Ф9). Владелец принял все рекомендации раздела «Решения для владельца» (Р1–Р18). Р3: LLVM 19 на всех ОС (LLVM 21 недоступен в окружении сборки; на Windows уже 19).

Правило: один агент за раз, одна фаза (или подфаза) на агента. После каждого шага — коммит и запись сюда. Продолжать с первой незакрытой фазы.

## Фазы
- [x] Ф0 — подтверждённые ошибки и быстрые исправления
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

### Ф0.8 — CORE-04 (коммиты 3704334, e90df0d)
- `immediate_condition` удалён из `KernelBuilder`, `FunctionBuilder`, splice-lancy и прямого LLVM-пути: бэкенды больше не помнят «последнюю константу». Сворачивание условий — во фронтенде: host-выбор макроса с ветками `Reg(Int|Bool)` даёт стадию `Word`, runtime-условие сворачивается через `RunArg::known`, поэтому все бэкенды и оба пути сборки видят одинаковый IR. Воспроизвести неверную компиляцию через публичный API не удалось; тест `language.rs::mixed_stage_conditions_fold_in_the_frontend_on_every_path` фиксирует семантику (известное условие — без `Branch`, неизвестное — с ним; прямой Text-build совпадает с `Text::compile`).
- Регрессия Ф0.6, найдена полным прогоном: отложенная ошибка host-`%` ловилась через `/`. `HostFault::Division*` теперь несут оператор (ловит тест `splice-text/tests/listing.rs`).

### Ф0.9 — BE-05 = CORE-15 (коммит 692d6ee)
- Отклонение от буквы Р11(а): «`extern` при подключённом LLVM» через Cargo-фичу невозможен, потому что фичи в воркспейсе унифицируются, и бинари lancy/cranelift без LLVM получили бы неразрешённые символы. Вместо этого на ELF символы `__jit_debug_*` определены как **weak** (`global_asm!`) и используются через `extern`. Статический LLVM даёт сильное определение (оно и используется, дубля нет), с `libLLVM.so` динамический линкер связывает LLVM с единственной экспортированной копией. Суть Р11(а) сохранена: в процессе одно определение, пишем под своим замком, GDB-слушатель ORC запрещён (документировано в `splice-llvm/src/jit.rs` и README). Windows без изменений: LLVM там DLL со своими приватными копиями.
- `links = "splice_jit_gdb"` (+ пустой `build.rs`, его требует Cargo).
- Тест `splice-jit/tests/registry.rs` (регистрация попадает в общий дескриптор). Ручная проверка BE-05: тот же тест, слинкованный с `JITLoaderGDB.cpp.o` из `libLLVMOrcTargetProcess.a`, на старом коде падает (`rust-lld: duplicate symbol`), на новом проходит, `nm` показывает одно (сильное) определение. GDB-смоук `debug_kernel` (lancy/llvm) работает, AArch64 под qemu — зелёный. Постоянная CI-джоба со статическим LLVM — Ф1.6.

### Ф0.10 — мелкие ошибки (коммиты 8b88d23, c1ae8c0, 1f97492, a7061c3, fe37be3)
- X64-20: `DebugPoint` вне debug ничего не выдаёт (x64 и AArch64); тест `lancy/tests/function.rs::release_code_drops_debug_points` (79 байт против 15 до фикса).
- HIR-20: `Select` по маске сравнения с «чужими» плечами доходит до обычных правил `Select` (`passes/simd-minmax.lancy`). Разбор правил на функции — в Ф5.
- LNC-05(ж): откат инлайнинга по любому лимиту размера (`CompileError::is_size_limit`: `FunctionTooLarge`, `BranchOutOfRange`, `WorkspaceTooLarge`, `FrameOverflow`); тест `jit/tests.rs::every_size_limit_of_the_expanded_body_falls_back_to_calls` (естественного репро нет: инлайненные соседние вызовы делят одну scratch-область). Остальное из LNC-05 (`Limit`, `#[non_exhaustive]`) — в своей фазе.
- SPL-17: стабильный rustdoc **не проверяет** коды ошибок `compile_fail` (проверено: `E9999` проходит). Поэтому каждый доктест без кода стоит после принимаемого двойника, который отличается только проверяемым свойством. trybuild-снимки — Ф1.5 (MAC-17).
- BE-18: `intrinsic(module, name, types) -> Result` с `body::Error::MissingIntrinsic` вместо паники; код тел возвращает `body::Error` (обёртка `BuilderError`). BE-17: явный флаг `root` в `Graph::function/body`.
- CORE-16: `bool` сохраняется как `value & 1` во всех трёх бэкендах (gdb-тест `debug_symbols.rs::gdb_shows_runtime_bools_as_rust_bools` под `#[ignore]`, прогнан здесь: до фикса `255`). Вынос правил сохранения в ядро (`Point::stores`) — в своей фазе (CORE-16, вторая часть).
- CORE-18: `check_initialization` смотрит на `Instr::operands()` (без Debug-переменных). В splice-lancy SSA-строитель читает Debug-переменные «мягко»: на пути без присваивания ставится нулевая заглушка, а настоящее чтение того же параметра остаётся ошибкой. Cranelift и LLVM обработки не требуют. Тест `debug_symbols.rs::source_points_may_name_partially_assigned_variables`.
- CORE-19: `SimdOp::may_trap` (доступ к буферам), `needs_status` только по `may_trap`; тест `frontend.rs::only_operations_that_can_fail_need_a_status_channel`. Таблица `SimdOp::info()` (CORE-10) заменит `may_trap` в своей фазе.
- CORE-23: perf-map: `O_NOFOLLOW`, только обычный файл своего uid, усечение устаревшей карты, одно предупреждение на процесс (юнит-тест). CORE-25: `e_machine` — константа по архитектуре с `compile_error!`. CORE-26: `Auto` даёт 32 байта только при AVX2, `neon` доступен на AArch64 вне Linux. Ветки Lancy `32 if cpu.avx2` теперь недостижимы через `Auto`; убрать их при переходе на профиль-значение (Ф3).
- Проверено: fmt, clippy `-D warnings`, `cargo test --workspace --release`; `SPLICE_SIMD=sse|avx2` для splice-cranelift/splice-lancy и SIMD/debug-тестов splice-kernel; lancy под qemu AArch64; `cargo check --target x86_64-pc-windows-msvc` для splice-jit/target/core.
