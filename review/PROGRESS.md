# Прогресс рефакторинга

Ветка: `refactor/review-fixes`. План: `review/parts/09-plan.md` (фазы Ф0–Ф9). Владелец принял все рекомендации раздела «Решения для владельца» (Р1–Р18). Р3: LLVM 19 на всех ОС (LLVM 21 недоступен в окружении сборки; на Windows уже 19).

Правило: один агент за раз, одна фаза (или подфаза) на агента. После каждого шага — коммит и запись сюда. Продолжать с первой незакрытой фазы.

## Фазы
- [x] Ф0 — подтверждённые ошибки и быстрые исправления
- [x] Ф1 — страховочная сетка: явный путь понижения, харнессы, CI
- [x] Ф2 — типизированные ошибки и общий `Trap`
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

### Ф1.1 — явный путь понижения (коммит e59f65a)
- Закрыто: MAC-01, MAC-08, SPL-04(а); Р4.
- `#[kernel]` = IR или `compile_error!` с исходной причиной и подсказкой; `#[kernel(rust)]` (на функции или методе `#[kernel] impl/trait`) — явный Rust-импорт. Эвристика `needs_rust_abi` (выбор Rust-ABI по синтаксису) удалена целиком вместе с `rust_region/LeafSyntax/calls_overload/supports_context_call`: маршрут — атрибут, а не догадка. `publish`/`trait_defaults` возвращают `syn::Result`, ошибки всех методов блока собираются вместе. Корень крейта — `#[kernel(crate = path)]` (позиционный путь никто не использовал, и он конфликтовал с `rust`).
- Rust-ядро публикует тот же API: `build_with_options`/`ir_with_options` (опции не влияют) и `print` (ошибка). Заимствованный `#[comptime]` у Rust-ядра — ошибка компиляции (раньше `Err` при `build`), ветки `borrowed_environment` удалены.
- Отклонение от плана: `Lowering`-константа и `TypedIr::lowering()` не добавлены. После Р4 маршрут детерминирован атрибутом, а у Rust-ядра `ir()` возвращает `RustImport`, не `TypedIr`; тест-сторож и `assert_lowered` стали бы тавтологией. «Каждое `#[kernel]` понижено» гарантирует макрос и проверяет trybuild.
- Найдено и исправлено: вызов локального значения-функции (`callback(x)` у `#[comptime] callback: fn(..)`) генерировал неверный Rust — теперь понятная ошибка (ui-тест). Заглушка `__SpliceOpaqueMethod` остаётся: она нужна для методов обычных (не `#[kernel]`) `impl`, которые макрос не видит; убрать можно только с моделью типов (Ф7).
- Тесты: 15 ядер с намеренным Rust-путём помечены `#[kernel(rust)]` (rust_regions + language); `trybuild` ui-тесты `splice/tests/ui/` (метка цикла, `i64`, MAC-02 составное присваивание `usize`, методы, требование трейта `Self::Output`, опции атрибута, заимствованный comptime). `compound_index_assignment_matches_rust_usize` проходил только через откат — заменён ui-кейсом; **Ф4 обязана вернуть его рантайм-версию** (см. git 16cb379) вместе с настоящим `Index`.
- CI arm64: `specialization_limits` падал под QEMU на 10-секундном watchdog (wall-clock). Watchdog удалён: завершение доказывают `is_err`-утверждения бюджета, зависание ловит `timeout-minutes` джобы (Ф1.6).
- Проверки: fmt, clippy `-D warnings`, `cargo test --workspace --release` — зелёные.

### Ф1.2 — переносимые тесты lancy (коммит 657e1a5)
- Закрыто: LNC-01, A64-04.
- `traps/pointers/control_flow/integer_runtests/generated/numeric_coverage(+conversions, rounding)` перенесены в `lancy/src/jit/tests/` без `cfg(target_arch)` (не в `lancy/tests/`: нужны `pub(crate)` `hir::eval`, `interpret`, `interpret_raw`). x64-специфичное осталось в `lir/x64/tests` (`isel.rs::all_scalar_casts_emit_call_free_machine_code`, `packed_fp`).
- `numeric_coverage`: кроме «мягких» входов — граничные (NaN, sNaN, ±0, ±inf, субнормальные, экстремумы, ничьи округления, MIN/MAX/−1, 0-делители, сдвиги ≥ ширины), каждое значение в паре с каждым; сравнивается `Result` вместе с видом ловушки. Расхождений нет ни на x64, ни на NEON/SVE/SVE2.
- ABI-проба callee-saved переносимая: `ScalarKernel::invoke_with` (production `invoke` — его частный случай) позволяет войти через вызывающего с канарейками: iced на x64, `asm!` для x19–x28/d8–d15 на AArch64; нормальный выход и ловушка, Always/Never. Проверено, что порча x21/d12 ловится. Новые тесты: 8 потоков на одном ядре + `drop` в другом потоке; `EmptyCode`.
- Проверки: `cargo test -p lancy --release` при `SPLICE_SIMD=sse|avx2|auto`; под `qemu-aarch64 -cpu max`: neon, sve VL16, sve2 VL32/VL64 — зелёные; clippy x64 и aarch64; `cargo check --target x86_64-pc-windows-msvc --tests`.

### Min/Max с плавающей точкой (коммит 6d1d681, по итогам CI)
- `hir::eval` задаёт Min/Max явно как IEEE 754-2019 minimumNumber/maximumNumber: NaN (в т.ч. sNaN) уступает числу, `min(-0, +0) = -0`, `max(-0, +0) = +0`. Ранее ответ зависел от цели: FMINNM на arm64 возвращал quiet NaN для sNaN, MINPS на x64 отдавал второй операнд при ±0.
- x64: скалярный и упакованный путь (вкл. F16) после MINPS/MAXPS чинят ничью через OR/AND знаков; aarch64 F16 идёт через `ordered_minmax`. Тесты `numeric_coverage` (граничные входы) прогнаны при `SPLICE_SIMD=sse|avx2|avx512|auto` и под QEMU neon/sve/sve2.

### Ф1.3 — инварианты lancy (коммит fc6e28d)
- Закрыто: HIR-09, X64-19. Отложено: A64-16 (golden-таблица `llvm-mc`) — в Ф6; HIR-13 (`DebugStore` при инлайнинге) — своя фаза.
- `checkpoint(f, pass)`: `f.verify()` + паника в `cfg(any(test, debug_assertions))` после каждого прохода `simplify` и после `inline`. SCCP проверяется после `prune` (до него недостижимые блоки законно ссылаются на удалённые значения).
- Новые тесты: `witnesses::index` против наивной модели (случайные вложенные/смежные интервалы); SCCP с uniform-векторами (`filetests/passes/uniform-vector-constants.lancy`); рекуррентность при `trip > i32::MAX`; планировщик — 300 случайных тел, частичный порядок Ordered/Barrier соблюдён и перестановки действительно происходят; `Liveness` против итеративного dataflow на случайных CFG; исчерпывающий roundtrip `logical_immediate` против DecodeBitMasks (5334 значений для 64 бит, 1302 для 32).
- X64-19: пул констант обязателен (`Native::emit_allocated(.., pool)`); путь `mov r10`-splat и тестовый `Native::emit` удалены.

### CI arm64: SVE с VL 48 байт (коммит 08b19f1)
- Строка QEMU `sve2 max 48 32` падала SIGSEGV в `sql_plan` только с LLVM. На main было LLVM 21, сейчас 19 (Р3); AArch64-бэкенд LLVM считает vscale степенью двойки, а VL 48 даёт vscale 3. Теперь `target::scalable`: SVE-код только при VL степени двойки, иначе NEON фиксированной ширины профиля; ядро, собранное с SVE, отказывается работать в потоке с VL не степени двойки. Тест `aarch64_tests::sve_code_is_only_generated_and_run_at_power_of_two_vector_lengths` (prctl 32/48). Локально aarch64 LLVM не линкуется — проверено clippy под aarch64, рантайм подтверждает CI.
- `frontend.rs::collection_computed_by_the_application_can_specialize_a_kernel` падал в debug: в debug-сборке `+` — `CheckedAdd`. Ожидание теперь следует профилю сборки.
- Проверки: fmt, clippy 1.94.1 `-D warnings`, `cargo test --workspace --release`, `cargo test -p splice-kernel` (debug) — зелёные.

### Ф1.4 — харнесс filetests (коммит fb4a60b)
- Закрыто: LNC-10, LNC-06(г), LNC-09, LNC-11, LNC-15.
- LNC-10: `build.rs` удалён. Отклонение от плана: не `[[test]] harness=false`, а in-crate тест `filetests::filetests`, который обходит `filetests/` во время запуска, исполняет файлы параллельно и сообщает все падения разом; фильтр — `LANCY_FILETEST=<подстрока пути>`. Причина: харнессу нужны `pub(crate)` стадии (интерпретатор, проходы, листинги, `check_machine` обоих бэкендов); `__testing`-модуль сделал бы их частью публичного API крейта. `cargo package -p lancy` собирается.
- LNC-06(г): директива, этап которой не достигнут, — ошибка (`all_checks_reached`). Нашлась реальная дыра: AArch64 `check_machine` не потреблял `@aarch64`-проверки.
- LNC-09: `@run` типизирован по сигнатуре (литерал без суффикса берёт тип параметра, суффикс обязан совпасть; hex — битовый образ, `0x3c00_f16`; `vec.i32[1 2]` повторяется до ширины профиля), по паре буферов на каждый `ptr`, сравнение результата по байтам типа; натив через `call_words_raw`. Интерпретатор работает на `Bits`, буфер `k` адресуется как `k << 32`. Фикстура `passes/typed-runs.lancy`.
- LNC-15: `InterpretError { Trap(CallError), OutOfBuffer { buffer, offset }, Unsupported, Invalid }`; таблицы строк в харнессе и `unreachable!` удалены, тесты сравнивают `CallError` напрямую.
- LNC-11: `@aarch64 <профиль> has|lacks <имя>`; 68 кодировок в `ENCODINGS` с именами по `llvm-mc`, 189 строк фикстур переписаны. Неизвестное имя/профиль — ошибка. Поиска «только в `text_len`» не потребовалось: у AArch64 нет литерального пула, все слова — инструкции.
- Проверки: fmt, clippy 1.94.1 (x64, aarch64, `experimental-simd`), `cargo test --workspace --release`; lancy под QEMU neon VL16, sve VL16/32, sve2 VL48/64; typed-runs при `SPLICE_SIMD=sse|avx2|avx512`.

### Ф1.5 — splice-testkit и тесты (коммиты b4b10d5, a47dfb3, 9de13da, 5316a83)
- SPL-02: крейт `splice-testkit` (`publish = false`), `each_backend!(B => …)` — единственный список {Lancy, Cranelift, Llvm, LlvmManualSimd}. Все ≈60 групп `check::<B>` и ручные пары build/compile в `splice/tests` переведены; покрытие Cranelift/LlvmManualSimd выросло с 18/4 до всех тестов — расхождений не нашлось. Оставлены одиночные `build::<Lancy>` там, где проверяется ошибка фронтенда (бэкенд не участвует).
- SPL-03/MAC-17 (дифф. часть): `agree()` — матрица бэкенд × {build, compile(ir)} × {Debug, Release}, одинаковые наблюдения включая буферы после ошибки; `splice-testkit/tests/differential.rs` воссоздаёт `zz_diff.rs` со сверкой с Rust: 23 `Binary` против `checked_eval`, приведения, float, сравнения, целые полосы, `first_n`, compress/expand, u16 gather/scatter у 32768/65535, частичные доступы, narrow/widen. Прогнано при `SPLICE_SIMD=auto|sse|avx2|avx512` и в debug.
- SPL-14: `builders!(B => …)` в бенчмарках (аргументы один раз), `xorshift` в `benches/common`; `cargo bench -- --test` зелёный.
- SPL-15: свой `tracing::Layer` записывает пути закрытых спанов; тест проверяет наличие и вложенность стадий, а не формат и не точное число.
- SPL-18: LLVM-ветка дерева из 32 сканов идёт на глубине 4, Lancy — 5.
- SPL-17 (trybuild): `splice/tests/ui/{signatures,declarations,bodies}.rs` — каждая достижимая диагностика `#[kernel]`; снимки под 1.94.1. В `splice-macros` удалены недостижимые ветки (дубль проверки скалярной сигнатуры, ресивер в `parameter()` → `unreachable!`, конец диапазона → `expect`). Непокрываемо: «expected a cfg condition» — rustc раскрывает пустой `#[cfg_attr()]` до макроса.
- splice-jit: тест читает символьный файл обратно через `object`+`gimli` (ELF, символы, строки, переменные); `tests/registry_threads.rs` — 8 потоков × 200 регистраций.
- CORE-14: модульные тесты `hir::verify` — по одному на каждую ветку ошибки (пустая функция, метки, вызов, регистры, debug, SIMD, copy, буферы, инициализация на всех путях, выпадение за конец).
- BE-15: закрыт дифференциальным тестом; матрица профилей в CI — в Ф1.6. `assert_lowered` не вводился (отклонение 1.1).
- Закрыто: SPL-02, SPL-03, SPL-14, SPL-15, SPL-17(trybuild), SPL-18, MAC-17, CORE-14, BE-15.
- Проверки: fmt, clippy 1.94.1, `cargo test --workspace --release`.

### Ф1.6 — CI и сборка (коммиты 02905d3, d9e17f7, 3df2b4a)
- SPL-07: библиотека `splice-core` называется `splice_core`; бэкенды пишут `splice_core::` и `#[kernel(crate = splice_core)]`; `KernelType` получил тот же `#[splice(crate = path)]`, что `ComptimeKey` (общий разбор, trybuild на неизвестную/повторную опцию, тест в `splice-lancy/tests/staging.rs`). Доктесты языка перенесены в фасад: в ядре они падали бы на отсутствии `splice`, а не на проверяемом свойстве. `RUSTDOCFLAGS=-D warnings cargo doc --workspace --no-deps` чистый.
- SPL-08: у `splice-kernel` нет dev-зависимости от `splice-llvm`; LLVM в тестах/примерах — через опциональную зависимость фасада под `llvm`, бенчмарк сравнения — `required-features = ["llvm"]`. `each_backend!` раскрывает LLVM по фиче вызывающего крейта; `splice-testkit` по умолчанию включает `splice-kernel/llvm`, поэтому `cargo test --workspace` гоняет все бэкенды, а `cargo test -p splice-kernel` работает без LLVM (Lancy + Cranelift). Ручные пары Lancy/LLVM в тестах заменены на `each_backend!` (+Cranelift, +LlvmManualSimd). Тесты `cranelift`/`llvm` фасада исполняются в `--workspace` и `--all-features`.
- SPL-11: `cargo xtask` (`xtask/src/main.rs`): `ci [--quick]`, `lint`, `test [--debug] [--target]`, `features`, `profile SIMD [--target] [--expect-bytes]`, `static-llvm`, `doc`, `bench`; единственный список пакетов профильных прогонов (`PROFILE_PACKAGES`, теперь с `splice-cranelift`, `splice-testkit` и всеми тестами `splice-kernel`). Workflow — только вызовы xtask: lint (x64, arm), test {ubuntu-x64, ubuntu-arm, windows-2022} × {release, debug}, features, profile (sse/avx2/avx512/avx10, neon/sve/sve2; недоступный на раннере — пропуск с сообщением), qemu (sve VL32, sve2 VL64, sve2 VL48 — профиль обязателен через `--expect-bytes`); `concurrency`, `timeout-minutes`, `rust-cache`; `push` только на main (PR больше не гоняется дважды). AGENTS.md и README ссылаются на xtask.
- LNC-08: clippy и тесты `lancy --features experimental-simd` в `lint`/`features`.
- BE-05 (джоба): Linux-LLVM `prefer-dynamic` вместо `force-dynamic`; `xtask static-llvm` подсовывает llvm-config без `libLLVM.so`, собирает `splice-kernel --features llvm,cranelift --test debug_symbols` со статическими архивами и проверяет `ldd` (нет libLLVM) и `nm` (по одному `__jit_debug_descriptor`/`__jit_debug_register_code`).
- `--no-default-features`: `cargo check -p splice-kernel`; `cargo bench … -- --test` — в `features`.
- Проверки: `cargo +1.94.1 xtask ci` локально зелёный (lint, workspace release, features, doc, static-llvm). Windows исполняется впервые — результат по CI.
- Первый debug-прогон в CI нашёл два теста `splice-lancy/tests/staging.rs`, рассчитанных только на release: `ir()` берёт `BuildOptions::default()`, который в debug проверяет переполнение (`CheckedAdd`/`CheckedMul`, решение SPL-03). Ожидания теперь зависят от `cfg!(debug_assertions)`; `xtask test` идёт с `--no-fail-fast`, чтобы падения не прятали друг друга. Локальный debug-прогон воркспейса других расхождений не дал.
- CI на 3df2b4a: Windows release зелёный с первого раза (LLVM 19.1.7 через `install-llvm-action`, включая `cargo test -p lancy`); статическая линковка на раннере требовала `libPolly.a` (Ubuntu называет Polly в `llvm-config --link-static`) — в джобу добавлен `libpolly-19-dev`.
- CI на cdf9a56: все 19 джоб зелёные (lint ×2, test 3 ОС × release/debug, features+static-llvm, profile ×7, qemu ×3). Ф1 закрыта.

### Ф2.1 — общий `Trap`, единая кодировка статуса, типизированный `CallError` (коммит 9610d7a)
- Закрыто: CORE-02 (ловушки), BE-14, CORE-03 (а, б, в — сторона вызова), A64-14 (abort и `unreachable!`), часть LNC-05(е).
- `splice_target::abi`: `#[repr(u32)] #[non_exhaustive] Trap { Bounds=1, DivisionByZero, DivisionOverflow, Overflow, RustPanic, Abi }`, `Trap::status()` (код в старшем слове), `decode_status`, `rust_entry` — единственная граница `catch_unwind` у Rust-входов; текст паники кладётся в thread-local слот (`take_panic_message`). Модуль ABI остаётся в splice-target (его используют и lancy, и splice-core); перенос `SliceDescriptor`/ABI Rust-вызова — Ф4.4/4.5.
- Неверная форма вызова на Rust-входе (форма операции контекста, диспетчер, арность пакета, неподдержанный адаптер) — `Trap::Abi`, а не «паника».
- Lancy: слот статуса хранит коды `Trap`; side exit-ы x64/AArch64 индексируются `Trap`, Rust-вызов пробрасывает код самого импорта (раньше любой сбой превращался в `RustPanic`). `eval`/`integer` возвращают `Trap`. `lancy::CallError` — `#[non_exhaustive] { Target(TargetError), Signature(SignatureError), Trap(Trap) }`. Числовой помощник AArch64 возвращает любой код и ловит панику (`Trap::Abi`) вместо abort. Общий хвост `trap_if` для пути после вызова (вторая половина A64-14) — в Ф6 вместе с A64-03.
- Cranelift возвращает `Trap::status` (было `-1..-5`) и пробрасывает статус вызываемой функции/импорта без изменений; LLVM пишет в out-параметр коды `Trap`, Rust-вызов — код импорта.
- `splice_core::CallError` — `#[non_exhaustive] { Trap, RustPanic(String), Signature(SignatureError), Target(TargetError) }`, `Executable::call` и `call_boundary` типизированы, `validate_buffers` → `SignatureError`, `From<CallError> for String` удалён. `Binary::checked_eval` → `Result<i32, Trap>`. Вариант `Rust(String)` из плана не понадобился: Rust-ядро, упавшее при вызове, — это тоже `RustPanic`.
- Тесты сравнивают варианты; новые: `splice-testkit::every_backend_reports_the_same_typed_trap` (все ловушки, текст паники импорта, `Abi` от импорта; 4 бэкенда × build/compile × Debug/Release), filetest `rust-import-memory.lancy` с кодом `Abi` от импорта (падает на старом эмиттере x64), типизированные `SignatureError` в `splice-cranelift/tests/execution.rs`, `RustPanic("completion failed")` из операции контекста.
- Проверки: `cargo +1.94.1 xtask ci --quick` зелёный; lancy под QEMU AArch64 (sve2 VL16/VL32, neon) зелёный, clippy aarch64; `cargo check --target x86_64-pc-windows-msvc --tests` для lancy/splice-target/splice-core/splice-lancy/splice-cranelift.

### Ф2.2 — ошибки lancy (коммит 57fce44)
- Закрыто: LNC-05 (а–е, з; ж закрыт в Ф0), HIR-22, HIR-15 (документированием `# Panics`, сигнатуры меняет Ф5/Ф9.1), A64-14 закрыт в Ф2.1.
- `CompileError` — `#[non_exhaustive] { InvalidIr(IrError), Unsupported(Limit), Target(TargetError), Os { operation, source }, Internal(InternalError) }`; `Limit = Recursion | Frame | Workspace { slots } | Registers | BranchRange { offset, target } | CodeSize { bytes }`. Откат инлайнинга — `CompileError::is_limit()` (размерные лимиты, `Recursion` не откатывается — как раньше).
- Ошибки `regalloc2`/`iced-x86`/энкодера AArch64 → непрозрачный `InternalError` с `source()`; для x64 сборки — приватный тип модуля `Failure`, поэтому публичных `From` от сторонних типов нет, форма enum не зависит от цели. Display больше не повторяет причину (`transparent` или только имя операции + `source`).
- «Код больше 2 ГиБ» — `Limit::CodeSize`, а не выдуманный `BranchOutOfRange`.
- `splice_target::simd()` → `TargetError { InvalidVariable, UnknownProfile, Unavailable, VectorLength, VectorLengthNotPowerOfTwo }`; `SimdProfile(String)` удалён. В splice-бэкендах до Ф2.3 временно `map_err(to_string)`.
- `IrError::UseSimdForm` и `IrError::PointerCast` вместо `InvalidNumeric` для вектора в скалярной операции и приведений указателя; почему разрешён только `Ptr→U64` bitcast — в документации варианта. `Ptr↔Usize` не разрешал: изолирующий выбор и бэкенды этого не лоуерят.
- Тесты: `ssa::tests::verify::scalar_operations_name_the_vector_and_pointer_misuse`, `scalable.lancy` ждёт новые тексты, тест fallback перебирает все размерные `Limit`, `compile_preserves_the_verifier_error_from_a_callee` сравнивает вариант вместо `source().downcast`.
- Проверки: `cargo +1.94.1 xtask ci --quick` зелёный; lancy под QEMU sve VL32; clippy aarch64; `cargo check --target x86_64-pc-windows-msvc --tests -p lancy`.

### Ф2.3 — ошибки splice (коммит 3edb64e)
- Закрыто: CORE-03 (г и сторона сборки), CORE-17, SPL-19, LNC-05 остаток (сторона splice-бэкендов). CORE-20: сентинел `Reg(u32::MAX)` остаётся до Ф4.2 (нужен `Result` у `KernelBuilder`); прочие пункты CORE-20 не трогал.
- `splice_core::BuildError` — `#[non_exhaustive] { Staging(String), Comptime(Trap), RustPanic(String), Verify { instr, reason }, Unsupported(String), Target(TargetError), Backend { backend, source } }`; `splice_core::Result<T>` = `Result<T, BuildError>`. `phase_error!` и `From<String>`/`From<…> for String` удалены — `BuildError` и `CallError` больше не конвертируются друг в друга. `CompileInput::Error` у `Function` — `BuildError`.
- Верификатор: `Verify { instr, reason }`; ошибка в вызываемой функции переносится на инструкцию вызова (`BuildError::at`). Display не повторяет причину: `Backend` пишет только имя бэкенда, сама ошибка (`lancy::CompileError`, `gimli`, LLVM, Cranelift) — в `source()`.
- Бэкенды: `map_err(to_string)` заменены на `BuildError::backend("lancy" | "cranelift" | "llvm" | "JIT debug info" | "text IR", e)`, лимиты («too many …») — `Unsupported`, неверное использование `KernelBuilder` — `Verify`; временные `simd().map_err(to_string)` из Ф2.2 — просто `?` (`TargetError`). `splice-llvm` `body::Error` — `thiserror`-enum. Ошибка LLVM-вызова (`self.error`) теперь забирается, а не клонируется.
- Макросы порождают `BuildError::staging(…)`; сбой `comptime!` — `BuildError::Comptime(trap)` (`HostFault::message()` → `HostFault::trap()`), выход за границы в `comptime!` — `Comptime(Trap::Bounds)`; паника при сборке (`build_boundary`) — `RustPanic(текст)`.
- CORE-17: `ContextSignature::with_borrows` проверяет раскладку в `const fn` (неверная — ошибка компиляции, доктесты `compile_fail,E0080`), `size: u32`; `validate()`, `check_context_layout`, `check_context_call` и их вызовы удалены; `ContextFrame::root` возвращает `Self`.
- Тесты сравнивают варианты (`Verify`/`Staging`/`Unsupported`/`Comptime`); новые проверки: `error::tests` (паника при сборке → `RustPanic`, Display без повтора причины), `overflow.rs` и `known_bounds.rs` ждут `Comptime(Trap::Overflow | Trap::Bounds)`, `column_table` — `Staging`. Документация: варианты `BuildError` в `docs/splice-language.md`.
- Отклонение: `Catalog::new` в примере `relational` оставлен с `Result<_, String>` — это ошибка предметной области примера, не API Splice.
- Проверки: `cargo +1.94.1 xtask ci --quick` зелёный; `cargo check --target aarch64-unknown-linux-gnu --all-targets` (lancy, splice-core, splice-lancy, splice-cranelift, splice-kernel); `cargo check --target x86_64-pc-windows-msvc --tests` (splice-core, splice-jit, splice-lancy, splice-cranelift).
- Полный `cargo +1.94.1 xtask ci` локально зелёный. Ф2 закрыта; отложено: CORE-20 сентинел → Ф4.2, вторая половина A64-14 → Ф6, сигнатуры HIR-15 → Ф5/Ф9.1, `SliceDescriptor`/ABI Rust-вызова → Ф4.4/4.5.
