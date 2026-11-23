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
