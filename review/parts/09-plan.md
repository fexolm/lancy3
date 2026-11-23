# 09 — Итоговый план исправлений (синтез частей 00–08)

Источник: отчёты `00-clippy.md` … `08-splice-facade-docs-ci.md`. Репозиторий не менялся. К исходникам я обращался только для проверки сквозных утверждений:
- `simd.yml`: шаг с QEMU вызывает `-p lancy … -p splice --tests` одной командой. Поэтому из-за SPL-01 падает весь шаг, включая тесты lancy под SVE. Утверждение A64-04 «CI уже гоняет qemu с VL 16/32/64» после коммита `e209c39` неверно.
- `splice-core/Cargo.toml` содержит `[lib] name = "splice"`, это подтверждает SPL-07.
- `immediate_condition` действительно объявлен в трейте `KernelBuilder` (`frontend/mod.rs:59`) и вызывается из `control.rs:145` (CORE-04).

Правила AGENTS.md и codex.md считаются обязательными для каждого шага:
- оба пути сборки (`build::<B>` и `compile(&ir)`);
- сверка с независимым эталоном;
- для каждой оптимизации — отдельный агент, который проектирует пример и контрпример;
- замеры «до/после» через `cargo bench -p splice-kernel --bench backends`;
- `SAFETY:`-комментарии;
- запрет автовекторизации;
- `LlvmManualSimd` без LV и SLP.

---

## Сводка

### Итоги по серьёзности

| Серьёзность | Число |
|---|---|
| crit | 1 |
| high | 36 |
| med | 72 |
| low | 69 |
| **Всего с ID** | **178** |

Сюда не входит отчёт clippy (часть 00): у него нет ID. В индексе он оформлен как три синтетические записи CLIP-01…03 без серьёзности.

### Итоги по частям

| Часть | Префикс | Всего | crit | high | med | low |
|---|---|---|---|---|---|---|
| 01 lancy HIR | HIR | 25 | – | 3 | 9 | 13 |
| 02 lancy LIR + x64 | X64 | 21 | – | 5 | 8 | 8 |
| 03 lancy aarch64 | A64 | 18 | – | 3 | 7 | 8 |
| 04 lancy misc/API/тесты | LNC | 18 | – | 2 | 8 | 8 |
| 05 splice-macros | MAC | 26 | – | 8 | 11 | 7 |
| 06 splice-core/text/target/jit | CORE | 27 | – | 6 | 10 | 11 |
| 07 splice-бэкенды | BE | 22 | 1 | 5 | 9 | 7 |
| 08 фасад/тесты/docs/CI | SPL | 21 | – | 4 | 10 | 7 |
| 00 clippy (pedantic) | CLIP | 3 синт. | – | – | – | – |

Примерно 25 находок дублируют друг друга или связаны отношением «следствие — причина». Они сведены в темы ниже. Независимых корневых причин около 16.

### 10 главных проблем

1. **HIR-01 (+HIR-13):** подтверждённая ошибка компиляции (miscompile). Инлайнер копирует `CallScratch`/`DebugStore`, и scratch вызывающей и вызываемой функции оказывается одной областью: `Always` даёт 1919, `Never` — 719.
2. **MAC-02 ← CORE-01, MAC-07:** `usize` живёт в 32-битном регистре с i32-проверками. Подтверждено: на пустом срезе `i -= 1` в Rust паникует, а JIT возвращает `Ok(1)`. Корень — скалярная система только на `i32`.
3. **BE-01 (crit), BE-02:** Cranelift падает с SIGSEGV при `SPLICE_SIMD=sse` и не компилирует `Cast f64→i64`. Оба случая воспроизведены, ни один тест их не ловит (SPL-02, BE-15).
4. **MAC-01, MAC-08, SPL-04:** ядро или метод `impl` молча уходит в Rust-импорт. Пользователь и тесты этого не видят, API модуля меняется, падение производительности на порядки.
5. **SPL-01 (+SPL-11, LNC-08), BE-05/CORE-15:** CI сломан переименованием пакета. Профили SSE/AVX2/AVX-512 на уровне Splice и весь QEMU-SVE шаг не исполняются. При статическом LLVM линковка падает на дублирующихся символах GDB JIT.
6. **MAC-03, MAC-04, MAC-05:** `build` паникует на не-i32 host-арифметике в неисполняемой ветке. На корректном пользовательском коде сгенерированный код не компилируется: `comptime!(self.x)`, `use Status::*`, коллизия `Signature`.
7. **CORE-04:** отсечение ветки зависит от пипхол-состояния бэкенда (`immediate_condition`). Один пропущенный сброс в одном бэкенде — и получится латентный miscompile.
8. **CORE-02/05/06/07, BE-07, BE-14:** вызов ядра, коды ловушек, план Rust-вызова, верификатор и семантика «запись до ошибки» реализованы трижды, с разными кодировками и разным покрытием.
9. **CORE-12, X64-06, BE-04, X64-03:** цель задаётся глобальным состоянием процесса. Один профиль даёт три разных набора ISA в трёх бэкендах. Win64-ABI не исполняется нигде, хотя Windows — основная платформа замеров.
10. **X64-01, X64-02, A64-01, A64-02 (+X64-04/A64-03):** обрывы производительности. Скаляры вне i32 идут через стековый `Lanes`, нет `vzeroupper`, float min/max и f16 на aarch64 считаются по полосам (60–250 инструкций). При этом есть три копии численной семантики.

---

## Сквозные темы

Для каждой темы ниже: ID, корень, целевой дизайн и что после него можно удалить.

### T1. Скалярная система типов только на `i32`
**ID:** CORE-01 (корень) → MAC-02, MAC-07, SPL-12, CORE-16 (`bool` = −1 в DWARF), CORE-09, LNC-09 (`@run` только i32), LNC-03 (ручная упаковка в `u64`), MAC-09 (списки примитивов), SPL-10 (спека обещает «правила Rust для типов»).

**Корень.** В переносимом IR `ValueType::Scalar` не несёт типа. `Instr::Const(Reg, i32)`, `Executable::call(&[i32]) -> i32`, импорты `extern "C" fn(i32,i32)->i64`, слова пакетов и debug-слоты рассчитаны на одно 32-битное слово. В Lancy HIR, наоборот, типы есть (I64/U32/F32/…), то есть ограничение живёт только в splice-слоях. Макрос «угадывает» тип по синтаксису, потому что передать его в IR некуда.

**Целевой дизайн.**
- `splice-core/src/hir/types.rs`: `enum ScalarType { Bool, I32, U32, I64, U64, Index, F32, F64 }` (`Index` = настоящий `usize`, на 64-битных целях 64 бита, беззнаковые checked-операции), `ValueType::Scalar(ScalarType)`, `struct Constant { bits: u64, ty: ScalarType }`. В `Function` вместо `arity` хранится `params: Vec<ScalarType>`.
- `Binary::checked_eval(Constant, Constant) -> Result<Constant, Trap>` — единственный эталон скалярной семантики фронтенда.
- `KernelBuilder::constant(Constant) -> Result<Reg>`. ABI ядра работает словами `u64` с типом из сигнатуры (см. T5).
- Макрос оперирует типизированными хэндлами `Reg<T: KernelScalar>`, а тип литерала выводит rustc (T3).
- `debug::Scalar` удаляется в пользу `ScalarType`. `Bool` сохраняется как `value & 1` (CORE-16).

**Что удаляется:**
- обход в `schema_plan` через `first_n(1)` и `load_select` (SPL-12);
- запрет «cast the runtime index to i32» и `Scalar::{Int, Index, Length, Inferred}` в макросе;
- особые ветки `bool = -1`;
- `ImportWord`/`RunArg` через `i32`.

### T2. Молчаливый откат и стратегия «попробуй, при ошибке откатись»
**ID:** MAC-01 (корень), MAC-08, MAC-16, MAC-20, SPL-04, SPL-10, CORE-27 (эталон, медленный в Rust-пути), MAC-12 (неподдерживаемые конструкции уходят в откат).

**Корень.** `import::expand_function` и `methods::publish` выбрасывают ошибку скалярного понижения и публикуют Rust-импорт. Внутри понижения применяется спекулятивный разбор `if let Ok(..) = host_raw(..)`, а ошибки, известные при компиляции, через autoref откладываются до `build`.

**Целевой дизайн.**
- `#[kernel]` означает «скалярное ядро или `compile_error!` с исходным span».
- `#[kernel(rust)]` (то же для метода) — явный Rust-регион.
- Выбранный путь виден: `const __SPLICE_LOWERED: Lowering` и `TypedIr::lowering() -> Lowering::{Ir, RustImport}`.
- Оба пути публикуют одинаковый API (`print`/`build_with_options` у импорта возвращают типизированную ошибку).
- Маршрут выбирается по анализу стадий (T3), а не пробой.
- Ошибки, известные при развёртке, становятся `const { assert!(..) }` или `compile_error!`.

**Что удаляется:** ветка отката в `import.rs:39-49,462-471`, `let Ok(..) else { return Vec::new() }` в `methods.rs`, заглушка `__SpliceOpaqueMethod`, двойные попытки `host_raw`/`paired_operands`, `.or_else` в `memory.rs`.

### T3. Макрос интерпретирует подмножество Rust 6–9 раз без модели языка
**ID:** MAC-06 (корень), MAC-09, MAC-11, MAC-12, MAC-13, MAC-18, MAC-19, MAC-21, MAC-22, MAC-23, MAC-24, MAC-25, MAC-10, MAC-07 (типы).

**Корень.** `expr_with_hint`, `required_expr`, `source_expr`, `Flow::expr`, `region::Collect`, `evaluable`, `pure_expr`, `HostCheck`/`PrepareHost`, `needs_rust_abi` — у каждого свои таблицы узлов и свои области видимости. Имена разрешаются строками (последний сегмент пути, заглавная буква, сравнение спанов), а узлы идентифицируются адресом в памяти.

**Целевой дизайн (раскладка модулей `splice-macros/src/`):**
- `kast/` — нормализованный «Kernel AST»: `Let`, `If`, `Loop{label}`, `Break{label,value}`, `Match{guard}`, `Call{resolved}`, `MethodCall{kind}`, `Comptime(e)`, стабильные `NodeId` от счётчика в пределах одной развёртки.
- `resolve.rs` — дерево областей видимости: замыкания, вложенные `let`, паттерны; каждое использование связано с `NodeId` объявления.
- `types.rs` — одна таблица примитивов и классификация путей с поддержкой `::core::primitive::*`. Пользовательские типы классифицируются через `<T as KernelValue>::KIND`, а не по имени.
- `stage.rs` — единственный анализ стадий: worklist по зависимостям «локал → присваивания», `return` учитывает стадию условия.
- `emit/{ir.rs, eval.rs, print.rs}` — три эмиттера над `kast`.
- `intrinsics.rs` — реестр `name → lowering`, на нём же встроенные `min`/`max`/`saturating_*`/SIMD (MAC-13).
- `api.rs` — один `PublicApi` для скалярного и Rust-пути. Адаптеры — обобщённые функции в `splice-core::native`, а не код на месте вызова (MAC-10, MAC-19).
- Рекомендуемая стратегия типов (MAC-07) — `Reg<T>`-хэндлы и вывод через rustc. Минимальная унификация в нормализаторе остаётся запасным вариантом.

**Что удаляется:**
- шесть дублирующихся списков примитивов;
- `flow::key` по `ptr as usize`, метки `'__splice_source_{адрес}`, `NEXT: AtomicUsize`;
- `HashMap<String, Binding>`, сравнение file/line/col в `host_kind`;
- `body.contains("break '__splice_iteration")`;
- `RebasePaths` и перенос тела в `mod <fn>`;
- тройное понижение тела (`__splice_inline`/`eval`/`print`), если `print` генерируется по запросу.

### T4. Корректность и гигиена кода, который генерирует макрос
**ID:** MAC-03, MAC-04, MAC-05, MAC-15, MAC-22, MAC-23, MAC-26 (≈ CORE-25), SPL-07 (жёсткий `::splice`), SPL-20/CORE-08 (`cfg!(debug_assertions)` не того крейта).

**Корень.** Генератор исполняет нативную арифметику Rust с паникой. Пути пишутся неквалифицированно. Пользовательские типы переносятся в дочерний `mod`. Режим переполнения угадывается по argv rustc. Пути к файлам склеиваются из `CARGO_MANIFEST_DIR` и cwd.

**Целевой дизайн.**
- Трейт `HostArith` (`checked_*` для всех примитивов) с отложенной ловушкой.
- Только полные пути: `::core::result::Result::Ok`, `::std::format!`.
- Алиасы `type __SpliceP0 = …;` вне `mod`, все сгенерированные элементы с префиксом `__Splice`.
- Корень крейта задаётся через `#[kernel(crate = path)]`/`#[splice(crate = …)]` единообразно для всех derive.
- Режим переполнения: `splice_core::overflow_checks()` — проба в крейте пользователя, генерируется макросом и кэшируется в `OnceLock`.
- Путь к файлу — `Span::local_file()`.
- Режим `BuildOptions` всегда передаётся явно, `impl Default` удаляется.

**Что удаляется:** `overflow.rs` (разбор argv), эвристика поиска файла по предкам в `splice-jit/dwarf.rs`, `faulting_i32_arithmetic`-развилка.

### T5. Контракт splice-бэкенда слишком узок: вызов, ловушки, Rust-вызовы, верификатор, память и debug-слоты ×3
**ID:** CORE-02, CORE-05, CORE-06, CORE-07, CORE-08, CORE-10, CORE-11, CORE-16, CORE-19, CORE-24, CORE-04, BE-07, BE-08, BE-10, BE-14, BE-16, BE-17, BE-21, MAC-19, SPL-20.

**Корень.**
- Ядро отдаёт бэкенду высокоуровневые операции и не держит общих примитивов.
- Методы `KernelBuilder` инфаллибельны, поэтому каждый бэкенд проверяет поток сам.
- Общего `replay` нет, поэтому у LLVM два полных понижения.
- Статусы кодируются тремя способами: `1..5`, `-1..-5`, `5<<32`.

**Целевой дизайн (`splice-core/src/`):**
- `abi.rs` (вместе с `splice_target::abi`, объединить в один модуль):
  - `#[repr(u32)] enum Trap { Bounds=1, DivZero, DivOverflow, Overflow, RustPanic, Abi }`;
  - `#[repr(C)] struct SliceDescriptor { data: *mut u8, len: u64 }`;
  - `const fn status(Trap) -> i64` и `fn decode_status(i64) -> Result<u64, Trap>`;
  - единый ABI Rust-вызова `unsafe extern "C" fn(ctx: *mut u8, op: u32, packet: *const u8) -> i64`.
- `backend.rs` — новый трейт бэкенда из трёх точек:
  - `trait Emitter: KernelBuilder` (все методы `-> Result<_, BuildError>`);
  - `fn finish(self) -> Result<Code>`;
  - `unsafe fn enter(&Code, words: *const u64, slices: *const SliceDescriptor) -> i64`.
- `validate.rs` — `struct Validating<B>`: потоковая проверка типов, буферов, меток и вызовов. Определённая инициализация считается по CFG-скелету. Прямой путь всегда идёт через него. `hir/verify.rs` использует те же `check_*`.
- `replay.rs` — `fn replay(&Function, &mut dyn KernelBuilder)`. Переносимый путь каждого бэкенда = `replay` поверх прямого билдера.
- `codegen.rs` — у `Codegen` два метода: `compile(&Function, BuildOptions)` и `stage(sig, body, BuildOptions)` с дефолтом `compile(&materialize(..))`.
- `call.rs` — обобщённый `call`: проверка сигнатуры один раз (через `TypeId`), дескрипторы на стеке (`SmallVec<[_;8]>`), декодирование статуса, текст паники Rust-импорта из thread-local слота.
- `legalize/memory.rs` — `Store/StoreSelect/Scatter*/CompressStore` раскрываются в `bounds_mask` → `first_fault_prefix` → `masked_store` → `trap_if(any, Bounds)`. Правило «u16-индексы при len > 65535» и порядок записи scatter задаются здесь.
- `legalize/rust_call.rs` — `rust_call_plan(..) -> RustCallPlan { entry, layout, tags }`. Одна инструкция `Instr::RustCall { target: RustTarget::{Static, Context{..}}, words, slices }`.
- `debug_slots.rs` — `Point::stores(&Function) -> SlotStore { offset, source, convert: None|SextI64|MaskToLanes|BoolToByte }`.
- `hir/simd_info.rs` — `const fn SimdOp::info() -> OpInfo { name, operands, output, memory, elements }`. Из этой таблицы выводятся `validate`, `output`, имя в листинге и `may_trap`.
- `splice-jit`: `Published<M> { registration, memory }` с правильным порядком `Drop` и builder `Publish::new(prefix).function(..).frame(..).finish()`.
- `immediate_condition` удаляется: известность условия передаётся как `Word::{Known, Runtime}`.

**Что удаляется:**
- в LLVM `lower/body.rs` (≈700 строк) или `direct.rs` (рекомендую оставить прямой билдер и удалить `body.rs`), `direct/flow.rs`, `capture`-развилки;
- в Lancy `lower/portable.rs`, проверки и `error: Option<String>` в `lower/mod.rs`/`seal.rs`, `trap_bounds = BoundsCheck(0,0)`, ветвление короткой записи через пользовательские метки, кэш `shared_loads` (переносится в Lancy, см. T7);
- в Cranelift холодный путь `ordered(check_lanes)` и собственная сборка пакета;
- `Dispatcher::Scalars`, `RuntimeContext::new`, `check_context_call`, `RustCall.arity`;
- три `debug_info.rs`-копии публикации и три копии кодов и строк ловушек;
- `immediate`/`literal`-поля бэкендов.

### T6. Знание об операциях размазано по `match`-ам (lancy и splice)
**ID:** HIR-02, HIR-10, HIR-14, HIR-16, HIR-22, HIR-24, X64-05, X64-09, X64-11, X64-15, X64-21, A64-11, A64-12, A64-18, CORE-07, CORE-10, CORE-19, LNC-07.

**Корень.** Нет единого описания операции или инструкции. Тип результата, эффекты, операнды, планирование, кодирование, листинг и парсер — независимые `match`, связанные позиционно (аллокации в эмиттере по индексу `operands()`), а на `_ => Barrier` новая инструкция тихо получает неверную классификацию.

**Целевой дизайн.**
- **Lancy HIR** (`hir/ssa/inst/info.rs`):
  - `impl Op { fn info(&self) -> OpInfo; fn operands(..) /* включая side-table Call/PacketCall */; fn operands_mut(..); fn result_type(..) -> Result<Option<Type>, IrError> }`;
  - `Effects { reads, writes, trap: Option<TrapKind{DivZero,LaneIndex,Bounds,Overflow}>, debug, call }`;
  - один `Literal` (или канонизация `Literal(I32)` → `Const` и отказ от бит выше ширины);
  - `Fact::Uniform` вместо неявного «Constant для векторов»;
  - второй шаг — общие `Op::Bin/Unary/…` для скаляра и вектора, в `Simd` остаются только чисто векторные операции.
- **LIR** (`lir/inst_macro.rs`):
  - `define_inst!` генерирует вариант, `operands()`, именованную структуру аллокаций (`AluAllocs { dst, lhs, rhs }`), временные регистры как `reg_temp`, обязательный класс планирования (без `_`) и стабильный мнемонический листинг;
  - для x64 и aarch64 один и тот же макрос;
  - поддержка векторной операции — таблица `Form { supported: fn(&Target)->bool, emit }`, `select` = «нашлась форма»;
  - классификация k-масок — одна функция `MaskForm`.
- **AArch64-кодировщик:** декларативная таблица (`macro_rules!`) `Code = база, Form, feature` генерирует enum, `required_feature`, `lower` и арность. Псевдоформы с зашитыми операндами уходят, `Ldr_64` получает собственную базу `0xf9400000`, суффиксный API (`Saddl`) удаляется.
- **Идиомы:** `Simd::PrefixMask`, `Simd::MaskPopcount`, `Simd::CompressStore` становятся каноническими HIR-операциями. Их канонизирует `simplify`, isel сопоставляет одну операцию.
- **Текст:** `hir::text` — `Display` и `parse` в одном синтаксисе, имена из той же таблицы (LNC-07). Для LIR — мнемонический принтер.

**Что удаляется:**
- пары `Op::Bin(..) | Op::Simd(Simd::Bin(..))` в каждом проходе;
- `new_instruction` с `unwrap_or(Type::I32)`;
- магические `reg(2 + usize::from(..))`, `n(allocs.len()-1)`;
- ручные `clobbers()` для v17–v19;
- шаблонное распознавание `compact_store`/`prefix_mask` в isel;
- `Debug`-листинг в ≈870 строках фикстур;
- отдельные таблицы имён парсера.

### T7. Конвейер проходов и хранилище Lancy HIR
**ID:** HIR-03, HIR-04, HIR-05, HIR-06, HIR-07, HIR-08, HIR-09, HIR-11, HIR-12, HIR-17, HIR-18, HIR-19, HIR-20, HIR-25, LNC-17, LNC-18, BE-09.

**Корень.**
- `simplify_with_cfg` — монолит с ручной передачей `Option<DomTree>`, `Facts`, `Bounds`.
- Арена рёбер и операндов только растёт, мёртвые рёбра распознаются по эвристике.
- Метаданные срезов вычисляются арифметикой индексов `Value`.
- Идентичность функций — по адресу `Arc`.

**Целевой дизайн (`hir/transforms/`):**
- `pass.rs`: `trait Pass { fn run(&self, f, a: &mut Analyses) -> Preserved }`, `struct Analyses { cfg, dom, loops, facts, bounds }` с ленивым вычислением, `Preserved { cfg, values }`;
- конвейер задаётся списком `&[&dyn Pass]`, в `cfg(any(test, debug_assertions))` после каждого прохода вызывается `f.verify()`;
- `limits.rs`: `struct Limits` с обоснованием каждой константы;
- `Function::compact()` после структурных проходов; либо рёбра принадлежат терминатору;
- `Vec<SliceParam{ptr,len,elem,writable}>` и `DenseMap<Value, u32>`, `Function::param_index`;
- `DenseMap<FunctionId, Arc<Function>>`, call-сайты один раз переписываются в `FunctionId`;
- `loops::run` применяет все группы преобразований за раунд к непересекающимся циклам;
- после циклов повторяется дешёвая фаза SCCP + rewrite + copy-prop, плюс локальная нумерация значений (VN) по доминаторному дереву;
- CSE загрузок из неизменяемых срезов делается в Lancy (BE-09).

**Что удаляется:**
- эвристика `span.len == params.len()`;
- `binary_search` по ID параметров;
- `cfg!(feature="experimental-simd")` в стабильном `bounds.rs`;
- O(n)-верификация на каждый публичный `replace_op`;
- `FunctionIds<*const T>` во всех стадиях;
- кэш загрузок в splice-lancy.

### T8. Три копии численной семантики и медленные fallback-и (x64 `Lanes`, aarch64 ручной IEEE, `hir::eval`)
**ID:** X64-04, A64-03, A64-14, HIR-21, X64-01, A64-01, A64-02, A64-15.

**Корень.** x64 держит ≈1600 строк поэлементного ассемблера (`inst/lanes*`), aarch64 ≈1000 строк ручного IEEE (`remainder.rs`, `special.rs`, `half.rs`, `integer_float.rs`), а `hir::eval` — эталон, который в продакшене используется только как libcall aarch64. Поддержку держат только дифференциальные тесты, а для горячих операций (min/max, f16, скаляры вне i32) нет нативных форм.

**Целевой дизайн.**
- `lancy/src/support/numeric/` — численная семантика, вынесенная из `hir/eval.rs` и `hir/integer.rs`. От неё зависят HIR (интерпретатор, свёртки) и fallback.
- `lancy/src/lir/fallback.rs` — общий `Site`/`primitive` для обоих бэкендов: `extern "C"` → `numeric::evaluate`, внутренняя ошибка возвращается кодом `Internal`, а не через `unreachable!` и abort.
- Нативные формы для горячего: скалярные `cmp/setcc/cmov/ucomis/min/max/shift/div` всех ширин на x64; `fcmeq+bif+fminnm`, SVE `.H`, `fcvtl/fcvtn`, `bsl`-copysign и классификация на aarch64.
- Холодное (fmod, nextup f16, трансцендентные) уходит в общий fallback.

**Что удаляется:** `lir/x64/inst/lanes.rs`, `lanes/{integer,float,convert}.rs` (≈1600 строк), основная часть `aarch64/backend/{remainder,special,half,integer_float}.rs`, aarch64-привязка `hir::eval` под `cfg(target_arch)`.

### T9. Дублирование между x64 и aarch64 в общем LIR
**ID:** X64-10, A64-13, X64-12, X64-18, A64-05, X64-13, X64-14.

**Корень.** Расчёт кадра, межпроцедурный `written`, раскладка аргументов `Call`, покрытие соседней загрузки (`#[cfg(target_arch)]` внутри target-independent `Context`) написаны в каждом бэкенде отдельно. Модуль aarch64 компилируется только на aarch64.

**Целевой дизайн.**
- `lir/frame.rs`: `FrameRequirements::scan(body)` через `TargetInst::frame_use()`, все суммы через `checked_add`.
- `lir/interprocedural.rs`: `written(body, alloc, allocatable)`.
- `lir/args.rs`: раскладка слов по 8 байт.
- `Context::cover(value)` без `cfg` по архитектуре.
- `BlockBuilder` вместо неявного протокола `edge()`/`end_block()`.
- Оба бэкенда компилируются на всех хостах, `cfg` остаётся только у исполнения и `jit::memory`.

**Что удаляется:** дублирующие куски `x64/compile.rs` и `aarch64/backend/compile.rs`, двойной `reverse()`, мёртвый в stable `previous/covered`.

### T10. Цель — глобальное состояние процесса, ISA переводится трижды
**ID:** CORE-12, X64-06, X64-03, BE-04, BE-13, LNC-16, A64-05, A64-13(б), CORE-26, CORE-27, HIR-21(в), HIR-23, SPL-05, BE-11 (кэш `TargetMachine`).

**Корень.** `SPLICE_SIMD` + `OnceLock` + `Features::host()` (≈40 вызовов, включая `operands()`), `is_x86_feature_detected!` в обход профиля, ABI через `cfg!(windows)`. Каждый бэкенд сам переводит профиль в ISA.

**Целевой дизайн (`splice-target/src/`):**
- `pub struct Target { arch, abi: Abi, vector_bytes, features: Features::{X86(X86Features), Arm64(Arm64Features)} }`;
- `X86Features` дополняются `bmi2`, `popcnt`, `lzcnt`; `Arm64Features` — `fp16`, `bf16`, `sve`, `sve2`;
- `Target::host()` — значение по умолчанию (учитывает `SPLICE_SIMD`), `pub const MIN/MAX_VECTOR_BYTES`;
- `Abi { argv, rust_args, volatile, saved, vec_scratch, shadow, saved_vec_bytes }` с `const SYSV`/`WIN64`/`AAPCS64`;
- lancy: `CompileOptions { target, inlining, debug }`, цель передаётся через `Rules`, `Emitter`, `CompiledFunction` и `Function<Inst>` (`spillslot_size`);
- splice: `BuildOptions { mode, target }`, `Function` хранит цель, под которую её стейджили;
- бэкенды только переводят `Target`: `llvm::feature_string(&Target)`, `cranelift::flags(&Target)`, `lancy::Target::from(&Target)`.

**Что удаляется:** `Features::host()` в эмиттере и `operands()`, `is_x86_feature_detected!` в isel, 12 веток `cfg!(windows)`, дубли `Features.bytes`/`Simd.bytes`/`vector_bytes()`, повторные `get_host_cpu_name`.

### T11. Строковые и непрозрачные ошибки
**ID:** CORE-03, LNC-05, LNC-15, SPL-19, HIR-15, HIR-22, A64-14, BE-18, CORE-17, CORE-20, LNC-11.

**Корень.**
- В splice везде `Result<T, String>`, `From` в обе стороны между фазами.
- В lancy `CompileError` без `#[non_exhaustive]`, с cfg-вариантом `Assembly(IcedError)` и внешними типами в API.
- Тексты ловушек дублируются вручную.

**Целевой дизайн.**
- lancy:
  - `#[non_exhaustive] enum CompileError { InvalidIr(IrError), Unsupported(Limit), Os{op,source}, Internal(InternalError) }`, где `Limit = Registers|BranchRange|Workspace|CodeSize`;
  - `#[non_exhaustive] enum Trap`, `CallError { Signature, VectorLength, Trap(Trap) }`;
  - fallback инлайнинга по `e.is_limit()`.
- splice-core:
  - `CallError::{Trap(Trap), RustPanic(String), Rust(String), Signature(..)}`, `BuildError::{Unsupported{..}, Verify{instr, reason}, Backend(Box<dyn Error>)}`;
  - без `From<$name> for String`;
  - `const fn` в `ContextSignature` паникует с причиной (ошибка компиляции).
- Тесты сравнивают варианты, а не строки.

**Что удаляется:** таблица `CallError → String` в filetests, 32 строковых сравнения в тестах, сентинел `Reg(u32::MAX)`, `phase_error!`-конверсии.

### T12. Тестовая инфраструктура не покрывает матрицу «бэкенд × путь × режим × профиль × ОС»
**ID:** SPL-02, SPL-03, SPL-04, SPL-14, SPL-15, SPL-17, SPL-18, SPL-19, BE-15, LNC-01, LNC-06, LNC-09, LNC-10, LNC-11, LNC-15, A64-04, A64-16, X64-13, X64-19, HIR-09, MAC-17, CORE-13, CORE-14.

**Корень.**
- ≈60 локальных `check::<B>` с перекосом: Lancy 50, Cranelift 18, Text 0.
- Тесты, не зависящие от архитектуры, лежат в `lir/x64/tests`.
- Нет межбэкендного дифференциального теста, `trybuild`, парсера splice-text и тестов `splice-jit`.
- Харнесс filetests исполняет не продакшен-конвейер.

**Целевой дизайн.**
- `splice-testkit` (`publish = false`):
  - `enum Backend { Lancy, Cranelift, Llvm, LlvmManualSimd }` под фичами;
  - `for_each(spec, |cfg, exe| …)` по {backend} × {build, compile(ir)} × {Debug, Release} × {Target-профили в одном процессе после T10};
  - `assert_lowered`, сравнение буферов после ошибки, генератор «операция × элемент × граничные данные» (перенос `zz_diff.rs` из копии ревьюера `l19`; в репозитории его нет, его нужно воссоздать).
- `lancy/tests/portable/`: traps, pointers, control_flow, integer_runtests, generated, numeric_coverage со «злыми» входами и сравнением `Result`; ABI-проба для x64 (iced) и aarch64 (`aarch64::Encoder`); многопоточный тест `Kernel`.
- filetests — отдельная цель `[[test]] harness = false` с обходом каталога в рантайме.
- Бэкенды отдают снимки стадий через `sink`, непотреблённые директивы считаются ошибкой, `@run` принимает типизированные литералы и несколько буферов.
- `trybuild` ui-тесты на все `Error::new_spanned` и на гигиену.
- `.splice`-golden-тесты после парсера (CORE-13).
- `splice-jit`: ELF/DWARF через `object`/`gimli`, многопоточный реестр.
- Golden-таблица кодировок aarch64 (llvm-mc), исчерпывающий `logical_immediate`, рандомизированные тесты планировщика и liveness.

**Что удаляется:** ≈60 локальных `check`, `compile_all!` в `simd.rs`, `check_backends`, `Builders { lancy, llvm, cranelift }`, x64-`selected()`, `build.rs` lancy, две обёртки «исполнить сырые байты», магические маски `@aarch64 … 0xfffffc1f`.

### T13. CI, сборка, версии LLVM, упаковка
**ID:** SPL-01, SPL-11, SPL-08, SPL-07, SPL-13, SPL-16, SPL-21, LNC-08, LNC-10, BE-03, BE-05 (= CORE-15), BE-06, X64-03 (Windows-job).

**Корень.**
- Команды размножены по CI, AGENTS.md и README.
- Нет Windows, debug-режима, матрицы фич, `cargo doc`, `publish --dry-run` и прогона со статическим LLVM.
- Две версии LLVM (19 на Windows, 21 на Linux).
- Две библиотеки с именем `splice`.

**Целевой дизайн.**
- `xtask` (`cargo xtask ci|bench|doc`) — единственный источник команд для CI, AGENTS.md и README.
- Матрица {ubuntu-x64, ubuntu-arm, windows-x64} × {release, debug}.
- Профильная джоба: все бэкенды при фиксированном `SPLICE_SIMD` (после T10 — в одном процессе).
- Отдельные шаги:
  - `--features experimental-simd`, `--all-features`, `--no-default-features`;
  - `cargo doc --workspace --no-deps -D warnings`;
  - `cargo publish --workspace --dry-run`;
  - сборка со статическим LLVM;
  - `bench -- --test`.
- `[workspace.package]`, `[workspace.dependencies]`, `[workspace.lints]` (`missing_docs`, `unsafe_op_in_unsafe_fn`, `clippy::undocumented_unsafe_blocks`).
- Библиотека `splice-core` переименовывается в `splice_core`.
- Одна закреплённая версия LLVM.

**Что удаляется:** `build.rs` lancy, `capture_attribute_kind` и аналогичные развилки по версии, `.codex/config.toml` из git, `codex.md` (сливается в AGENTS.md).

### T14. Производительность кодогенерации и рантайма
**ID:** X64-01, X64-02, X64-07, X64-08, X64-16, X64-17, X64-20, A64-01, A64-02, A64-06, A64-07, A64-08, A64-09, A64-10, A64-15, A64-17, LNC-04, LNC-13, LNC-17, CORE-11, CORE-21, CORE-27, BE-11, BE-12, MAC-10, MAC-14, HIR-07, HIR-08, HIR-18, HIR-19, SPL-18.

**Корень.** Регистровый файл используется не полностью: xmm16–31 и v8–v15 простаивают, x13–x15 зарезервированы. Скаляры спиллятся на полную ширину вектора. Workspace выровнен на 8 байт и выделяется `Vec` на каждый вызов. JIT-инфраструктура (LLJIT, ISA, mmap) создаётся на каждое ядро.

**Целевой дизайн.** Описан по шагам в Ф8. Каждый шаг идёт после структурного рефакторинга соответствующего слоя, чтобы оптимизация не добавляла дублей. Каждый подтверждается замером.

### T15. Публичная поверхность, unsafe/FFI, документация
**ID:** LNC-02, LNC-03, LNC-05, LNC-12, LNC-14, SPL-06, SPL-09, SPL-10, SPL-12, CORE-09, CORE-20, CORE-22, CORE-23, CORE-24, CORE-25, BE-19, BE-20, BE-22, A64-17, HIR-15, CLIP-01…03.

**Корень.** Публичный API сложился из того, что понадобилось тестам. Документация описывает желаемое как реализованное. В FFI нет `SAFETY:`, константы ОС переписаны вручную, у JIT-кода нет unwind-информации.

**Целевой дизайн.**
- lancy: `lancy::{Kernel, CompileOptions, CompileError, Trap, CallError}`, `lancy::hir::{Function, Op, Type, Value, Block, IrError, text}`, `lancy::debug::{FunctionInfo, FrameInfo}` с геттерами.
  - Вызов: `unsafe fn call_packed(&self, ws: &mut Workspace, args, result)` и безопасный `typed::<A, R>()`.
  - AArch64-энкодер, `slot_key!`, `ControlFlowGraph`/`DomTree` становятся `pub(crate)` (или прячутся за feature).
  - `#![warn(missing_docs, unreachable_pub)]`.
- splice-kernel:
  - явный список реэкспортов, бэкенды модулями `splice::{lancy, llvm, cranelift, text}`;
  - `#[doc(hidden)] pub mod __private` для кода, который генерирует макрос;
  - запечатанные конструкторы `TypedIr`/`TypedKernel`;
  - README и docs подключаются как доктесты, спецификация делится на «Реализовано (ссылка на тест)» и «Планируется».
- FFI: `libc`/`windows-sys`, `// SAFETY:` у каждого блока, регистрация unwind-таблиц (`RtlAddFunctionTable`, `__register_frame`).

**Что удаляется:** `pub use splice_core::*`, `pub use lir::aarch64`, `#[macro_export] slot_key!`, дубли путей `hir::X`/`hir::ssa::X`, `workspace_words`, `call_words`/`call_raw` (поглощаются `call_packed`), исторические таблицы в README.

### T16. Отладочная информация, JIT-регистрация, ABI корня
**ID:** CORE-15/BE-05, CORE-16, CORE-18, CORE-23, CORE-24, CORE-25, MAC-26, LNC-14, A64-17, X64-20, HIR-13.

Темы пересекаются с T4, T5, T13 и T15, там же даны исправления. Сюда сведены ошибки отладочного режима, который CI не исполняет (SPL-03):
- `bool` показывается в GDB как 255;
- Debug-чтения ломают проверку определённой инициализации;
- perf-map с переиспользуемыми адресами;
- неверный путь к исходнику в DWARF;
- конфликт символов `__jit_debug_*` с libLLVM;
- нет frame record x29 на aarch64;
- `nop` от `DebugPoint` в release.

---

## План исправлений

Обозначения:
- **[∥]** — шаг можно делать параллельно с соседними шагами той же фазы (разные крейты или модули);
- **[риск]** — большой объём или высокая вероятность регрессий; делать отдельной веткой с полным прогоном матрицы;
- **Проверка** — команда или тест, которые доказывают готовность.

Базовая проверка в конце каждого шага:
```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --release
```
После Ф1 вместо неё: `cargo xtask ci --quick`.

Зависимости фаз:
```
Ф0 → Ф1 → Ф2 → Ф3 ─┬─► Ф4 (splice-core) ──┬─► Ф7 (макросы; Ф7.1 можно начинать параллельно с Ф4)
                   └─► Ф5 (lancy HIR) ─► Ф6 (lancy LIR)
                       Ф5.4 ─► Ф4.6 (Op::Trap нужен легализации)
Ф5+Ф6 ─► Ф8.1–8.5;  Ф4 ─► Ф8.6;  всё ─► Ф9
```

### Ф0. Подтверждённые ошибки и быстрые исправления
Каждый шаг начинается с регрессионного теста, который падает до исправления.

**0.1 Починить CI после переименования.** ID: SPL-01, SPL-16(п.4), SPL-21.
- Файлы: `.github/workflows/simd.yml` (строки 39, 78, 90: `-p splice` → `-p splice-kernel`), `AGENTS.md:19`, `.gitignore` (`target/`, `tests/fixtures/**/Cargo.lock`).
- Проверка: `cargo pkgid splice-kernel`; `grep -rn -- '-p splice\b' .github AGENTS.md README.md` пусто; обе джобы зелёные, в том числе шаг QEMU-SVE. Ожидайте, что после возврата профилей вскроются BE-01 и, возможно, другие ошибки.

**0.2 BE-01 (crit): Cranelift SIGSEGV на `sse`.** [∥]
- Файлы: `splice-cranelift/src/lib.rs`, `lower/simd.rs`.
- Обход: невыровненную векторную загрузку не вливать в операнд legacy-SSE (отдельное значение с барьером или выровненные флаги только там, где выравнивание доказано). Собрать минимальный CLIF-репро и отправить апстрим.
- Тест: `splice-cranelift/tests/sse_unaligned.rs` — ядро `Load(F32,buf,0); Load(F32,buf,1); Eq; Count`.
- Проверка: `SPLICE_SIMD=sse cargo test -p splice-cranelift --release`.

**0.3 BE-02: Cranelift `Cast f64→i64`.** [∥]
- Файл: `splice-cranelift/src/lower/simd.rs:564-579`. Поэлементно: `extractlane` → `fcvt_to_sint_sat.i64` → `insertlane`.
- Тест: «все допустимые `Cast/Widen/Narrow` × 4 бэкенда × 2 пути» с NaN и ±inf (зачаток харнесса Ф1.5).
- Проверка: тест зелёный во всех профилях.

**0.4 HIR-01 + HIR-13: scratch и debug-workspace при инлайнинге.** [∥]
- Файлы: `lancy/src/hir/ssa/inst/mod.rs` (`Op::CallScratch { offset, bytes }`), `hir/transforms/passes/inline.rs:265-283` (сдвиг смещений callee на высоту scratch вызывающей функции; так же для `DebugStore`), x64 и aarch64 isel/frame (учёт `offset`), `memory.rs` (must-alias по смещению).
- Тесты: `filetests/passes/inline-call-scratch.lancy` (`@run` для `Always` и `Never` даёт 719), такой же с `DebugStore`; Rust-тест из отчёта (1919 против 719).
- Проверка: `cargo test -p lancy --release`, тест падает на старом коде.

**0.5 MAC-02 (быстрый фикс): `usize`-арифметика в составном присваивании.** [∥]
- Файл: `splice-macros/src/scalar/lower/mod.rs:771-787` (применить проверку `Index` из `binary()` в `assignment()`). Полное исправление — Ф7.2.
- Тест: `splice/tests/language.rs` — `let mut i = xs.len(); i -= 1;` на пустом срезе; `i *= 2` при `i ≥ 2^30`; debug и release; JIT обязан совпасть с Rust (ошибка или ловушка, но не `Ok(1)`).
- Проверка: тест; trybuild-сообщение, если конструкция отвергается.

**0.6 MAC-03: отложенные ошибки host-арифметики для всех примитивов.**
- Файлы: `splice-macros/src/scalar/lower/{host,comptime}.rs`, `splice-core/src/frontend/scalar.rs` (трейт `HostArith` с `checked_*`).
- Тест: `host_div(#[comptime] d: usize = 0, flag)`: `build` Ok, `call(flag=false)` Ok, `call(flag=true)` даёт ловушку деления; `comptime!` с переполнением возвращает `Err`, а не панику; прогон с `panic=abort` (отдельный тест-бинарь).
- Проверка: `cargo test -p splice-kernel --test overflow` в debug и release.

**0.7 MAC-04, MAC-05, MAC-22, MAC-23: гигиена.**
- Файлы: `splice-macros/src/{import.rs, scalar/mod.rs, scalar/abi.rs, scalar/methods.rs, declaration.rs}` и все `emit!` с `Ok/Err/Some/Option/String/format!`.
- Тесты `splice/tests/hygiene.rs`:
  - пользовательский `struct Signature`;
  - `enum Status { Ok, Err }` + `use Status::*`;
  - модуль с `#![no_implicit_prelude]`;
  - `mod foo` рядом с `#[kernel] fn foo`;
  - `comptime!(self.n + 1)` в `#[kernel] impl`;
  - `use super::x;` в теле;
  - `#[allow(unused)] x: i32`.
- Проверка: тесты компилируются и проходят.

**0.8 CORE-04: убрать `immediate_condition`.** [риск: средний]
- Файлы: `splice-core/src/frontend/mod.rs:59`, `hir/builder.rs:123-138,299-301`, `splice-lancy/src/lower/mod.rs` (поле `immediate` и все сбросы), `splice-llvm/src/lower/direct.rs:301-303`, `splice-macros/src/scalar/lower/control.rs:145`.
- Макрос передаёт `Word::{Known, Runtime}`. Константное условие в `branch` бэкенд может свернуть сам, без влияния на staging.
- Тест: ядро со смешанным по стадиям `if`, где последняя инструкция перед условием — `Const` в другой регистр и runtime-запись в регистр условия; листинг `Text` совпадает со структурой, которую исполняют Lancy и LLVM.
- Проверка: тест плюс `rg immediate_condition` пусто.

**0.9 BE-05 = CORE-15: дубли `__jit_debug_*` с libLLVM.**
- Файлы: `splice-jit/src/registry.rs`, `splice-jit/Cargo.toml` (`links = "splice_jit_gdb"`), `splice-llvm`.
- Вариант по решению Р11: при подключённом LLVM не определять символы, а объявить их `extern` и использовать определения LLVM под своим замком. Требование «не включать GDB-listener ORC» задокументировать.
- Проверка: CI-джоба со статическим LLVM (Ф1.6) линкуется; `nm -D` показывает одно определение.

**0.10 Мелкие ошибки, подтверждённые по исходникам (пакет, у каждой свой тест).** [∥]
- CORE-16 (`bool` в debug-слоте: сохранять `value & 1`; DWARF `bool` 1 байт). Тест: разбор DWARF из Ф1.5 или gdb-тест под `#[ignore]`.
- CORE-18 (Debug-чтения не учитывать в `check_initialization`). Тест: переменная не инициализирована на одном пути, Debug-сборка ок.
- CORE-19 (`may_trap` для SIMD-операций с памятью; `needs_status` только по `may_trap`).
- CORE-23 (perf-map: `O_NOFOLLOW`, усечение своего файла, одно предупреждение).
- CORE-25 (`e_machine`: `compile_error!` для неподдерживаемых архитектур).
- CORE-26 (`Auto` выбирает 32 байта только при AVX2; `Neon` для aarch64 вне Linux).
- BE-17 (явный флаг `root`), BE-18 (`intrinsic(..) -> Result`).
- LNC-05(ж) (fallback инлайнинга по категории лимита, включая `WorkspaceTooLarge`/`FrameOverflow`).
- X64-20 (не выдавать `nop` на `DebugPoint` вне debug).
- HIR-20 (ветка `Select→Min/Max` не должна пропускать обычное упрощение `Select`).
- SPL-17 (коды ошибок в `compile_fail`-доктестах).
- Проверка: базовая.

### Ф1. Страховочная сетка: явный путь понижения, харнессы, CI
Без этой фазы рефакторинги Ф4–Ф7 небезопасны.

**1.1 Явный путь понижения.** ID: MAC-01, MAC-08, SPL-04(а,б). [риск: может вскрыть скрытые откаты]
- Файлы: `splice-macros/src/import.rs:39-49,462-471`, `scalar/methods.rs`, `scalar/methods/declaration.rs`, `splice-core/src/typed.rs` (`Lowering`, `TypedIr::lowering()`), `splice/tests/rust_regions.rs` (пометить шесть намеренных ядер `#[kernel(rust)]`).
- `#[kernel]` без возможности понижения даёт `compile_error!` с исходной ошибкой. «Borrowed comptime → Rust import» тоже становится ошибкой компиляции.
- Проверка:
  - trybuild: метка цикла и `i64`-параметр дают понятную ошибку со span;
  - тест-сторож: множество ядер с `Lowering::RustImport` равно множеству помеченных;
  - сборка всех примеров и бенчмарков.

**1.2 Переносимый набор тестов lancy.** ID: LNC-01, A64-04. [∥]
- Перенести `lir/x64/tests/{traps,pointers,control_flow,integer_runtests,generated,numeric_coverage*}` в `lancy/tests/portable/` (или `src/jit/tests/`, если нужны `pub(crate)`-хуки) без `cfg(target_arch)`.
- В `numeric_coverage` добавить «злые» входы (NaN, sNaN, ±0, ±inf, субнормальные, MIN, −1, сдвиги ≥ ширины, деление на 0 и MIN/−1) и сравнение `Result` (вид ловушки).
- ABI-проба callee-saved для aarch64 (x19–x28, d8–d15) на нормальном выходе и при ловушке; многопоточный тест `ScalarKernel`; тест `EmptyCode`.
- Проверка: `cargo test -p lancy --release` на x86 (sse/avx2/auto) и под `qemu-aarch64 -cpu max` (neon, sve VL16, sve2 VL32/64).

**1.3 Инварианты lancy.** ID: HIR-09, X64-13, X64-19, A64-16. [∥]
- `f.verify().expect(pass)` после каждого прохода в `cfg(any(test, debug_assertions))`.
- Юнит- и `.lancy`-тесты из HIR-09: `witnesses::index`, SCCP с uniform-векторами, рекуррентность при `trip ≥ 2^31`, `call_scratch`/`debug_store` с инлайнингом.
- Рандомизированные тесты планировщика (частичный порядок Ordered/Barrier) и `Liveness` против наивного dataflow.
- Тестовый `Native::emit` всегда с пулом констант.
- Golden-таблица `(Code, операнды) → u32` из `llvm-mc -show-encoding` и тест «каждый `Code` в таблице»; исчерпывающий roundtrip `logical_immediate`.
- Проверка: `cargo test -p lancy --release`.

**1.4 Харнесс filetests.** ID: LNC-10, LNC-06(г), LNC-09, LNC-11, LNC-15. [∥]
- `lancy/Cargo.toml`: `[[test]] name="filetests" harness=false`, обход каталога в рантайме, удалить `build.rs`. Доступ к стадиям через `#[doc(hidden)] pub mod __testing` или feature `testing`.
- В конце `execute` выдавать ошибку на непотреблённые директивы.
- `@run` с типизированными литералами (`1.5f32`, `vec.i32[..]`, `-> 2.5f64`) и несколькими буферами.
- `InterpretError { Trap(CallError), OutOfBuffer, Unsupported }`.
- `@aarch64 has/lacks <мнемоника>` с поиском только в `text_len`.
- Обновить `filetests/README.md`.
- Проверка: `cargo test -p lancy --test filetests`; `cargo package -p lancy` без `filetests/` собирается.

**1.5 `splice-testkit` и дифференциальный тест.** ID: SPL-02, SPL-03, SPL-14, SPL-15, SPL-18, BE-15, MAC-17, CORE-14, SPL-17.
- Крейт `splice-testkit` (`publish = false`): матрица {Lancy, Cranelift, Llvm, LlvmManualSimd} × {build, compile(ir)} × {Debug, Release}, сверка с Rust-эталоном, между бэкендами и по состоянию буферов после ошибки.
- `assert_lowered` для каждого ядра, кроме `#[kernel(rust)]`.
- Генератор «операция × элемент × граничные данные»: воссоздать `zz_diff.rs`, сверять все 23 скалярные `Binary` с `checked_eval`, SIMD-приведения, сдвиги, gather/scatter у границ 32768/65535.
- Перевести все ≈60 `check::<B>` и `Builders` бенчмарков на этот крейт; `xorshift` вынести в `benches/common.rs`.
- `trybuild` ui-тесты на все `Error::new_spanned`.
- Свой `tracing::Layer` вместо подстрок формата.
- В `sql_plan` уменьшить глубину для LLVM.
- `splice-jit`: ELF через `object`, DWARF через `gimli`, многопоточный реестр; юнит-тесты `hir/verify.rs` на каждую ветку ошибки.
- Проверка: `cargo test -p splice-kernel --all-features` под `SPLICE_SIMD=sse|avx2|auto` (и `neon` на ARM) зелёный. Каждая новая находка харнесса исправляется в этом же шаге с регрессионным тестом.

**1.6 CI и сборка.** ID: SPL-11, SPL-08, SPL-07, LNC-08, X64-03 (джоба), BE-05 (джоба). [риск: SPL-07 механически трогает все бэкенды]
- `xtask`: команды `ci`, `ci --quick`, `bench`, `doc`; AGENTS.md и README ссылаются на них.
- Workflow:
  - матрица {ubuntu-x64, ubuntu-arm, windows-x64} × {release, debug};
  - профильная джоба: все бэкенды на каждом `SPLICE_SIMD`, список пакетов в одной переменной;
  - `--features experimental-simd` (тест и clippy), `--all-features`, `--no-default-features`;
  - сборка со статическим LLVM;
  - `cargo bench -p splice-kernel --bench backends -- --test`;
  - `concurrency`, `timeout-minutes`, кеш.
- SPL-07: `[lib] name = "splice_core"` для splice-core, бэкенды на `splice_core::`, затем шаг `cargo doc --workspace --no-deps` с `RUSTDOCFLAGS=-D warnings`.
- SPL-08: dev-бэкенды под фичами (`required-features`), тесты фасада `cranelift`/`llvm` исполняются.
- Проверка: все джобы зелёные; Windows-джоба исполняет `cargo test -p lancy`, в том числе ABI-пробу (см. 3.2).

### Ф2. Типизированные ошибки и общий `Trap`
**2.1 Общий `Trap` и декодирование статуса.** ID: CORE-02 (ловушки), BE-14.
- Файлы: `splice-target/src/abi.rs` → `splice_core::abi` (или наоборот, один модуль), три бэкенда (коды `1..5`/`-1..-5`/`5<<32` → `status(Trap)`), `splice-core/src/frontend/{scalar,native}.rs`, `runtime_context.rs`.
- Проверка: `rg '5 << 32|RUST_PANIC|-5' splice-*` пусто вне `abi`; тесты testkit сравнивают варианты `Trap`.

**2.2 Ошибки lancy.** ID: LNC-05 (а–е, з), LNC-15, HIR-15, HIR-22, A64-14. [∥ с 2.3]
- Файлы: `lancy/src/error.rs`, `jit/mod.rs`, `x64/emit.rs:279-284` (`Limit::CodeSize` вместо фиктивного `BranchOutOfRange`), `hir/error.rs` (варианты `ExpectedScalar`, `PointerCast`, `UseSimdForm`), `aarch64/backend/operations.rs` (`primitive` возвращает `Internal`, без abort).
- Публичные паникующие методы `Function` возвращают `Option`/`Result` или документируют `# Panics`.
- Проверка: исчерпывающий `match` по `CompileError` компилируется на x86 и aarch64; в публичном API нет `iced_x86`/`regalloc2`.

**2.3 Ошибки splice.** ID: CORE-03, CORE-17, CORE-20 (сентинел), SPL-19. [∥]
- Файлы: `splice-core/src/{lib.rs, typed.rs, runtime_context.rs, hir/builder.rs}`, `splice-target/src/abi.rs`, все тесты со строковыми сравнениями.
- Сообщение паники Rust-импорта сохраняется в thread-local слоте; отдельный `Trap::Abi` для неверной формы операции.
- `const fn with_borrows` паникует с причиной.
- Проверка: тест «паника в импорте → `CallError::RustPanic("текст")`»; `rg 'Result<[^>]*, String>' splice-core splice-lancy splice-llvm splice-cranelift` пусто.

### Ф3. Цель как значение
**3.1 `splice_target::Target` и `Abi`.** ID: CORE-12, CORE-27 (константы), HIR-21(в).
- Файлы: `splice-target/src/{lib,profile,abi}.rs`, `splice-core/src/{simd,debug}.rs`.
- Проверка: юнит-тест «`Target::from_profile(p)` для всех профилей»; `MIN/MAX_VECTOR_BYTES` используются в `Storage`, `Mask`, `bounds.rs` (`debug_assert`).

**3.2 lancy принимает цель параметром.** ID: X64-06, X64-03, A64-05, A64-13(б), LNC-16, HIR-23. [риск: ≈40 мест в x64]
- Файлы:
  - `lancy/src/jit/{mod,kernel}.rs` (`CompileOptions { target }`);
  - `lir/x64/{emit,compile,isel}.rs`, `inst/mod.rs` (`operands()`/`clobbers()` без хоста: нужное кладётся в `Inst` на isel), `isel/masks.rs`, `isel/rules/{memory,mask}.rs` (без `is_x86_feature_detected!`);
  - `lir/x64/inst/mod.rs:67-122` → `Abi::{SYSV, WIN64}`;
  - `lir/mod.rs`, `aarch64/mod.rs` (оба бэкенда собираются на всех хостах);
  - tracing target `lancy::compile`.
- Тесты:
  - JIT с `Abi::WIN64` вызывается на Linux через `extern "win64"`, проба порчи nonvolatile;
  - файлтесты `@aarch64-isel` исполняются на x86 и наоборот;
  - матрица профилей в одном процессе.
- Проверка: `cargo test -p lancy` на Linux и Windows; `rg 'Features::host\(\)|is_x86_feature_detected' lancy/src/lir` пусто.

**3.3 Бэкенды переводят одну цель.** ID: BE-04, BE-13. [∥ с 3.2]
- Файлы: `splice-llvm/src/target.rs` (полная строка признаков: `+avx2,+fma,+f16c`, `+avx512f,+bw,+vl,+dq,+vbmi2`), `splice-cranelift/src/lib.rs` (`has_fma`, `popcnt`, `lzcnt`, `bmi*`; `x86_compress`/`shuffle_bytes` по цели), `splice-lancy`.
- Проверка: снапшот-тест «профиль → набор признаков» для каждого бэкенда; `rg is_x86_feature_detected splice-cranelift splice-llvm` пусто.

**3.4 LLVM: одна версия и честный `LlvmManualSimd`.** ID: BE-03, BE-06, BE-18 (проверка minor). Зависит от решения Р3.
- Файлы: `splice-llvm/Cargo.toml`, `src/lib.rs` (`Llvm::version` сверяет major.minor), `src/lower.rs` (удалить `capture_attribute_kind`), `src/jit.rs:130-162` (явный текст конвейера `default<O3>` без `loop-vectorize`/`slp-vectorizer` или `-vectorize-loops=false -vectorize-slp=false`; `noimplicitfloat` удалить).
- Проверка: тест `manual_simd_disables_loop_vectorization_and_slp` плюс новый тест «memcpy-идиома по-прежнему оптимизируется»; CI на обеих ОС с одной версией.

### Ф4. splice-core: типизированные скаляры и единый контракт бэкендов
Делается параллельно с Ф5/Ф6 (другие крейты). Ф4.6 ждёт Ф5.4.

**4.1 `ScalarType`.** ID: CORE-01, CORE-09, CORE-16 (удаление `debug::Scalar`). [риск: меняет ABI и все бэкенды]
- Файлы:
  - `splice-core/src/hir/{simd,mod,verify,buffer}.rs`, `frontend/{mod,scalar}.rs`, `lib.rs`, `runtime_context.rs`, `debug.rs`;
  - `splice-target/src/abi.rs`, `splice-text`;
  - `splice-lancy/src/lower/*`, `splice-llvm/src/lower/*`, `splice-cranelift/src/lower*`;
  - `BufferArg` → структура `{data, len, kind, PhantomData}`.
- Проверка: testkit — все `ScalarType` × все `Binary` × граничные значения × 4 бэкенда × 2 пути против `checked_eval`; макрос на этом шаге продолжает эмитить `I32`/`Bool`/`Index` (совместимость до Ф7.2).

**4.2 `KernelBuilder -> Result` и `Validating<B>`.** ID: CORE-06.
- Файлы: `splice-core/src/{frontend/mod.rs, validate.rs (новый), hir/verify.rs}`, `splice-lancy/src/lower/{mod,seal}.rs`, `splice-llvm/src/lower/direct{,/flow}.rs`, генерация вызовов в макросе (`?`).
- Проверка: testkit — каждое ядро из негативных тестов `verify` даёт одну и ту же `BuildError::Verify` на прямом и переносимом пути всех бэкендов; `rg 'get_or_insert|exhausted register handles' splice-lancy` пусто.

**4.3 `replay`, упрощённый `Codegen`, удаление дублей.** ID: CORE-07, CORE-08, SPL-20. [риск]
- Файлы: `splice-core/src/{replay.rs (новый), codegen.rs, lib.rs}` (два метода, без `impl Default for BuildOptions`), `splice-llvm/src/lower/body.rs` (удалить), `splice-lancy/src/lower/portable.rs` (удалить), `splice-text` (через `replay`), Cranelift (прямой путь бесплатно).
- Проверка: testkit (оба пути совпадают у всех бэкендов); листинги не изменились; удалено ≥1000 строк.

**4.4 Общий путь вызова ядра.** ID: CORE-02, CORE-11, BE-14, BE-21.
- Файлы: `splice-core/src/{call.rs, abi.rs}`, `Executable` → `enter`, три `lib.rs` бэкендов (`Vec` на вызов → стек), `buffer.rs` (проверки по типу `CallArgs` один раз), проверка SVE один раз при сборке.
- Проверка: testkit; бенчмарк «прогретый вызов» до и после (ожидается снижение накладных расходов, результат сверяется).

**4.5 Один механизм Rust-вызова.** ID: CORE-05, MAC-19 (адаптеры), BE-16.
- Файлы: `splice-core/src/hir/mod.rs` (`Instr::RustCall { target: RustTarget }`), `legalize/rust_call.rs`, `runtime_context.rs` (удалить `Dispatcher::Scalars`, `new`, скалярную точку входа), `splice-target/src/abi.rs`, три бэкенда (материализация плана; в LLVM `alloca` в entry вместо `set_operand`), `splice-macros/src/scalar/{abi.rs, lower/calls.rs, methods/declaration.rs}` (один генератор адаптера, packet).
- Проверка: `rust_context.rs`, `rust_import_words.rs`, `static_rust_packet.rs` зелёные на 4 бэкендах; замер быстрого пути для ≤2 слов (решение Р7).

**4.6 Легализация памяти, debug-слоты, публикация.** ID: BE-07, BE-08, BE-10, CORE-16 (сохранения), CORE-24. Зависит от Ф5.4 (`Op::Trap/TrapIf` в Lancy). [риск]
- Файлы: `splice-core/src/legalize/memory.rs`, `debug_slots.rs`; `splice-llvm/src/lower/body/simd.rs:460-711`, `splice-cranelift/src/lower/simd.rs:608-930`, `splice-lancy/src/lower/simd.rs:295-453` (переход на примитивы); три `debug_info.rs` → `splice_jit::Publish`; `splice-jit/src/lib.rs` (`Published<M>`).
- Проверка: дифференциальный тест «полный и частичный выход за границы, u16-индексы, повторы scatter, запись до ошибки» на 4 бэкендах; gdb- или DWARF-тест `info locals` для bool и маски.

**4.7 Таблица `SimdOp::info`.** ID: CORE-10, CORE-19 (окончательно). [∥]
- Файлы: `splice-core/src/hir/simd.rs`, `splice-text/src/lib.rs:120-207`.
- Проверка: тест «для каждого `SimdOp` info согласован с `validate`/`output`/листингом».

**4.8 Парсер splice-text и прочее.** ID: CORE-13, CORE-21, CORE-22. [∥]
- Грамматика с типами регистров, именами буферов и debug-строками; `splice_text::parse`; тест `parse(print(f)) == f` на всех `Function` из тестов; тесты верификатора и бэкендов переводятся на `.splice`-файлы.
- Кэш ключа специализации и диагностика рекурсии.
- Конструкторы `TypedIr`/`TypedKernel` запечатываются через `__private::Token`.
- Проверка: roundtrip-тест; `compile_fail`-тест на подделку `TypedKernel`.

### Ф5. lancy HIR
**5.1 Таблица свойств операций.** ID: HIR-02 (шаг 1), HIR-10, HIR-14, HIR-16, HIR-24. [риск: средний]
- Файлы: `hir/ssa/inst/{mod,simd,info}.rs`, `ssa/function.rs` (`append` через `result_type`), `ssa/function/verify.rs`, `transforms/passes/{sccp,simplify,loops,facts,memory,dead,inline,unswitch}.rs`, `loops/fusion.rs`, `operation.rs`.
- Проверка: все файлтесты без изменений вывода; тест «`append` с неверным операндом даёт ошибку, а не I32»; верификатор отвергает `Literal` с битами выше ширины.

**5.2 Менеджер проходов и анализов.** ID: HIR-03, HIR-06, HIR-12, HIR-17, HIR-20, HIR-25, LNC-18.
- Файлы: `hir/transforms/{pass.rs, limits.rs}` (новые), `passes/simplify.rs` (декларативный список; правила как `fn rule_x`), `loops.rs`, `bounds.rs` (без `cfg!`; `Count::Fixed` включить безусловно с `.lancy`-тестом и контрпримером по AGENTS.md или удалить), `module.rs`, `ssa/function.rs` (один внутренний редактор).
- Проверка: файлтесты; verify-после-прохода (Ф1.3) зелёный; время компиляции на `cargo bench -p splice-kernel --bench backends` (стадия build) не хуже.

**5.3 Хранилище IR.** ID: HIR-04, HIR-05, HIR-11. [∥ с 5.4]
- Файлы: `ssa/function.rs` (`SliceParam`, `set_slice_params -> Result`, `compact()`, `param_index`), `dead.rs`, `loops.rs`, `loops/recurrence.rs`, `sccp.rs`.
- Проверка: тест «после unswitch, fusion и inline размер арены не растёт после `compact`»; тесты с невалидными `set_slice_params` дают `Err`.

**5.4 Канонические операции.** ID: X64-09, BE-08 (`Op::Trap`, `Op::TrapIf(kind)`), MAC-14 (`Binary::SaturatingAdd/Sub/Mul`). [∥]
- Файлы: `hir/ssa/inst/*`, `simplify.rs` (канонизация идиом), `x64/isel/rules/{memory,mask}.rs`, `isel/masks.rs` (сопоставление одной операции), `aarch64/backend/isel/*` (`whilelt`/`cntp`/`compact`), `numeric` (семантика saturating).
- Проверка: `.lancy`-тесты — вариации исходной формы (`Le` вместо `Lt`, `Sub(0,x)` вместо `Neg`, другой порядок `And`) дают ту же каноническую операцию; контрпример, где канонизация неприменима.

**5.5 Общие скалярные и векторные формы `Op`.** ID: HIR-02 (шаг 2). Решение Р8. [риск]
- Файлы: все проходы, isel обоих бэкендов, интерпретатор, листинг, парсер.
- Проверка: полный прогон Ф1 (x86 и qemu); diff листингов только в именах.

**5.6 Текстовый HIR.** ID: LNC-07 (HIR-часть). [∥]
- Файлы: `hir/text.rs` (`Display` + `parse`, имена из таблиц 5.1), `filetests/{listing,input}.rs` (удалить), миграция ≈870 строк фикстур скриптом.
- Проверка: roundtrip на всех фикстурах; принтер публичен (или под feature `text`).

### Ф6. lancy LIR и бэкенды
**6.1 Общая численная семантика и fallback.** ID: X64-04, A64-03 (п. 2, 3), A64-14, HIR-21 (а, б). [риск: временная регрессия скорости на редких операциях]
- Файлы: `support/numeric/` (из `hir/eval.rs`, `hir/integer.rs`), `lir/fallback.rs` (из `aarch64/backend/operations.rs:20-54`), `x64` (вызов fallback вместо `Inst::Lanes`), лимит инлайнера через `TargetLimits`, а не `regalloc2::VReg::MAX`.
- Удаляется `x64/inst/lanes*` (кроме форм, которые Ф8.1 заменит нативными; сначала Ф8.1 для горячих скаляров, затем удаление) и холодные ветки `aarch64/backend/{remainder,special,half}.rs`.
- Проверка: `numeric_coverage` (Ф1.2) на обоих бэкендах; замер на бенчмарках из codex.md: горячие кернелы не медленнее.

**6.2 Общий слой LIR.** ID: X64-10, A64-13 (а), X64-11, A64-18, X64-12, X64-18, X64-14. [∥ с 6.1]
- Файлы: `lir/{frame.rs, interprocedural.rs, args.rs}` (новые), `x64/compile.rs`, `aarch64/backend/compile.rs`, `lir/isel.rs` (`Context::cover` без cfg; правила возвращают `SmallVec` в прямом порядке), `lir/function.rs` (`BlockBuilder`), `lir/schedule.rs` (`Load` отдельно от `Store/Check`), `x64/schedule.rs` и `aarch64/backend/inst.rs` (без `_ =>`), `schedule/liveness.rs` (битовые множества).
- Проверка: тесты планировщика (Ф1.3) с новым классом `Load` (load не переезжает через store или check, две загрузки переставляются); файлтесты `instruction-scheduling`.

**6.3 Декларативное описание инструкций (`define_inst!`).** ID: X64-05, X64-15, X64-19, X64-21, A64-12. [риск] [∥ x64 и aarch64]
- Файлы: `lir/inst_macro.rs`, `x64/inst/{mod,simd}.rs`, `x64/inst/simd/**`, `x64/emit.rs`, `x64/isel/{masks.rs, rules/mask.rs}` (`MaskForm`), `aarch64/backend/{inst,backend}.rs` (temps как `reg_temp`, x10–x15 больше не зарезервированы навсегда).
- Инвариант узких целых записывается явно. `Native` переносится в side-таблицу, `size_of::<Inst>()` ≤ 32.
- Проверка: checker regalloc2 в тестах; `cargo test -p lancy` x86 и qemu; debug-проверка «записанные scratch ∈ temps»; тест `size_of`.

**6.4 Таблица кодировщика aarch64.** ID: A64-11. [∥]
- Файлы: `lir/aarch64/{mod,instruction,operand}.rs`.
- Проверка: golden-таблица (Ф1.3) без изменений; `Assembler::put` проверяет `is_available` в debug.

**6.5 Стадии бэкенда для filetests и принтер LIR.** ID: LNC-06 (а–в), LNC-07 (LIR-часть).
- Файлы: `filetests/mod.rs`, `lir/x64/tests/filetests/*` (удалить `selected()`), `aarch64/backend/tests.rs`, `lir/printer.rs`.
- Проверка: файлтест, который упирается в `FunctionTooLarge`, проходит через fallback `Inlining::Never` так же, как продакшен.

### Ф7. splice-macros
**7.1 Нормализатор Kernel AST.** ID: MAC-06, MAC-09, MAC-11, MAC-12, MAC-18, MAC-20, MAC-21, MAC-23, MAC-24, MAC-25. [риск: самый большой рефакторинг макроса] Можно начинать параллельно с Ф4.
- Файлы: `splice-macros/src/{kast/, resolve.rs, types.rs, stage.rs, emit/{ir,eval,print}.rs}`. Удаляются `scalar/flow.rs`, `scalar/lower/{source,comptime(частично),host(частично)}.rs`, `import/region.rs` (`Collect`).
- Проверка:
  - детерминизм: две развёртки побайтно равны (тест на `-Zunpretty=expanded` или `proc_macro2`-юнит);
  - trybuild по каждому неподдерживаемому узлу;
  - метки, `break 'l`, `break value`, guard в known-`match` и исчерпывающий `bool`-match понижаются в IR (`assert_lowered`);
  - весь testkit.

**7.2 Типы через rustc.** ID: MAC-07, MAC-02 (окончательно), SPL-12. Зависит от 4.1 и 7.1.
- Файлы: `splice-core/src/frontend/` (`Reg<T: KernelScalar>`, `lit::<T>`), `splice-macros/src/emit/ir.rs`, `splice/examples/schema_plan.rs` (скалярные `col[row]`, помощник `dispatch_element!`).
- Проверка: `while i < xs.len()`, `let i: usize = 0`, `5u8`, `1i64`, ветки `0`/`xs.len()` понижаются и совпадают с Rust в debug и release (включая переполнение `usize`); тест-сторож откатов (1.1) не меняется.

**7.3 Ошибки при компиляции вместо autoref.** ID: MAC-16, CORE-17 (сторона макроса). [∥]
- Проверка: trybuild на «field has no Splice layout», «drop glue», «borrowed environment».

**7.4 Реестр интринсиков.** ID: MAC-13, MAC-14 (сторона макроса). [∥]
- `#[intrinsic("name")]` публикует `const`; бэкенд держит реестр; встроенные `saturating_*`/`min`/`max`/SIMD переписываются как интринсики; `saturating_*` понижается в HIR-операцию 5.4.
- Проверка: `saturating_add` не порождает Rust-вызов (IR-проверка); бенчмарк до и после.

**7.5 Объём сгенерированного кода.** ID: MAC-10, MAC-19 (`PublicApi`). [∥]
- Проверка: развёртка `size.rs` уменьшилась (базовая 120 012 байт) и время сборки (0,30 с против 0,12 с) зафиксированы в тесте-бенчмарке; `print` по запросу.

**7.6 Режим переполнения и пути к исходникам.** ID: MAC-15, MAC-26, CORE-25 (DWARF-путь). [∥]
- Файлы: удалить `overflow.rs`; `splice_core::overflow_checks()`; `Span::local_file()`; `splice-jit/src/dwarf.rs` без обхода предков.
- Проверка: тест с `-C overflow-checks=on/off` в release и debug; DWARF-путь абсолютный.

### Ф8. Производительность
Каждый шаг выполняется так:
- до реализации — агент проектирования тестов (пример и контрпример по AGENTS.md);
- `.lancy`-тест преобразования;
- замер «до/после» в release на одной нагрузке со сверкой результата: `cargo bench -p splice-kernel --bench backends`, время сборки и вызова отдельно, `LlvmManualSimd` как эталон.

**8.1 Нативные скалярные правила x64.** ID: X64-01. Зависит от 6.3.
- Тест «ни одна скалярная операция ≤ 8 байт не уходит в fallback» (probe по всем `Action × Type`). Затем удалить остаток `lanes*` (6.1).

**8.2 VEX/EVEX для скаляров, `vzeroupper`, xmm16–31.** ID: X64-02, X64-07. Зависит от 6.3.
- Тест листинга: при AVX нет legacy-SSE мнемоник, `vzeroupper` стоит перед `call` и `ret` корня; регистры 16–31 аллоцируются при EVEX.

**8.3 Спиллы, workspace, пул констант.** ID: X64-08, A64-10, X64-17, LNC-04, X64-16. [∥ с 8.4]
- Спилл и move по ширине типа; workspace выровнен на 64 и переиспользуется через `Kernel::workspace()`/`call_with`; база спиллов кратна ширине; пул констант с дедупликацией (`HashMap`) и выравниванием 64.

**8.4 aarch64.** ID: A64-01, A64-02, A64-03 (п. 1), A64-06, A64-07, A64-08, A64-09, A64-15, A64-17. Зависит от 6.3 и 6.4; Р1.
- Порядок:
  1. min/max с NaN-фиксапом;
  2. f16/bf16 (SVE `.H`, `fcvtl/fcvtn`, `fp16`/`bf16` в `Arm64Features`);
  3. copysign, классификация, signum;
  4. регистры (x13–x15, x21, v8–v15 как non-preferred);
  5. `b.cond`/`cbz` с релаксацией;
  6. `ldp/stp/ldur/regoff/movn`, пул литералов;
  7. кэш PTRUE;
  8. мелочи A64-15;
  9. frame record x29.
- Проверка: тесты `lacks` (нет межбанковых `fmov` и циклов для F32/F64 min/max); число инструкций по objdump на 14 ядрах из отчёта 03; `numeric_coverage` под qemu на 4 профилях.

**8.5 Оптимизации HIR.** ID: HIR-07, HIR-08, HIR-18, HIR-19, BE-09. Зависит от 5.2 и 5.3.
- Все группы за раунд с маской `touched`; повтор дешёвой фазы; локальная VN; ключ `facts.root` для проверок и `CheckOverflow`; удаление неиспользуемых загрузок в доказанных границах; CSE загрузок из неизменяемых срезов (кэш в splice-lancy удаляется).
- Проверка: `.lancy`-тесты с контрпримерами (алиасинг, запись между загрузками, недоказанные границы); время компиляции не растёт (codex.md).

**8.6 JIT-инфраструктура и рантайм.** ID: BE-11, BE-12, LNC-13, LNC-17, CORE-11 (остатки), CORE-21, CORE-27, SPL-18 (анализ сверхлинейности LLVM). Р15.
- Общий LLJIT и `ResourceTracker` на ядро; кэш `TargetMachine` и ISA; один стековый слот на функцию в Cranelift; аллокатор кодовых регионов (только после замера); оценка размера до инлайнинга; `chunks_exact` или `std::simd` в эталоне `Lane`.

**8.7 Переснять таблицы README.** ID: SPL-05.
- Скрипт генерирует таблицы из `target/criterion` с метаданными: коммит, версии rustc/LLVM/Cranelift, фактическая ISA каждого бэкенда. Только после BE-03, BE-04 и BE-06. Исторические столбцы переносятся в `docs/bench-history.md`.

### Ф9. Публичный API, unsafe, документация, crates.io
**9.1 API lancy.** ID: LNC-02, LNC-03, LNC-05 (остаток), HIR-15 (документация).
- `Kernel` (переименование `ScalarKernel`), `call_packed` и `typed::<A, R>()`; `workspace_words` удаляется; энкодер aarch64 `pub(crate)` или feature `aarch64-asm` (Р5); `slot_key!` `pub(crate)`; убрать `pub mod ssa`/`transforms`; `DebugFunction` с геттерами; `#![warn(missing_docs, unreachable_pub)]`.
- Проверка: `cargo rustdoc -p lancy -- -D missing_docs`; интеграционные тесты `lancy/tests/` используют только публичный API (в том числе срезы, `compile_debug`, `Inlining::Never`).

**9.2 Фасад и документация splice.** ID: SPL-06, SPL-09, SPL-10, CORE-20, CORE-22 (если не сделан в 4.8).
- Явные реэкспорты и модули бэкендов, `__private`; примеры и бенчмарки только через фасад; README как доктест (`#![doc = include_str!]` или `cfg(doctest)`-модуль); разделение спецификации на «Реализовано (тест)» и «Планируется», `tests/spec/*.rs` на каждую строку таблиц api-cases; удалить мёртвый API.
- Проверка: `cargo test --doc -p splice-kernel` с примерами README; `cargo doc --workspace -D warnings`.

**9.3 unsafe/FFI и отладка.** ID: LNC-12, LNC-14, BE-19, BE-20, BE-22, A64-17 (документация ABI корня). [∥]
- `// SAFETY:` у каждого `unsafe` (lint `clippy::undocumented_unsafe_blocks` в `[workspace.lints]`); `libc`/`windows-sys`; `debug_assert` результатов `munmap`/`VirtualFree`; unwind-регистрация на Win64 и Linux; `CodeMemory(JITModule)` с `unsafe impl Sync`; инвариант `i32`-смещений GEP в `SAFETY` и тест; BE-22 закрыть документацией (символы по имени — только при появлении кэша, Р15).
- Проверка: clippy-lint зелёный; тест «исключение или раскрутка через JIT-кадр» на Windows-джобе (опционально).

**9.4 Готовность к crates.io и правила агентов.** ID: SPL-13, SPL-16, CLIP-01…03.
- `[workspace.package]`, `[workspace.dependencies]`, `[workspace.lints]`; `readme`, `keywords`, `categories`, docs.rs-метаданные для `splice-llvm`; `exclude` тестов или publish-совместимые dev-deps; `cargo publish --workspace --dry-run` в CI.
- `codex.md` сливается в AGENTS.md (все бэкенды, оба пути, оба режима, профили); `.codex/` удаляется из git (Р16).
- Clippy pedantic: аудит ≈250 кастов (CLIP-01) через `try_from` и `From` или с комментарием; шумовые lint-ы явно разрешаются в `[workspace.lints]` (CLIP-02); функции длиннее 150 строк разбиваются по ходу Ф5–Ф7 (CLIP-03).
- Проверка: `cargo publish --workspace --dry-run`; `cargo clippy --workspace --all-targets -- -D warnings` с новыми lint-ами.

---

## Решения для владельца

| # | Вопрос | Варианты | Рекомендация |
|---|---|---|---|
| Р1 | Ручной IEEE-код aarch64 (`remainder/special/half/integer_float`) | (а) оставить; (б) холодное отдать общему fallback, горячее перевести на нативные формы | **(б).** Дифференциальный тест показал, что семантика верна, но это ≈1000 строк третьей копии и 60–290 инструкций на операцию. Нативные формы короче и быстрее, fmod и редкие f16-операции не горячие. |
| Р2 | x64 `inst/lanes*` (≈1600 строк) | (а) оставить как fallback; (б) удалить после нативных скалярных правил и общего fallback | **(б)**, в порядке Ф8.1 → удаление. |
| Р3 | Версия LLVM | (а) 19 везде; (б) 21 везде, со сборкой LLVM-C для Windows; (в) две версии, обе в CI | **Одна версия.** 21, если удастся стабильно получить dev-сборку LLVM для Windows (официальный архив `clang+llvm-*-windows-msvc`). Иначе 19 на обеих ОС. Вариант (в) удваивает CI и всё равно даёт несопоставимые замеры. |
| Р4 | Откат в Rust при неподдерживаемом ядре | (а) ошибка компиляции + `#[kernel(rust)]`; (б) откат с предупреждением; (в) как сейчас | **(а).** Сейчас откатываются только шесть намеренных ядер. Явность нужна тестам, производительности и стабильности API. |
| Р5 | Позиционирование lancy и объём публичного API | (а) самостоятельный JIT на crates.io с минимальным стабильным API; (б) внутренний бэкенд splice (`#[doc(hidden)]`, без semver-обещаний) | **(а) с минимальной поверхностью:** `Kernel`, `CompileOptions`, `hir`, `debug`, ошибки. Энкодер aarch64 и slotmap — `pub(crate)`. Если цели публиковать lancy отдельно нет, выбрать (б) и не тратить время на документацию 800 элементов. |
| Р6 | Набор скалярных типов и семантика `usize` | (а) `Bool, I32, U32, I64, U64, Index(=usize, 64 бит), F32, F64`; (б) минимум `Bool, I32, Index` | **(а).** `Index` как настоящий `usize` с беззнаковыми checked-операциями, иначе MAC-02 вернётся. Добавление типов после смены ABI дешёвое, сама смена ABI дорогая. |
| Р7 | ABI Rust-вызова | (а) только packet; (б) packet + быстрый путь ≤2 слов, выбор в общем плане | **(а)**, быстрый путь добавить по замеру и только внутри `rust_call_plan`. |
| Р8 | Слияние скалярных и векторных `Op` в Lancy HIR (HIR-02, шаг 2) | (а) делать; (б) ограничиться таблицей `OpInfo` | **(а), но после 5.1–5.4.** Таблица снимает основную боль. Слияние убирает оставшиеся `X \| Simd(X)`, но трогает всё. |
| Р9 | Feature `experimental-simd` | (а) оставить и гонять в CI; (б) влить доказанное (`Count::Fixed` в bounds) в stable и сузить feature до модуля `experimental` | **(б)** + CI-прогон оставшейся feature. |
| Р10 | Семантика `CallScratch` (HIR-01) | (а) адресация по смещению со сдвигом при инлайнинге; (б) контракт «scratch не переживает `Call`» + верификатор | **(а):** не сужает контракт и чинит также `DebugStore` (HIR-13). |
| Р11 | Символы GDB JIT при наличии libLLVM (BE-05) | (а) при LLVM использовать его определения (extern) под своим замком и запретить GDB-listener ORC; (б) регистрировать через API LLVM; (в) переименовать символы (GDB их не увидит) | **(а)** сейчас, **(б)**, если понадобится debug-support ORC. |
| Р12 | Windows в CI | (а) полноценная джоба с LLVM; (б) только lancy + Lancy/Cranelift без LLVM | **(а)** при Р3 = одна версия; минимум **(б)** сразу: Win64-ABI сейчас не исполняется нигде. |
| Р13 | `LlvmManualSimd` | (а) явный текст конвейера без LV и SLP; (б) оставить `noimplicitfloat` и задокументировать отличие от codex.md | **(а):** codex.md требует именно «O3 без LV и SLP»; проверять тестом. |
| Р14 | `#[intrinsic]` и спецификация | (а) реализовать реестр (MAC-13); (б) убрать из спецификации | **(а):** на нём же переписываются встроенные `min`/`max`/`saturating`/SIMD, и пропадают захардкоженные имена. |
| Р15 | Пул кода, общий LLJIT, символы по имени (LNC-13, BE-11, BE-22) | делать сразу или по замеру | **По замеру:** сценарий «тысячи маленьких ядер» нужно подтвердить нагрузкой. |
| Р16 | `codex.md` и `.codex/config.toml` | слить в AGENTS.md и убрать из git или оставить | **Слить и убрать** (персональный конфиг); команды — через `xtask`. |
| Р17 | Режим переполнения в IR (MAC-15) | (а) runtime-проба в крейте пользователя; (б) явный атрибут или опция `BuildOptions` | **(а)** по умолчанию + переопределение через `BuildOptions`. |
| Р18 | Имя библиотеки `splice-core` (SPL-07) | (а) `splice_core` (ломает импорты бэкендов); (б) оставить `splice` | **(а):** иначе не работают `cargo doc --workspace` и docs.rs. |

---

## Индекс находок

Колонка «Фаза» ссылается на шаг плана. «см. X» означает дубликат или следствие: исправление делается в рамках X.

### HIR (часть 01)
| ID | Серьёзн. | Категория | Суть | Фаза |
|---|---|---|---|---|
| HIR-01 | high | bug | Инлайнер копирует `CallScratch`: scratch вызывающей и вызываемой функции совпадает (1919 против 719) | 0.4 |
| HIR-02 | high | arch | Скаляр и вектор — разные `Op`, свойства операций в 10–15 `match` | 5.1, 5.5 |
| HIR-03 | high | arch | Монолитный конвейер проходов без менеджера анализов и явной инвалидации | 5.2 |
| HIR-04 | med | hack | Метаданные срезов через арифметику индексов `Value`; `set_slice_params` паникует | 5.3 |
| HIR-05 | med | hack/perf | Арена рёбер и операндов только растёт; эвристика `span.len == params.len()` | 5.3 |
| HIR-06 | med | hack | `cfg!(experimental-simd)` меняет поведение стабильного `bounds` | 5.2 (Р9) |
| HIR-07 | med | perf | `loops::run`: одна группа за раунд, полный пересчёт анализов | 8.5 |
| HIR-08 | med | perf/arch | Нет повторного SCCP и copy-prop после циклов; `Op::Copy` доходит до isel; нет CSE/VN | 8.5 |
| HIR-09 | med | test | Проходы не проверяются верификатором; пробелы в юнит-тестах | 1.3 |
| HIR-10 | med | bug | `Literal` с битами выше ширины; два представления i32-константы | 5.1 |
| HIR-11 | med | hack | Позиция параметра через `binary_search` по ID | 5.3 |
| HIR-12 | med | arch | Два API редактирования; публичный `replace_op` — O(n) верификация на вызов | 5.2 |
| HIR-13 | low | bug | То же для `DebugStore` при инлайнинге (см. HIR-01) | 0.4 |
| HIR-14 | low | style | `new_instruction` молча подставляет `I32` | 5.1 |
| HIR-15 | low | style | Публичный API паникует на ошибках пользователя | 2.2, 9.1 |
| HIR-16 | low | style | `pure(Option<Type>)`, классификация ловушек разделена с `is_total` | 5.1 |
| HIR-17 | low | hack | Магические лимиты по файлам; `known.remove(0)` | 5.2 |
| HIR-18 | low | perf | `remove_redundant_checks` по сырым `Value`, `CheckOverflow` не дедуплицируется | 8.5 |
| HIR-19 | low | perf | Неиспользуемые загрузки в доказанных границах не удаляются | 8.5 |
| HIR-20 | low | style | Ранний `return` в правиле `Select` пропускает упрощение; правила в одном замыкании | 0.10, 5.2 |
| HIR-21 | low | arch | `eval` как рантайм aarch64; лимит инлайнера из regalloc2; дубли `MIN/MAX_VECTOR_BYTES` | 6.1, 3.1 |
| HIR-22 | low | style | `InvalidNumeric` для несвязанных ошибок; `Bitcast` только `Ptr→U64` | 2.2 |
| HIR-23 | low | style | Ширина `Ptr`/`Usize` зашита как 8 | 3.2 |
| HIR-24 | low | style | Неявная семантика `Fact::Constant` для векторов | 5.1 |
| HIR-25 | low | style | `simplify_bodies` зависит от порядка `module.functions` | 5.2 |

### X64 (часть 02)
| ID | Серьёзн. | Категория | Суть | Фаза |
|---|---|---|---|---|
| X64-01 | high | perf | Скаляры вне i32 идут через стековый `Inst::Lanes` (подтверждено) | 8.1 |
| X64-02 | high | perf | Смесь SSE и AVX без `vzeroupper` | 8.2 |
| X64-03 | high | test | 12 веток `cfg!(windows)` в ABI не исполняются нигде | 3.2, 1.6 |
| X64-04 | high | arch | ≈1600 строк поэлементного ассемблера — третья копия семантики | 6.1 (Р2) |
| X64-05 | high | arch | Знание об инструкции в 4–6 местах; позиционная связь аллокаций | 6.3 |
| X64-06 | med | arch/hack | `Features::host()` ≈40 раз, `is_x86_feature_detected!` в обход профиля | 3.2 |
| X64-07 | med | perf | xmm16–31 не используются | 8.2 |
| X64-08 | med | perf | Скалярные спиллы и move на полную ширину zmm | 8.3 |
| X64-09 | med | hack | isel распознаёт форму целого кернела (`compact_store`, `prefix_mask`) | 5.4 |
| X64-10 | med | arch | Расчёт кадра, `written`, раскладка аргументов дублируются с aarch64 | 6.2 |
| X64-11 | med | perf | Все загрузки `Ordered`; `_ => Barrier` | 6.2 |
| X64-12 | med | style/arch | `cfg(target_arch)` в общем `Context`, двойной `reverse` | 6.2 |
| X64-13 | med | test | Нет тестов планировщика, liveness и layout | 1.3 |
| X64-14 | low | perf | `Liveness::solve` O(V·B) | 6.2 |
| X64-15 | low | perf | `size_of::<Inst>() == 128` | 6.3 |
| X64-16 | low | perf | Пул констант без дедупликации и выравнивания | 8.3 |
| X64-17 | low | perf | Workspace выровнен на 8, спиллы пересекают кэш-линии | 8.3 |
| X64-18 | low | style | Неявный протокол builder-а `Function` | 6.2 |
| X64-19 | low | hack/test | Тестовый `emit` без пула — другой путь, чем в продакшене | 1.3 |
| X64-20 | low | style | `nop` на `DebugPoint` в release | 0.10 |
| X64-21 | low | style | Не описан инвариант верхних бит узких целых, два соглашения | 6.3 |

### A64 (часть 03)
| ID | Серьёзн. | Категория | Суть | Фаза |
|---|---|---|---|---|
| A64-01 | high | perf | Float min/max по полосам через GPR (≈60 инструкций на операцию) | 8.4 |
| A64-02 | high | perf | f16/bf16 программно по полосам (251 инструкция на `add`) | 8.4 |
| A64-03 | high | arch/perf | ≈1000 строк ручного IEEE; см. X64-04 | 6.1, 8.4 (Р1) |
| A64-04 | med | test | Нет дифференциального теста операций для aarch64; см. LNC-01 | 1.2 |
| A64-05 | med | arch | Бэкенд aarch64 компилируется только на aarch64 | 3.2 |
| A64-06 | med | perf | Не используются x13–x15, x21, v8–v15; вызов затирает все векторы | 8.4 |
| A64-07 | med | perf | `b.inv +8; b L` вместо `b.cond`/`cbz`; лишний `b epilogue` | 8.4 |
| A64-08 | med | perf | Нет `ldp/stp/ldur/regoff/movn`, пула литералов | 8.4 |
| A64-09 | low | perf | SVE: PTRUE на каждую операцию, маски через Z, `ld1b` с `add` | 8.4 |
| A64-10 | low | perf | Спиллы на полную ширину, невыровненная область спиллов; см. X64-08, X64-17 | 8.3 |
| A64-11 | med | arch/style | Три параллельные таблицы кодировщика, псевдоформы, `Ldr_64` через базу LDRH | 6.4 |
| A64-12 | med | arch/hack | Неявный контракт scratch-регистров v16–v19, x9–x17; см. X64-05 | 6.3 |
| A64-13 | low | arch | Дубли с x64 (см. X64-10); ширина спилла из глобала | 6.2, 3.2 |
| A64-14 | low | hack | Паника в `primitive` через `extern "C"` даёт abort; дубль пути ловушки | 6.1, 2.2 |
| A64-15 | low | perf | Мелочи: select I128 ветвлениями, деление NEON, `vmulu64` через GPR | 8.4 |
| A64-16 | low | test | С эталоном сверено ≈10 из 347 форм; нет теста `logical_immediate` | 1.3 |
| A64-17 | low | arch | Нет frame record x29; ABI корня не задокументирован | 8.4, 9.3 |
| A64-18 | low | perf | `_ => Barrier` на aarch64; см. X64-11 | 6.2 |

### LNC (часть 04)
| ID | Серьёзн. | Категория | Суть | Фаза |
|---|---|---|---|---|
| LNC-01 | high | test | Тесты, не зависящие от архитектуры, (≈1050 строк) лежат в x64 и не идут на ARM; нет ABI-пробы aarch64 | 1.2 |
| LNC-02 | high | arch | Случайная публичная поверхность: энкодер aarch64, slotmap, CFG/Dom, дубли путей, 802 элемента без документации | 9.1 (Р5) |
| LNC-03 | med | arch | Четыре метода вызова с ручной упаковкой `u64`; имя `ScalarKernel`; `workspace_words` | 9.1 |
| LNC-04 | med | perf | `Vec` под workspace и аргументы на каждый вызов; см. CORE-11 | 8.3 |
| LNC-05 | med | arch | `CompileError`: нет `non_exhaustive`, cfg-вариант, внешние типы, двойной Display, fallback по белому списку | 2.2, 0.10 (ж) |
| LNC-06 | med | test/arch | Харнесс filetests исполняет не продакшен-конвейер; `selected()` дублирует `compile`; неиспользованные проверки | 1.4 (г), 6.5 |
| LNC-07 | med | arch/test | Вход и листинг HIR несовместимы; фикстуры завязаны на `Debug` | 5.6, 6.5 |
| LNC-08 | med | test | `experimental-simd` не проверяется в CI | 1.6 |
| LNC-09 | med | test | `@run` только i32 и один буфер; README неполон | 1.4 |
| LNC-10 | med | hack | `build.rs` исполняется у потребителей и паникует без `filetests/` | 1.4 |
| LNC-11 | low | test/style | Проверки кодирования магическими масками по всему `code.bytes` | 1.4 |
| LNC-12 | low | style | `unsafe` FFI без `SAFETY:`, ручные константы ОС | 9.3 |
| LNC-13 | low | perf | Отдельный mmap на каждое ядро | 8.6 (Р15) |
| LNC-14 | low | arch | Нет unwind-информации у JIT-кода | 9.3 |
| LNC-15 | low | style | Интерпретатор переводит ошибки в строки и обратно | 1.4 |
| LNC-16 | low | arch | lancy захардкожен под splice (tracing target, глобальный профиль, паника `vector_bytes`) | 3.2 |
| LNC-17 | low | perf | Fallback инлайнинга повторяет всю компиляцию | 8.6 |
| LNC-18 | low | style | Идентичность функций по адресу `Arc`, `FunctionIds` во всех стадиях | 5.2 |

### MAC (часть 05)
| ID | Серьёзн. | Категория | Суть | Фаза |
|---|---|---|---|---|
| MAC-01 | high | arch/bug | Молчаливый откат ядра в Rust-импорт (подтверждено) | 1.1 (Р4) |
| MAC-02 | high | bug | `usize` в i32: JIT `Ok(1)` вместо паники (подтверждено) | 0.5, 7.2 |
| MAC-03 | high | bug | Host-арифметика не-i32 паникует в `build`, в том числе в неисполняемой ветке (подтверждено) | 0.6 |
| MAC-04 | high | bug | `comptime!(self.x)` в `#[kernel] impl` не компилируется (подтверждено) | 0.7 |
| MAC-05 | high | bug/hack | Гигиена: коллизии `Signature`, `use Status::*`, `mod foo` (подтверждено) | 0.7 |
| MAC-06 | high | arch | Подмножество Rust разбирается 6–9 обходами | 7.1 |
| MAC-07 | high | bug/arch | Нет вывода типов, литерал всегда i32 | 7.2 |
| MAC-08 | high | hack | Публикация методов `impl` молча глотает ошибки | 1.1 |
| MAC-09 | med | hack | Разрешение имён строками и последним сегментом; шесть списков примитивов | 7.1 |
| MAC-10 | med | perf | Развёртка ×365, тело понижается трижды (подтверждено) | 7.5 |
| MAC-11 | med | hack/perf | Недетерминированный вывод, узлы по адресу памяти (подтверждено) | 7.1 |
| MAC-12 | med | bug | Guard в known-`match`, метки, `break value` не понижаются | 7.1 |
| MAC-13 | med | arch | `#[intrinsic]` ничего не делает | 7.4 (Р14) |
| MAC-14 | med | perf | `saturating_*` — непрозрачный Rust-вызов на каждую операцию | 5.4, 7.4 |
| MAC-15 | med | hack | Режим переполнения по argv rustc внутри proc-macro | 7.6 (Р17) |
| MAC-16 | med | arch | Ошибки, известные при компиляции, откладываются autoref-ом до `build` | 7.3 |
| MAC-17 | med | test | Нет trybuild, тестов отката, гигиены, `usize` | 1.5 |
| MAC-18 | med | bug/hack | Области видимости `HashMap<String,_>`, сравнение спанов | 7.1 |
| MAC-19 | med | arch/dup | Два генератора API и два адаптера с разной обработкой паник | 4.5, 7.5 |
| MAC-20 | low | perf/style | Спекулятивное понижение и перепарс токенов | 7.1 |
| MAC-21 | low | perf | Неподвижная точка `Flow` с клонированием, `return` всегда runtime | 7.1 |
| MAC-22 | low | bug | `RebasePaths` не переписывает `UseTree` | 0.7 |
| MAC-23 | low | style | Посторонние атрибуты параметров отвергаются; непоследовательная детекция `comptime` | 0.7 |
| MAC-24 | low | style | Вводящие в заблуждение сообщения об ошибках | 7.1 |
| MAC-25 | low | hack | Печать решает про метки строковым поиском | 7.1 |
| MAC-26 | low | bug | Неверный DWARF-путь (`CARGO_MANIFEST_DIR` + относительный `Span::file`) | 7.6 |

### CORE (часть 06)
| ID | Серьёзн. | Категория | Суть | Фаза |
|---|---|---|---|---|
| CORE-01 | high | arch | Скалярная система только на `i32` на всех уровнях | 4.1 (Р6) |
| CORE-02 | high | arch/dup | Вызов ядра и статусы ловушек ×3 с разными кодировками | 2.1, 4.4 |
| CORE-03 | high | arch/bug | Строковые ошибки; сообщение паники теряется; не-паника выдаётся за панику | 2.3 |
| CORE-04 | high | hack | `immediate_condition`: отсечение ветки по пипхол-состоянию бэкенда | 0.8 |
| CORE-05 | high | arch/dup | Четыре механизма Rust-вызова, план ×3, `Dispatcher::Scalars` только для тестов | 4.5 (Р7) |
| CORE-06 | high | arch/dup | Верификатор в трёх копиях, прямой путь не проверяется общим кодом | 4.2 |
| CORE-07 | med | arch/dup | Нет общего `replay`, у LLVM две реализации понижения | 4.3 |
| CORE-08 | med | arch | Шесть точек входа `Codegen`, дефолты теряют `BuildOptions` | 4.3 |
| CORE-09 | med | arch | `BufferArg` — enum на 13 вариантов с четырьмя `match` | 4.1 |
| CORE-10 | med | arch | Семантика `SimdOp` разбросана примерно по 7 местам | 4.7 |
| CORE-11 | med | perf | Проверка буферов, `Vec` и `shape()` на каждый вызов и колбэк | 4.4, 8.6 |
| CORE-12 | med | arch | Цель — глобальный `SPLICE_SIMD` | 3.1 |
| CORE-13 | med | arch/test | В splice-text нет парсера, формат не круговой | 4.8 |
| CORE-14 | med | test | splice-jit без тестов; верификатор без юнит-тестов | 1.5 |
| CORE-15 | med | bug | Конфликт символов GDB JIT с libLLVM; подтверждён в BE-05 | 0.9 (см. BE-05, Р11) |
| CORE-16 | med | bug/dup | `bool` = 255 в GDB; сохранение debug-слотов ×3 | 0.10, 4.6 |
| CORE-17 | low | hack | `ContextSignature` молча даёт `size: None` | 2.3, 7.3 |
| CORE-18 | low | bug | Debug-чтения ломают проверку определённой инициализации | 0.10 |
| CORE-19 | low | bug | `may_trap` не учитывает SIMD-операции с памятью | 0.10, 4.7 |
| CORE-20 | low | style | Мёртвый и несогласованный API, сентинел `Reg(u32::MAX)` | 2.3, 9.2 |
| CORE-21 | low | perf | Ключ специализации сериализуется на каждый вызов; нет диагностики рекурсии | 8.6, 4.8 |
| CORE-22 | low | arch | `TypedIr::new`/`TypedKernel::new` позволяют подделать сигнатуру | 4.8 |
| CORE-23 | low | bug | perf-map только дописывается, `create_new` отключает карту | 0.10 |
| CORE-24 | low | arch | Порядок `Drop` регистрации держится на порядке полей; публикация ×3 | 4.6 |
| CORE-25 | low | bug | `e_machine` для чужих архитектур; поиск файла по предкам; см. MAC-26 | 0.10, 7.6 |
| CORE-26 | low | perf | `Auto` выбирает 32 байта без AVX2; нет `Neon` вне Linux | 0.10 |
| CORE-27 | low | perf | Эталонный `Lane` — скалярные циклы; константа 64 зашита | 3.1, 8.6 |

### BE (часть 07)
| ID | Серьёзн. | Категория | Суть | Фаза |
|---|---|---|---|---|
| BE-01 | **crit** | bug | Cranelift SIGSEGV при `sse` на f32-сравнении с невыровненной загрузкой | 0.2 |
| BE-02 | high | bug | Cranelift не компилирует `Cast f64→i64` | 0.3 |
| BE-03 | high | arch | LLVM 19 на Windows и 21 на Linux | 3.4 (Р3) |
| BE-04 | high | arch/bug | Профиль переводится в ISA трижды, переводы расходятся | 3.3 |
| BE-05 | high | bug | Дубль `__jit_debug_*` со статическим LLVM (подтверждает CORE-15) | 0.9 (Р11) |
| BE-06 | med | hack | `LlvmManualSimd` через `noimplicitfloat` — слабее, чем «O3 без LV и SLP» | 3.4 (Р13) |
| BE-07 | high | arch/dup | Семантика памяти и пакет Rust-вызова ×3; см. CORE-05, CORE-07 | 4.6 |
| BE-08 | med | hack | Ловушки Lancy через `BoundsCheck(0,0)` | 5.4, 4.6 |
| BE-09 | med | hack/perf | CSE загрузок в адаптере, а не в Lancy | 8.5 |
| BE-10 | med | arch | Короткая запись создаёт ветвление пользовательскими метками | 4.6 |
| BE-11 | med | perf | Новая JIT-инфраструктура на каждое ядро | 8.6 (Р15) |
| BE-12 | med | perf | Cranelift: новый стековый слот на каждую инструкцию | 8.6 |
| BE-13 | med | perf | Cranelift без fma, popcnt, bmi вне auto; см. BE-04 | 3.3 |
| BE-14 | med | arch | Три ABI входа и кодировки статуса; см. CORE-02 | 2.1, 4.4 |
| BE-15 | med | test | Нет дифференциального теста бэкендов и матрицы профилей; см. SPL-02 | 1.5 |
| BE-16 | low | hack | LLVM: размер `alloca` пакета дописывается задним числом | 4.5 |
| BE-17 | low | style | Корень определяется по имени `kernel0` | 0.10 |
| BE-18 | low | bug | `Intrinsic::find(..).unwrap()` паникует на другой версии LLVM | 0.10, 3.4 |
| BE-19 | low | safety-doc | Инвариант `i32`-смещений GEP не записан в `SAFETY` | 9.3 |
| BE-20 | low | style | `Mutex<CodeMemory>` только ради `Sync` | 9.3 |
| BE-21 | low | arch | Проверка SVE на каждом вызове только в LLVM; см. CORE-02 | 4.4 |
| BE-22 | low | perf | Адреса вшиты в код, кэш на диске невозможен | 9.3 (Р15) |

### SPL (часть 08)
| ID | Серьёзн. | Категория | Суть | Фаза |
|---|---|---|---|---|
| SPL-01 | high | bug | CI и AGENTS.md ссылаются на несуществующий пакет `splice` (подтверждено) | 0.1 |
| SPL-02 | high | test/arch | ≈60 локальных `check::<B>`, перекос покрытия бэкендов | 1.5 |
| SPL-03 | high | test | Режим `BuildOptions::Debug` фактически не тестируется | 1.5 |
| SPL-04 | high | test | Тесты не отличают JIT от отката в Rust; см. MAC-01 | 1.1, 1.5 |
| SPL-05 | med | test/doc | Таблицы README несопоставимы и устарели | 8.7 |
| SPL-06 | med | arch | Фасад неполон (нет `*Kernel`, `LlvmManualSimd`) и слишком широк (`pub use splice_core::*`) | 9.2 |
| SPL-07 | med | arch/bug | Две библиотеки `splice`: `cargo doc` падает, жёсткий `::splice` | 1.6 (Р18) |
| SPL-08 | med | test/arch | Матрица фич не тестируется, LLVM нужен для любого теста | 1.6 |
| SPL-09 | med | doc | Примеры README и api-cases не компилируются | 9.2 |
| SPL-10 | med | doc | Спецификация не различает реализованное и планы | 9.2 |
| SPL-11 | med | test | Нет Windows, debug, всех бэкендов в профилях, doc, package | 1.6 |
| SPL-12 | med | hack | `schema_plan` обходит i32-ограничение; см. CORE-01 | 7.2 |
| SPL-13 | med | arch | Не готово к crates.io: readme, dev-deps, docs.rs, workspace-метаданные | 9.4 |
| SPL-14 | med | arch/style | Харнесс бенчмарков зашит на три бэкенда, `xorshift` ×3 | 1.5 |
| SPL-15 | low | test | Тест трассировки сверяет формат вывода `fmt` | 1.5 |
| SPL-16 | low | style/arch | Правила агентов противоречивы; `codex.md` и `.codex/` | 0.1, 9.4 (Р16) |
| SPL-17 | low | test | `compile_fail`-доктесты без кода ошибки; см. MAC-17 | 0.10, 1.5 |
| SPL-18 | low | perf/test | Тесты `sql_plan` 6,4 с, сверхлинейный рост LLVM | 1.5, 8.6 |
| SPL-19 | low | test | Сравнение текста ошибок; см. CORE-03 | 2.3 |
| SPL-20 | low | bug | `BuildOptions::default()` по профилю splice-core; см. CORE-08 | 4.3 |
| SPL-21 | low | style | `.gitignore` `/target/` только в корне, Cargo.lock фикстуры | 0.1 |

### CLIP (часть 00, синтетические ID)
| ID | Серьёзн. | Категория | Суть | Фаза |
|---|---|---|---|---|
| CLIP-01 | – | style/bug-risk | ≈250 подозрительных кастов (truncation, sign, wrap) по 30+ файлам, особенно `x64/emit.rs`, `inst/lanes*`, `aarch64/instruction.rs`, `splice-lancy/lower/mod.rs`, `runtime_context.rs` | 9.4 (часть исчезает с удалением `lanes*` в 6.1) |
| CLIP-02 | – | style | Pedantic-шум: separators (348), `must_use` (159+26), `# Errors/# Panics` docs (82), wildcard imports (30) | 9.4 (`[workspace.lints]`) |
| CLIP-03 | – | style | Функции по 118–199 строк, однобуквенные биндинги | по ходу Ф5–Ф7 |
