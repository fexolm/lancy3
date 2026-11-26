# Splice

## Установка и публикация

Пакет `splice-kernel` по умолчанию включает макросы и бэкенд Lancy:

```toml
[dependencies]
splice-kernel = "0.1"
```

В коде используйте `splice::{kernel, Lancy}`. Для других бэкендов включите features `cranelift` или `llvm`, например `splice-kernel = { version = "0.1", features = ["cranelift"] }`. Чтобы отключить Lancy, укажите `default-features = false`. LLVM требует установленную версию, указанную ниже. Для самостоятельного использования Lancy добавьте `lancy = "0.1"`.

Для первого выпуска публикуйте пакеты по порядку, дожидаясь появления каждого в индексе crates.io: `splice-target`, `splice-macros`, `lancy`, `splice-core`, `splice-jit`, `splice-lancy`, `splice-cranelift`, `splice-llvm`, затем `splice-kernel`. Перед публикацией каждого пакета выполните `cargo publish -p ИМЯ --dry-run`, затем `cargo publish -p ИМЯ`.

Проект распространяется по выбору пользователя на условиях MIT или Apache-2.0.

Splice — язык для CPU-ядер, которые специализируются по данным приложения после его запуска. Функция с атрибутом `#[kernel]` сохраняет обычный вызов Rust. Макрос также создаёт фабрику JIT-кода. Параметр типа `B` у фабрики выбирает компилятор ядра.

```rust
use splice::{kernel, Cranelift, Lancy, Llvm};

enum Price { Retail, Wholesale(i32) }

#[kernel]
fn total(#[comptime] plan: &Price, price: i32, qty: i32) -> i32 {
    match plan {
        Price::Retail => price.wrapping_mul(qty),
        Price::Wholesale(discount) => price.wrapping_sub(*discount).wrapping_mul(qty),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = Price::Wholesale(3);
    let fast_build = total::build::<Lancy>(total::BuildArgs { plan: &plan })?;
    let cranelift = total::build::<Cranelift>(total::BuildArgs { plan: &plan })?;
    let llvm = total::build::<Llvm>(total::BuildArgs { plan: &plan })?;
    print!("{}", total::print(&plan)?);
    assert_eq!(fast_build.call(total::CallArgs { price: 10, qty: 4 })?, 28);
    assert_eq!(cranelift.call(total::CallArgs { price: 10, qty: 4 })?, 28);
    assert_eq!(llvm.call(total::CallArgs { price: 10, qty: 4 })?, 28);
    Ok(())
}
```

Вызов `build::<Lancy>` строит SSA HIR Lancy прямо при обходе кернела. Вызов `build::<Llvm>` таким же образом строит LLVM IR. Вызов `build::<Cranelift>` строит и проверяет переносимый IR, затем компилирует его в машинный код через Cranelift. Хостовый Rust выбирает вариант `match` по плану, вычисляет известные выражения и разворачивает `for` по `#[comptime]`-срезу. Параметры специализации могут содержать заимствования и пользовательские enum. Коллекцию специализации можно подготовить в приложении и передать в `BuildArgs` по ссылке. Вызов `comptime!(e)` требует вычислить выражение при сборке. Для изменяемой памяти конкретного запуска передайте `&mut [T]` в `CallArgs`.

В пределах одной сборки вложенные вызовы делят тело по объявлению функции, её параметрам типа и значениям `const`- и `#[comptime]`-аргументов. Обычная функция без параметров специализации строится один раз. Для чисел, строк и срезов ключ учитывает содержимое. Для своих структур и перечислений можно добавить `#[derive(splice::ComptimeKey)]`, а для параметра типа — ограничение `T: splice::ComptimeKey`. Аргументы с состоянием, для которых ключ не задан, сохраняют отдельный обход тела при каждом вызове.

`total::ir(&plan)?` возвращает переносимый IR после специализации: `splice::Function` с инструкциями `splice::hir::Instr`. В примере сигнатура содержит только `price` и `qty`, а скидка 3 появляется в теле константой перед умножением. Повторные вызовы одной специализации ссылаются на одно тело (`Instr::Call` с общим `Arc<Function>`). IR отражает семантику Splice до оптимизаций машинного бэкенда.

`kernel_name::print(...)` выполняет специализацию по тем же аргументам и возвращает Rust-код до построения IR. Входы `#[comptime]` исчезают из сигнатуры, а runtime-условия и циклы сохраняют форму Rust. Напечатанный код использует обычные операции Rust, включая `Lane` для SIMD. Побочные эффекты обычной Rust-функции происходят в `call`, когда выполнение доходит до вызова. Для вывода во время подготовки плана приложение может вызвать `println!` перед передачей `BuildArgs`.

Свободная функция `total` объявляет структуры `total::BuildArgs` и `total::CallArgs`. Фабрика принимает аргументы специализации по именам. У исполняемого ядра метод `call` принимает именованные аргументы запуска. Ошибки сборки и вызова имеют типы `splice::BuildError` и `splice::CallError`. Вариант `let ir = total::ir(&plan)?;` принимает аргументы специализации позиционно. Вызов `Lancy::compile(&ir)?`, `Cranelift::compile(&ir)?` или `Llvm::compile(&ir)?` сохраняет тип аргументов запуска. [Спецификация языка](docs/splice-language.md)

При вложенном вызове Rust получает срезы в порядке аргументов. Повторная передача общего среза и повторное заимствование `&mut [T]` сохраняют обычные правила Rust. Если вызов завершился паникой, выполненные перед ней записи видны через исходный буфер.

`#[kernel]` ставится на свободную функцию или целиком на `impl` или `trait`. Тело, которое Splice не может понизить в IR, — ошибка компиляции: Rust-ядер нет, а обычные Rust-функции и методы ядро вызывает как Rust-вызовы. Rust-методы типа пишут в обычном `impl` рядом с `#[kernel] impl`. Для специализации вызова через трейт размечают оба блока, `trait` и `impl Trait for T`. Вызов `<T as Trait>::f(&plan)` может специализироваться, пока живы заимствования внутри `T`. Обычный вызов `plan.f()` выполняет Rust-метод при запуске ядра.

| Крейт | Ответственность |
| --- | --- |
| [splice-core](splice-core/src/lib.rs) | Общий фронтенд, построение и проверка переносимого IR, трейты `Codegen` и `Executable`. |
| [splice-target](splice-target/src/lib.rs) | Определяет доступные на хосте расширения x64 и общую для Splice и Lancy ширину вектора. |
| [splice-macros](splice-macros/src/lib.rs) | Макрос `#[kernel]` разбирает Rust-синтаксис и генерирует обращения к фронтенду. |
| [splice-lancy](splice-lancy/src/lib.rs) | Переводит поток операций Splice или готовую переносимую функцию в SSA HIR и вызывает JIT Lancy. |
| [splice-cranelift](splice-cranelift/src/lib.rs) | Собирает CLIF всех функций кернела в одном модуле. Инлайнер Cranelift может встроить каждое тело один раз и расходует на дополнительные копии бюджет, равный исходному числу инструкций модуля. JIT компилирует точку входа и оставшиеся вызываемые функции с `opt_level=speed`. |
| [splice-llvm](splice-llvm/src/lib.rs) | Помещает точку входа и все внутренние функции кернела в один LLVM-модуль через Inkwell. Запускает `default<O3>` для всего модуля и устанавливает машинный код через ORC LLJIT. `to_llvm_ir(&ir)` печатает модуль до оптимизации. |
| [lancy](lancy/README.md) | Самостоятельный x64-бэкенд. Проверяет SSA HIR, строит LIR, распределяет регистры и устанавливает машинный код. |

Скалярные аргументы `i32` и типизированные срезы передаются двумя списками с сохранением порядка внутри каждого списка. Все три компилятора поддерживают вызовы других ядер, циклы, ранний возврат, проверяемые обращения к срезам и целочисленное деление. Обработка колонок выражается обычным циклом над срезами.

## Выбор колонки по схеме при сборке

Вход `#[columns(layout)] columns: &[BufferArg<'_>]` описывает набор колонок, который меняется между вызовами. Выражение `layout` вычисляется при `build` или `ir` и задаёт порядок и типы колонок через `BufferKind`. `columns[slot].as_slice::<T>()` получает `&[T]` для известного при сборке номера колонки. Поддерживаются `u16`, `u32`, `i32`, `i64`, `f32` и `f64`. Splice проверяет номер и тип в раскладке при сборке, а `Executable::call` проверяет переданные буферы перед входом в JIT.

```rust
#[kernel]
fn query(
    #[comptime] schema: &Schema,
    #[comptime] plan: &Plan,
    out: &mut [i32],
    #[columns(schema.buffer_kinds())] columns: &[BufferArg<'_>],
) -> i32 {
    match plan {
        Plan::Filter { expr: Expr::Ge { col, val } } => match schema.column(*col) {
            Some(Element::I32) => filter_ge(columns[*col].as_slice::<i32>(), *val, out),
            _ => -1,
        },
        Plan::Empty => 0,
    }
}
```

Здесь `filter_ge` — обычное ядро с `&[i32]` и порогом `#[comptime] val: i32`. При вызове `call` поля структуры обозначают выходной буфер и набор колонок в порядке раскладки. Одно ядро принимает разные батчи с этой раскладкой. Для другой раскладки нужно собрать специализацию по новой схеме. Ядро принимает одну таблицу после обычных параметров-срезов. Таблица доступна в точке входа, а выбранный из неё срез можно передавать во вложенные ядра. [Исполняемый пример и тесты](splice/tests/column_table.rs)

В `schema_plan` схема выбирает конкретные специализации `filter<T>`, `join<T>` и `project<T>`. Фильтр читает `i32`, `f32` и `i64`, соединение сравнивает ключи `i64`, а проекция вычисляет два булевых выражения над `u16` и `f64`. Ядра используют `&[T]` и `Lane<T>` с ограничением `T: LaneElement`. Элемент читается маскированной загрузкой одной полосы (`Lane::<T>::first_n(1)` и `load_select`): скалярное `col[row]` есть для конкретных типов `u32`, `i32`, `i64`, `f32`, `f64`, но не для обобщённого `T: LaneElement` — у `u16` нет скалярного регистра, а сравнение с comptime-значением `T` требует его типа при сборке. Режим `rust` печатает точку входа и все выбранные специализации. [Код примера](splice/examples/schema_plan.rs)

```sh
cargo run -p splice-kernel --release --example schema_plan
cargo run -p splice-kernel --release --example schema_plan -- rust
```

## Векторный SQL-план

В `sql_plan` дерево из `Scan`, `Filter`, `Project` и `Join` специализируется при сборке. Таблица хранит ключи и значения в двух непрерывных колонках. Одна полоса `Lane<i32>` соответствует одной строке. `Scan` читает пачку соседних строк, `Filter` вычисляет маску предиката и уплотняет обе колонки по этой маске, а `Project` вычисляет обе выходные колонки из одной исходной пачки. При последовательном чтении короткий маскированный пролог выравнивает адрес первого потока, хвост также проходит под маской. Начало второй колонки может иметь другое выравнивание.

При специализации каждый правый вход `Join` получает свой индекс с открытой адресацией и линейным пробированием. Строки с одинаковым ключом связаны в список значений. Левый вход проходит от скана через предикаты, проекции и соединения в виде SIMD-пачек. `gather` читает слоты полной пачки. Совпадения переходят к чтению значений, продолжающие поиск полосы уплотняются через `compress`, а `expand` заполняет свободные позиции новыми запросами. Чтение значений использует такую же дозагрузку, в том числе при обходе дубликатов. Когда вход исчерпан, оставшиеся полосы дочитываются скалярно. Хост передаёт отдельный срез состояния размером `plan.state_words()` слов для очередей SIMD и параметров индексов. В рабочем буфере размещаются правые хеш-индексы и итоговые пары `(key, value)`. Порядок выдачи после соединения свободный. [Код операторов](splice/examples/relational/kernels.rs) · [Специализация дерева](splice/examples/relational/pipeline.rs) · [План по пяти таблицам](splice/examples/relational/pricing.rs)

```sh
cargo run -p splice-kernel --release --example sql_plan -- lancy
cargo run -p splice-kernel --release --example sql_plan -- cranelift
cargo run -p splice-kernel --release --features llvm --example sql_plan -- llvm
cargo run -p splice-kernel --release --example sql_plan -- rust
cargo xtask bench sql_plan
cargo test -p splice-kernel --release --example sql_plan --test simd
```

Стенд Criterion.rs измеряет проекцию по 65 536 строкам и SQL-план с четырьмя хеш-соединениями. В таблице продаж 29 360 128 строк, в каждом из четырёх справочников по 1 835 008 товаров. Один вызов SQL-плана возвращает 11 010 048 строк. Данные фиксированы для всех бэкендов и готовятся до измерения.

Во всех группах стенда LLVM компилирует через `LlvmManualSimd`. На обоих путях сборки включён O3, а loop vectorization и SLP отключены. Явные SIMD-операции используют инструкции выбранного профиля. Исторические замеры SQL-плана в [docs/bench-history.md](docs/bench-history.md) относятся к прежней конфигурации с обычным O3.

Для одного SQL-плана выполните `cargo xtask bench sql_plan`. Фильтр `sql_plan/execute/build` выбирает вызовы ядер, собранных через `build::<B>`. HTML-отчёт и интервалы оценок доступны в `target/criterion/report/index.html`. [Код стенда](splice/benches/backends.rs) · [Как читать отчёт Criterion](https://criterion-rs.github.io/book/user_guide/command_line_output.html)

Каждый замер `compile/build` включает специализацию и установку нового исполняемого ядра. Для Cranelift этот путь включает построение переносимого IR. Группа `stage_ir` измеряет построение IR отдельно, а `compile/from_ir` передаёт всем трём бэкендам одну готовую функцию. Освобождение JIT-кода происходит после замера сборки. Группы `execute/build` и `execute/from_ir` повторно вызывают готовые ядра на одних входах с заранее выделенным выходным буфером. Для SQL-плана стенд до и после измерений проверяет значения и число вхождений каждой пары по отдельной скалярной формуле для исходных данных.

Текущие замеры всех бэкендов на каждом профиле — в [docs/benchmarks.md](docs/benchmarks.md): этот файл генерирует `cargo xtask bench-tables --run > docs/benchmarks.md` вместе с коммитом, версиями rustc, LLVM и Cranelift и набором расширений, который получил каждый бэкенд. Прежние замеры с другой машины и версий кода — в [docs/bench-history.md](docs/bench-history.md).

## SIMD в ядрах

`Lane<T>` представляет один вектор. JIT-ядро использует ширину цели из `BuildOptions`, обычная Rust-функция — ширину цели по умолчанию (`SPLICE_SIMD`), поэтому при сборке для цели по умолчанию они совпадают. `Lane::<T>::lanes()` возвращает число **элементов**. Например, для 32 байт это восемь `f32` или четыре `f64`, для 64 байт — шестнадцать или восемь. В переносимом IR размер вектора определяется типом элемента при компиляции бэкендом.

```rust
use splice::{kernel, Executable};
use splice_lancy::Lancy;

#[kernel]
fn add_block(a: &[u16], b: &[u16], dst: &mut [u16]) -> i32 {
    use splice::simd::Lane;
    let x = Lane::load(a, 0);
    let y = Lane::load(b, 0);
    (x + y).store(dst, 0);
    Lane::<u16>::lanes() as i32
}

let n = splice::simd::Lane::<u16>::lanes();
let (a, b, mut dst) = (vec![u16::MAX; n], vec![1u16; n], vec![9u16; n]);
let jit = add_block::build::<Lancy>(add_block::BuildArgs {})?;
let count = jit.call(add_block::CallArgs { a: &a, b: &b, dst: &mut dst })?;
assert_eq!(count as usize, n);
assert_eq!(dst, vec![0u16; n]);
```

- Элементы векторов и срезов поддерживают `u16`, `u32`, `i32`, `i64`, `f32` и `f64`. `+`, `-` и `*` для целых выполняются по модулю разрядности. Целые поддерживают побитовые операции, вещественные — деление, `sqrt` и `mul_add` с одним округлением. `splat` заполняет полосы одним значением, включая runtime-параметр `i32`. `iota()` нумерует их с нуля.
- `cast::<T>()` преобразует полосы одинаковой ширины по правилам Rust. `widen_low::<T>()` и `widen_high::<T>()` расширяют половину входного вектора. `narrow::<T>(high)` сужает и объединяет два вектора в порядке от младших полос к старшим. `reduce_sum()` сворачивает полосы `i32` с переполнением по модулю 2³².
- Поэлементные сравнения `simd_eq/ne/lt/le/gt/ge` дают `Mask<T>`. Маску, полученную сравнением целочисленных индексов, можно применять к вещественным данным той же разрядности. Маски поддерживают `&`, `|`, `^`, `!`, `any()`, `all()` и `select(yes, no)`. Для `a.add_masked(mask, b, fallback)` каждая выключенная полоса берётся из соответствующей позиции `fallback`.
- Маска `Lane::<T>::first_n(remaining)` включает первые `min(remaining, Lane::<T>::lanes())` полос. Для хвоста используйте `Lane::load_select(src, offset, mask, fallback)` и `value.store_select(dst, offset, mask)`. Смещения и индексы измеряются в элементах. `Lane::gather(src, indices)` и `value.scatter(dst, indices)` принимают столько индексов, сколько полос в данных. Для `u16` это `Lane<u16>`, для `i32/f32` — `Lane<i32>`, для `i64/f64` — `Lane<i64>`. Маскированные формы — `Lane::gather_select(src, mask, indices, fallback)` и `value.scatter_select(dst, mask, indices)`.
- Выключенная полоса не обращается к памяти. Записи идут по возрастанию номера полосы. При повторении индекса остаётся последняя выполненная запись. Первый активный индекс за границей завершает Rust-вызов паникой, JIT-вызов возвращает ошибку. Оба сохраняют все записи до этой полосы, в том числе при неполном последовательном `store`.

Для `Lane<i32>` доступны `compress(mask)` с нулевым хвостом и `compress_store(dst, offset, mask)` с записью ровно `mask.count()` элементов в порядке исходных полос. `prefix_sum()` вычисляет включительную префиксную сумму с переполнением по модулю 2³². `expand(mask, fallback)` заполняет выбранные позиции последовательными элементами из начала вектора. Остальные позиции сохраняют значения из `fallback`. `extract_wrapping` и `replace_wrapping` читают и заменяют полосу по индексу, взятому по модулю числа полос. `last()` извлекает последнюю полосу, `swizzle(indices)` возвращает ноль для индексов за пределами вектора. `interleave_low/high` и `deinterleave_even/odd` переводят пары колонок в чередующийся формат и обратно. `>>` с известным при специализации числом бит выполняет арифметический сдвиг вправо. Методы `simd_min` и `simd_max` сравнивают полосы как знаковые `i32`. `rotate_left(count)` циклически сдвигает биты каждой полосы на известное при специализации число по модулю 32. `Lane::<i32>::alignment_prefix(slice, offset)` возвращает число элементов до ближайшего выровненного адреса, включая ноль для уже выровненного входа.

Вспомогательный `#[kernel]` может принимать и возвращать `Lane<T>` или `Mask<T>`. При специализации такой вызов встраивается в вызывающее ядро. Для локального результата другого векторного ядра укажите тип, например `let keep: Mask<i32> = matches_rows(pred, keys, values);`.

[Примеры и сквозные тесты](splice/tests/simd.rs) · [Highway о числе полос и маскированном доступе](https://google.github.io/highway/en/master/quick_reference.html) · [Rust Portable SIMD о gather_select](https://doc.rust-lang.org/core/simd/struct.Simd.html#method.gather_select)

### Два SIMD-ядра в работе

[`simd_matmul`](splice/examples/workloads/matmul.rs) перемножает две прямоугольные матрицы `i32` в построчном формате. Одна полоса накапливает один столбец результата. Ядро размножает `A[row, k]` через `splat` и читает соседние элементы строки B одним вектором. Блок из 16 строк A переиспользует панель B глубиной до 32 элементов и шириной до четырёх векторов. Остаток строки обрабатывается под маской. Арифметика выполняется по модулю 2³².

[`simd_hash_lookup`](splice/examples/workloads/probe.rs) ищет пачку ключей в таблице с линейным пробированием. Каждая полоса вычисляет начальный слот своего ключа. `gather_select` читает очередной слот для активных полос, найденные позиции сохраняются, а совпадение или пустая ячейка выключает полосу. После полного обхода поиск завершается и в заполненной таблице. Ёмкость таблицы — степень двойки, ноль обозначает пустую ячейку. Результат запроса — номер слота или `-1`.

```sh
cargo run -p splice-kernel --release --example simd_matmul
cargo run -p splice-kernel --release --example simd_hash_lookup
cargo test -p splice-kernel --release --example simd_matmul --example simd_hash_lookup
cargo xtask bench simd_
```

Criterion использует те же функции, что примеры и тесты. Для матриц нагрузка равна 96×128 на 128×128, для поиска — 32 768 запросов к таблице из 16 384 слотов при заполнении 75% и доле попаданий 50%. Наборы данных детерминированы. Перед измерением и после него стенд сверяет весь выход с независимой скалярной реализацией. Группы `compile` и `execute` измеряют сборку и повторный вызов отдельно для каждого из двух путей — `build` и `from_ir`.

Замеры — в [docs/benchmarks.md](docs/benchmarks.md), прежние — в [docs/bench-history.md](docs/bench-history.md).

### Большие скалярные ядра

Четыре скалярных ядра используют фиксированные наборы данных и сверяют весь выход с независимым эталоном до и после замера. Размер указан в инструкциях точки входа переносимого IR после специализации.

| Нагрузка | Данные и работа ядра | Инструкции IR |
| --- | --- | ---: |
| [`large_audio_low_pass`](splice/benches/large/biquad.rs) | 48 000 PCM-отсчётов, ФНЧ Баттерворта 16-го порядка из восьми Q14 biquad-секций. Хост передаёт два слова состояния на секцию | 266 |
| [`large_parquet_delta`](splice/benches/large/delta.rs) | 32 769 значений `INT32`, распаковка и префиксная сумма после разбора заголовков. 256 полных блоков с ширинами мини-блоков 7, 11, 17 и 23 бита. [Формат Parquet](https://parquet.apache.org/docs/file-format/data-pages/encodings/#delta-encoding-delta_binary_packed--5) | 2 788 |
| [`large_per_channel_requantize`](splice/benches/large/requantize.rs) | 4 096 позиций по 16 каналов, поканальные коэффициенты Q12 для активации и residual, округление и насыщение до `int8` | 356 |
| [`large_decimal_csv`](splice/benches/large/csv.rs) | 8 192 строки по четыре знаковых значения с тремя дробными разрядами, упаковка ASCII в `i32` | 362 |

```sh
cargo xtask bench 'large_(audio|parquet|per_channel|decimal)' --warm-up-time 1 --measurement-time 2 --sample-size 30 --noplot
```

Замеры — в [docs/benchmarks.md](docs/benchmarks.md), прежние — в [docs/bench-history.md](docs/bench-history.md).

### Большие SIMD-ядра

Группы `large_simd_*` обрабатывают полные векторы и хвост под маской. Стенд сверяет каждое слово результата со скалярным эталоном, включая оба пути сборки.

| Нагрузка | Данные и работа ядра |
| --- | --- |
| [`large_simd_yuv_to_rgba`](splice/benches/large/vector/yuv.rs) | 262 147 пикселей, преобразование Y′CbCr 4:4:4 в RGBA8 с целочисленной цветовой матрицей и насыщением |
| [`large_simd_integral_moments`](splice/benches/large/vector/integral.rs) | Кадр 515×257, две интегральные таблицы для локального среднего и дисперсии |
| [`large_simd_murmur3`](splice/benches/large/vector/hash.rs) | 65 539 составных ключей из восьми колонок `INT32`, один ключ на полосу. [Эталон MurmurHash3 x86_32](https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp#L88-L138) |
| [`large_simd_remap_rgb16`](splice/benches/large/vector/remap.rs) | Плитка 145×145, три 16-битных канала, перенос пикселей по поканальным картам радиальной коррекции |

```sh
cargo xtask bench large_simd_ --warm-up-time 0.5 --measurement-time 1 --sample-size 20 --nresamples 10000 --noplot
```

Для сравнения исполнения на другой ширине задайте `SPLICE_SIMD=avx2` или `SPLICE_SIMD=sse` и замените фильтр `large_simd_` на `'large_simd_.*/execute/'`.

Замеры — в [docs/benchmarks.md](docs/benchmarks.md), прежние — в [docs/bench-history.md](docs/bench-history.md).

## Кернелы для сравнения оптимизаций

Сценарии `gap_` измеряют сборку и повторный вызов отдельно для прямого `build` и переносимого IR. Стенд сверяет весь записанный буфер с эталоном для каждого бэкенда. Во всех замерах LLVM используется `llvm_manual_simd`. [Код кернелов и эталонов](splice/benches/gaps/mod.rs)

| Сценарий | Что нагружает |
| --- | --- |
| `gap_tone_passes`, `gap_tone_fused` | Три последовательных прохода по 65 539 пикселям и та же кривая с ручным объединением проходов |
| `gap_local_energy` | Два накопителя во вложенном цикле по окну из 31 отсчёта с периодической границей |
| `gap_feature_score` | Три вызова общей функции с фиксированными режимами и нечётными делителями |
| `gap_alias_reload`, `gap_alias_hoisted` | Повторное чтение коэффициента после записи в отдельный срез и ручной вынос всех четырёх загрузок |
| `gap_ramp_integral` | Суммирование линейных последовательностей с длиной, начальным значением и шагом из входного буфера |
| `gap_envelope_state` | Четыре изменяемых локальных состояния с условными обновлениями внутри цикла. Lancy получает SSA напрямую, LLVM продвигает свои временные слоты в регистры |
| `gap_loop_conditional_scale` | Условная обработка 65 539 отсчётов с масштабом, зависящим от размеров буферов |
| `gap_loop_dead_checksum` | Копирование 65 539 отсчётов со специализированным отключением результата контрольной суммы из внутреннего цикла |

```powershell
cargo xtask bench '^gap_' --warm-up-time 0.3 --measurement-time 0.7 --sample-size 15 --nresamples 10000 --noplot
```

Замеры — в [docs/benchmarks.md](docs/benchmarks.md), прежние — в [docs/bench-history.md](docs/bench-history.md).

Фронтенд передаёт виды аргументов `BufferKind` из `&[T]` и `&mut [T]` при прямой сборке и в переносимом IR. Для двух разных параметров `BufferKind::may_alias` даёт консервативный ответ о возможном пересечении. Два среза для чтения могут иметь общие элементы, изменяемый срез имеет исключительный доступ. LLVM использует этот контракт для `readonly` и `noalias`. Lancy получает сведения о срезах через проверенный интерфейс Splice. На `gap_alias_reload` они позволяют выносить чтение коэффициентов за пределы цикла после проверки длины.

Проход памяти Lancy переиспользует загрузку из среза только для чтения и после записи в другой срез, короткого пути векторной записи или Rust-вызова: такой срез не меняется, пока ядро работает. Первая проверка границ остаётся на месте. На `gap_alias_reload` это убирает одну из шести загрузок за итерацию.

## Время этапов компиляции

Splice публикует вложенные спаны `tracing` уровня `DEBUG` с target
`splice::compile` (бэкенды — `splice_core::compile`, сам Lancy — `lancy::compile`). Для замеров добавьте в приложение `tracing-subscriber = "0.3"`
и установите подписчик до вызова `build::<B>` или `B::compile`.

```rust
use tracing_subscriber::fmt::format::FmtSpan;

tracing_subscriber::fmt()
    .with_max_level(tracing_subscriber::filter::LevelFilter::DEBUG)
    .with_span_events(FmtSpan::CLOSE)
    .init();
```

При закрытии спана поле `time.busy` показывает его длительность, включая вложенные
этапы. У Lancy отдельно видны проверка HIR, инлайнинг и оптимизация, выбор инструкций,
распределение регистров, генерация и установка кода. У LLVM видны построение и
проверка IR, O3, генерация объектного кода и линковка. Если этап выполняется для
нескольких функций или повторяется при компиляции, он появится в выводе несколько
раз. Спаны работают без зависимости от Tokio.

## Отладка JIT-ядер

`kernel::build::<B>(kernel::BuildArgs {})` и `kernel::ir()` выбирают режим по `debug_assertions` вызывающего крейта. В обычной debug-сборке Lancy, Cranelift, LLVM и LLVM Manual SIMD сохраняют исходные строки, значения переменных на этих строках и отдельные кадры скалярных вызовов. В release-сборке действуют оптимизации бэкенда. Режим и цель можно явно задать первым аргументом `build_with_options::<B>(BuildOptions::new(Mode::Debug, target), ...)` или `ir_with_options(BuildOptions::host(Mode::Debug)?, ...)`. При компиляции переносимого IR бэкенд по умолчанию использует записанные в нём режим и цель. `B::compile_with_options(&ir, BuildOptions::new(Mode::Release, target))` позволяет отдельно запустить оптимизированную компиляцию, в том числе для другой цели.

В GDB или IDE с отладчиком, поддерживающим GDB JIT Interface, можно остановиться по имени `splice_jit::<имя>`, пройти по строкам исходного Rust и посмотреть значения через `info locals`. Для точки останова в файле символы появляются после сборки ядра. Срезы показывают указатель и длину, векторы показывают полосы, а маски используют целочисленное представление полос. В отладочной информации также доступны неизменяемые примитивные значения специализации. Сложные значения `#[comptime]` и другие данные, существующие только на хосте, можно посмотреть в кадре специализации. SIMD-помощники, которые фронтенд разворачивает в вызывающую функцию, используют её машинный кадр.

Список GDB JIT Interface (`__jit_debug_descriptor`) общий для процесса, им владеет `splice-jit`. Поэтому Cargo не допускает две версии `splice-jit` в одном графе (`links = "splice_jit_gdb"`). Если LLVM слинкован статически, используются его определения символов. Отладочную поддержку ORC и GDB-слушатель LLVM включать нельзя: LLVM писал бы в тот же список без замка `splice-jit`.

```sh
cargo build -p splice-kernel --example debug_kernel
gdb --args target/debug/examples/debug_kernel lancy
```

```gdb
set breakpoint pending on
break splice_jit::child
run
next
info locals
backtrace
```

Пример также принимает `cranelift`, `llvm` и `llvm_manual_simd`. Для профилирования Linux используйте `SPLICE_PERF_MAP=1`. Все нативные бэкенды записывают имена и диапазоны в `/tmp/perf-PID.map`, в том числе при release-сборке. Формат perf-map привязывает имя к адресу. При многократной пересборке и освобождении ядер повторно использованный адрес может объединить сэмплы разных ядер.

```sh
cargo build -p splice-kernel --example debug_kernel --release
SPLICE_PERF_MAP=1 perf record -- target/release/examples/debug_kernel lancy 1000000
perf report --stdio
```

## Сборка

Нужен Rust 1.94+. На Windows x86-64, Linux x86-64 и Linux AArch64 доступны Lancy, Cranelift и LLVM. Бэкенды Lancy и Cranelift собираются отдельно от LLVM. Для приложения с Cranelift включите feature `cranelift` пакета `splice-kernel`.

Для `splice-llvm` на всех платформах выбран LLVM 19.1, поэтому тесты и замеры на разных ОС сравнивают один и тот же компилятор.

* **Windows x64 MSVC.** Сборка находит LLVM в стандартном каталоге или в компонентах C++ Clang для Visual Studio. Для другой папки установки задайте `LLVM_SYS_191_PREFIX` до сборки. В ней должны быть `lib/LLVM-C.lib` и `bin/LLVM-C.dll`. Cargo подготавливает DLL для `cargo run`, тестов и бенчмарков. Для самостоятельного запуска готового приложения положите соответствующую DLL рядом с исполняемым файлом или добавьте её каталог в `PATH`.
* **Linux x86-64 и AArch64.** LLVM линкуется динамически, а если `llvm-config` не сообщает о `libLLVM.so` — статически (сборку со статическим LLVM проверяет `cargo xtask static-llvm`). Нужны инструменты разработки, включая `llvm-config-19`, и разделяемая библиотека LLVM; в Ubuntu 24.04 достаточно пакета `llvm-19-dev`. Для другой установки укажите префикс через `LLVM_SYS_191_PREFIX`, например `/usr/lib/llvm-19`.

Без LLVM работают `cargo test -p splice-kernel` (Lancy и Cranelift) и тесты остальных крейтов, кроме `splice-llvm` и `splice-testkit`. Те же проверки, что в CI, запускает `cargo xtask ci` (`--quick` — только rustfmt, Clippy и тесты воркспейса); список команд — в [xtask](xtask/src/main.rs).

Версии Inkwell и LLVM указаны в [манифесте бэкенда](splice-llvm/Cargo.toml). Операции над LLVM-инструкциями, типами, значениями и базовыми блоками используют [Inkwell Builder](https://docs.rs/inkwell/0.11.0/inkwell/builder/struct.Builder.html).

### Выбор SIMD-профиля

Цель сборки — значение `splice::Target` в `BuildOptions { mode, target }`: набор расширений ISA, ширина вектора и соглашение о вызовах. `Target::from_profile(profile)` строит цель профиля, и в одном процессе можно собирать ядра для нескольких целей. `SPLICE_SIMD` задаёт только цель по умолчанию (`Target::host()`), которую берут `kernel::build`, `kernel::ir` и обычная Rust-функция. `auto` выбирает профиль по возможностям CPU и ОС. Явный профиль позволяет проверить более раннюю ISA на новом процессоре: `sse` — SSE2, `avx2` — x86-64-v3, `avx512` — x86-64-v4, и на любой машине это один и тот же набор. Если CPU или ОС не поддерживают цель, JIT возвращает ошибку сборки. Обычная Rust-функция сообщает об ошибочном `SPLICE_SIMD` паникой при запросе полос.

| Платформа | `SPLICE_SIMD` | Логическая ширина |
| --- | --- | --- |
| x86-64 | `sse`, `avx2`, `avx512` | 16, 32, 64 байта соответственно |
| x86-64 | `avx10` | По CPUID и состоянию XSAVE. Для LLVM нужны хотя бы 32 байта, LLVM использует набор AVX10.1. |
| Linux AArch64 | `neon` | 16 байт |
| Linux AArch64 | `sve`, `sve2` | 16, 32 или 64 байта, в пределах SVE VL потока при первом использовании SIMD |

На AArch64 `auto` предпочитает SVE2, затем SVE и NEON. Lancy выбирает машинные формы в собственном `isel` и кодирует их своим энкодером. LLVM выбирает инструкции SVE/SVE2 для фиксированных векторов, когда позволяет их ширина. Для 128-битных векторов LLVM может выбрать NEON. Lancy и LLVM проверяют SVE VL вызывающего потока перед выполнением ядра. Cranelift использует 128-битные части общего логического вектора. Все бэкенды переводят одну цель: Lancy получает её расширения как есть, LLVM — базовый CPU `x86-64`/`generic` со строкой признаков цели, Cranelift — соответствующие флаги ISA; CPUID ни один из них не спрашивает. Cranelift и LLVM собирают только соглашение о вызовах C этой ОС, Lancy — также Win64 и SysV на любой x86-64 ОС. [Бэкенд Lancy](lancy/README.md#бэкенд-aarch64)

В PowerShell задайте профиль перед вызовом Cargo, например `$env:SPLICE_SIMD = 'avx2'`.

```sh
# На x86-64 с AVX2
SPLICE_SIMD=avx2 cargo run -p splice-target --example simd_profile
SPLICE_SIMD=avx2 cargo test -p splice-kernel --release --test simd

# На Linux AArch64
SPLICE_SIMD=neon cargo test --workspace --release

# На Linux AArch64 с SVE2
SPLICE_SIMD=sve2 cargo test -p splice-llvm --release
SPLICE_SIMD=sve2 cargo test -p splice-kernel --release --test simd

# На Linux AArch64 с QEMU user-mode
SPLICE_SIMD=sve2 \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER='qemu-aarch64 -cpu max,sve-default-vector-length=64' \
cargo test -p splice-llvm --release --target aarch64-unknown-linux-gnu
```

GitHub Actions проверяет NEON на нативном ARM64-раннере и запускает SVE/SVE2 под QEMU с длинами 32, 48 и 64 байта. На x86-64 CI выполняет SSE и остальные профили, которые ОС видит на процессоре раннера. Тест SVE2 проверяет наличие `eor3` в листинге LLVM и сравнивает результат обоих путей компиляции с обычным поэлементным вычислением. [Workflow](.github/workflows/simd.yml)
