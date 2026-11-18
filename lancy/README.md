# Lancy JIT

Lancy компилирует функции над `i32` и срезами `i32` в машинный код для Windows и Linux на x86-64. У скалярной функции, цикла по срезу и составного SQL-запроса один HIR и общие проходы компилятора. Текущий x64-бэкенд выпускает для этих циклов скалярные инструкции. Автоматической SIMD-векторизации в этой версии пока нет.

## Специализация по конфигурации

```rust
use lancy::{kernel, ScalarFn};

enum Expr { Add(i32), Scale(i32) }

const fn factor(mut exponent: i32) -> i32 {
    let mut value = 1;
    while exponent > 0 {
        value *= 2;
        exponent -= 1;
    }
    value
}

#[kernel]
fn project(#[comptime] expr: &Expr, price: i32) -> i32 {
    match expr {
        Expr::Add(n) => price + *n,
        Expr::Scale(exponent) => price * factor(*exponent),
    }
}

let plan = Expr::Scale(3);
let input = project::build(&plan)?;
let native = ScalarFn::<1>::compile(&input)?;
assert_eq!(native.invoke([5]), project(&plan, 5));
assert_eq!(native.invoke([-1]), -8);
# Ok::<(), String>(())
```

`#[comptime]` означает, что значение известно в момент построения ядра. План может поступить после запуска процесса. Для `Scale(3)` выбор ветки и вычисление коэффициента завершаются при построении, а машинный код умножает входную цену на восемь. Исходную Rust-функцию можно вызывать для проверки результата.

`lancy::kernel` предоставляет собственный макрос крейта. Сгенерированный `project::build` возвращает регистровое представление `lancy::input::Function`, которое принимает `ScalarFn::compile`. В проектах с несколькими бэкендами макрос `splice::kernel` позволяет выбрать компилятор при построении. Вызов `project::build::<splice_lancy::Lancy>(&plan)` направляет операции сразу в построитель Lancy. Вызов `project::ir(&plan)` материализует переносимый регистровый IR Splice.

### Вызовы других ядер

```rust
use lancy::{kernel, ScalarFn};

#[kernel]
fn discount(floor: i32, price: i32) -> i32 {
    if price > floor { price - floor } else { 0 }
}

#[kernel]
fn invoice(#[comptime] rate: i32, price: i32) -> i32 {
    let floor = rate * 2;
    discount(floor, price)
}

let code = ScalarFn::<1>::compile(&invoice::build(3)?)?;
assert_eq!(code.invoke([10]), invoice(3, 10));
# Ok::<(), String>(())
```

В `invoice::build(3)` выражение `rate * 2` даёт 6. Вызов `discount` получает константу и значение цены. Если вызываемое ядро объявляет параметр `#[comptime]`, его аргумент должен быть известен при построении.

```compile_fail
use lancy::kernel;
#[kernel] fn bounded(#[comptime] bound: i32, x: i32) -> i32 { x - bound }
#[kernel] fn invalid(x: i32, y: i32) -> i32 { bounded(x, y) }
```

Ранние выходы, короткое замыкание, ветвления, вложенные вызовы и циклы доступны обычным скалярным ядрам. Цикл по известной при построении коллекции может развернуть последовательность операций над runtime-значением. [Политика повторных попыток](examples/retry_policy.rs) · [Проверки языка](tests/scalar_language.rs) · [Проверки специализации](tests/staging.rs)

### Запрос по дереву плана

В примере `sql_plan` у каждого `Join(left, right, combine)` два независимых дочерних плана. На любом уровне возможны `Scan`, `Filter`, `Project` и следующий `Join`. Пример считает стоимость продаж по пяти таблицам с ключом `product_id`.

```text
Project(adjusted_value + 1)
└─ Join(+)
   ├─ Join(×)
   │  ├─ Filter(quantity >= 2, Scan(sales))
   │  └─ Scan(prices)
   └─ Join(×)
      ├─ Join(+)
      │  ├─ Filter(enabled = 1 AND product_id >= 0, Scan(eligibility))
      │  └─ Scan(promotions)
      └─ Project(product_id, min(adjustment * 2, 20), Scan(adjustments))
```

`query::build::<Lancy>(&plan)` специализирует дерево. Полученное ядро принимает каталог смещений, пары `(key, value)` и рабочий срез. Оно выполняет сканирование, фильтрацию, проекции, построение хеш-таблиц и обход цепочек совпадений. Результат лежит в начале рабочего среза, возвращаемое число задаёт количество строк. При изменении размеров или содержимого таблиц тот же код можно вызвать снова.

`Project` может изменить оба поля, в том числе ключ следующего соединения. Внутреннее эквисоединение выдаёт строку для каждой пары совпадений, включая дубликаты. Дочерние результаты материализуются в рабочем срезе. `Catalog::run` сообщает об ошибке при нехватке места. [Ядра операторов](../splice/examples/relational/kernels.rs) · [План по пяти таблицам](../splice/examples/relational/pricing.rs) · [Запуск запроса](../splice/examples/sql_plan.rs)

### Что взяли из CubeCL

CubeCL поддерживает enum, вариант которого известен при построении ядра. В Splice это позволяет выбирать операторы SQL-плана через обычный `match` и встраивать данные из выбранного варианта в ядро. Условие по входной строке остаётся ветвлением в сгенерированном коде. [Enum в CubeCL](https://burn.dev/books/cubecl/language-support/enum.html)

## От HIR до машинного кода

| Стадия | Представление |
| --- | --- |
| HIR | В `lancy::hir::Function` каждое значение определяется один раз. Параметры базового блока получают значения с входящих рёбер, включая обратное ребро цикла. Инструкции, блоки, значения и рёбра имеют отдельные идентификаторы. [SSA HIR](src/hir/ssa/mod.rs) |
| Редактирование и оптимизация | `replace_op` и `replace_ops` сохраняют идентификаторы инструкций и их результатов. `merge_blocks` объединяет совместимые блоки и подставляет аргументы ребра. Общий проход сворачивает константы и объединяет линейные последовательности блоков. [Редактирование](src/hir/ssa/edit.rs) · [Проходы](src/hir/passes.rs) |
| LIR до regalloc | Инструкции лежат в массиве, операнды всё ещё обозначают SSA-значения. Вызовы и списки аргументов переходов вынесены из основного потока. Сравнение с единственным использованием в переходе может стать самим условием перехода. [LIR](src/lir/ssa.rs) |
| LIR после regalloc | Линейный скан учитывает живость значений, в том числе на обратных рёбрах. Машинные операции обращаются к физическим регистрам, слотам рабочей памяти и непосредственным значениям. Передача параметров блока превращается в перемещения на конкретном ребре. На этом этапе SSA заканчивается. [Аллокатор x64](src/lir/x64/scalar/regalloc.rs) · [Машинный LIR](src/lir/x64/scalar/machine.rs) |

Вручную можно построить SSA-граф, проверить его и передать в JIT. В примере `join` принимает значение, выбранное предыдущим блоком.

```rust
use lancy::{hir::{Binary, Function, Op, Terminator}, ScalarKernel};

let mut f = Function::new(1, vec![]);
let entry = f.entry();
let x = f.params(entry)[0];
let join = f.create_block();
let selected = f.append_param(join);
let zero = f.append(entry, Op::Const(0)).unwrap();
let positive = f.append(entry, Op::Bin(Binary::Gt, x, zero)).unwrap();
let yes = f.add_edge(join, &[x]);
let no = f.add_edge(join, &[zero]);
f.set_terminator(entry, Terminator::Branch { cond: positive, yes, no });
f.set_terminator(join, Terminator::Return(selected));

f.verify()?;
let code = ScalarKernel::compile_hir(&f)?;
assert_eq!(code.call(&[-3])?, 0);
assert_eq!(code.call(&[7])?, 7);
# Ok::<(), String>(())
```

`ScalarKernel::baseline_hir` сразу выпускает исполняемый код. `ScalarKernel::compile_hir` сначала прогоняет общие оптимизации HIR. Аргументы JIT могут быть числами `i32` и срезами `&[i32]` или `&mut [i32]`, результат имеет тип `i32`. В `call_with_buffers` числа и срезы передаются двумя списками в порядке из сигнатуры. Сложение, вычитание и умножение работают по модулю 2³². При недопустимом индексе, делении на ноль или переполнении знакового деления вызов возвращает ошибку. Предшествующие записи остаются видимыми.

## Фоновая компиляция

`AdaptiveBufferKernel::compile_hir` принимает SSA-граф, устанавливает базовую версию и заказывает оптимизированную в фоновом потоке. Каждый вызов выбирает готовую версию на входе. Для переносимого плана Splice путь к этому API начинается с `splice_lancy::lower(&query::ir(&plan)?)?`.

```rust
use lancy::{kernel, hir, AdaptiveBufferKernel, BufferArg};

#[kernel]
fn increment(out: &mut [i32]) -> i32 {
    out[0] += 1;
    out[0]
}

let hir = hir::from_input(&increment::build()?)?;
let kernel = AdaptiveBufferKernel::compile_hir(hir)?;
let mut out = [40];
assert_eq!(kernel.call_with_buffers(&[], &mut [BufferArg::Write(&mut out)])?, 41);
kernel.wait_for_optimization()?;
assert_eq!(kernel.call_with_buffers(&[], &mut [BufferArg::Write(&mut out)])?, 42);
# Ok::<(), String>(())
```

Для функций со скалярными аргументами `AdaptiveScalar<N>::with_assumptions` принимает ожидаемые значения. `profiled` наблюдает первые 32 вызова и выбирает до четырёх стабильных аргументов. Когда проверка на входе перестаёт выполняться, вызов переходит на универсальный машинный код. [Фоновая компиляция функций со срезами](src/jit/scalar/adaptive_buffer.rs) · [Специализация по скалярным аргументам](src/jit/scalar/adaptive.rs)

## Запуск

Нужны Rust 1.94 или новее и x86-64. Примеры с обоими бэкендами используют системную библиотеку LLVM. Для сравнительных сценариев с C++ нужен компилятор C++17. Команды запускаются из корня workspace. [Зависимости и настройка](../README.md)

```sh
cargo test --release -p lancy
cargo run --release -p lancy --example retry_policy
cargo run --release -p lancy --example sql_kernel
cargo run --release -p splice --example sql_plan -- lancy
cargo bench -p splice --bench backends
cargo run --release -p lancy-customers -- joins
```
