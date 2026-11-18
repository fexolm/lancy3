# Lancy

Lancy компилирует SSA-функции над `i32` и срезами `i32` для x86-64 на Windows и Linux. Вход в JIT — `ScalarKernel::compile(&hir::Function)`. Компиляция синхронна и выпускает одну версию машинного кода.

## Функция с ветвлением

В примере блок `join` получает либо аргумент функции, либо ноль. В SSA значение, выбранное на переходе, передаётся через параметр блока.

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

let code = ScalarKernel::compile(&f)?;
assert_eq!(code.call(&[-3])?, 0);
assert_eq!(code.call(&[7])?, 7);
# Ok::<(), String>(())
```

## Путь от HIR до вызова

1. **HIR.** Функция хранит SSA-значения, инструкции, блоки и рёбра под типизированными индексами. Верификатор проверяет определения и использования значений, аргументы рёбер и сигнатуры вызовов. [Модуль HIR](src/hir/mod.rs)
2. **LIR.** Lowering переводит блоки и операции в нижнее представление. До распределения регистров операнды являются SSA-значениями. [Модуль LIR](src/lir/mod.rs)
3. **Regalloc и x64.** Аллокатор назначает регистры и слоты рабочей памяти с учётом живости значений на рёбрах, включая обратные переходы циклов. Параметры блоков превращаются в параллельные перемещения. Эмиттер кодирует инструкции и переходы. [Бэкенд x64](src/lir/x64/mod.rs)
4. **JIT.** Рантайм устанавливает исполняемый код и владеет его памятью до уничтожения `ScalarKernel`. Для срезов используется `call_with_buffers`. Вызов проверяет сигнатуру, а сгенерированный код проверяет индексы и ошибки целочисленного деления. Записи, выполненные до ошибки, видны вызывающей стороне. [Модуль JIT](src/jit/mod.rs)

Путь компиляции обходится без оптимизирующих проходов. Специализация Rust-параметров `#[comptime]` выполняется во фронтенде Splice до передачи операций в Lancy. В приложении с макросами используйте `#[splice::kernel]` и `my_kernel::build::<splice_lancy::Lancy>(...)?`. [Язык и примеры Splice](../README.md) · [Политика повторных попыток](../splice/examples/retry_policy.rs)

```sh
cargo test -p lancy --release
cargo test -p splice-lancy --release
cargo run -p splice --release --example retry_policy
```
