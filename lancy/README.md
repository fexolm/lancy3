# Lancy

Lancy компилирует SSA-функции над `i32` и машинными указателями `ptr` для x86-64 на Windows и Linux. Вход в JIT — `ScalarKernel::compile(&hir::ssa::Function)`. Компиляция синхронна и выпускает одну версию машинного кода.

## Функция с ветвлением

В примере блок `join` получает либо аргумент функции, либо ноль. В SSA значение, выбранное на переходе, передаётся через параметр блока.

```rust
use lancy::{hir::ssa::{Binary, Function, Op, Terminator}, ScalarKernel};

let mut f = Function::scalar(1);
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
# Ok::<(), Box<dyn std::error::Error>>(())
```

Ошибку верификации можно разобрать через `CompileError::InvalidIr(IrError)`. При сбое установки кода `CompileError::ExecutableMemory` сохраняет имя системного вызова и исходную `std::io::Error`. Вызовы готового кернела возвращают `CallError`.

## Путь от HIR до вызова

Указатель можно передать в блок или функцию, вернуть из вызова, прочитать из памяти и записать в память. Для `xs[i]` фронтенд Splice передаёт срез парой `(ptr, len)` и строит три инструкции.

```text
BoundsCheck(i, len)
addr = PtrOffset(ptr, i, Four)
value = Load(I32, addr)
```

`PtrOffset` расширяет знаковый индекс до ширины указателя перед умножением на шаг в байтах. `Load` указывает тип результата, а `Store` использует ширину записываемого значения. Оба допускают невыровненный адрес.

1. **HIR.** Функция хранит SSA-значения, инструкции, блоки и рёбра под типизированными индексами. Верификатор проверяет определения и использования значений, аргументы рёбер и сигнатуры вызовов. [SSA-функция](src/hir/ssa/function.rs) · [Граф потока управления](src/hir/transforms/cfg.rs) · [Дерево доминаторов](src/hir/transforms/domtree.rs)
2. **LIR.** Lowering переводит блоки и операции в нижнее представление. До распределения регистров операнды являются SSA-значениями. [Инструкции](src/lir/instruction.rs) · [Lowering](src/lir/ssa.rs)
3. **Regalloc и x64.** Аллокатор назначает регистры и слоты рабочей памяти с учётом живости значений на рёбрах, включая обратные переходы циклов. Параметры блоков превращаются в параллельные перемещения. Эмиттер кодирует инструкции и переходы. [Распределение регистров](src/lir/x64/regalloc.rs) · [Выход из SSA](src/lir/x64/machine.rs) · [Эмиссия x64](src/lir/x64/emit.rs)
4. **JIT.** Рантайм владеет исполняемым кодом до уничтожения `ScalarKernel`. Для сигнатуры из `i32` используется `call`. Для прямого вызова HIR с указателями используется `unsafe call_raw`: вызывающая сторона отвечает за валидность памяти. Безопасный интерфейс Splice принимает Rust-срезы и проверяет права доступа. Записи до ошибки индекса или деления видны вызывающей стороне. [ScalarKernel](src/jit/kernel.rs)

Путь компиляции обходится без оптимизирующих проходов. Специализация Rust-параметров `#[comptime]` выполняется во фронтенде Splice до передачи операций в Lancy. В приложении с макросами используйте `#[splice::kernel]` и `my_kernel::build::<splice_lancy::Lancy>(...)?`. [Язык и примеры Splice](../README.md) · [Политика повторных попыток](../splice/examples/retry_policy.rs)

```sh
cargo test -p lancy --release
cargo test -p splice-lancy --release
cargo run -p splice --release --example retry_policy
```
