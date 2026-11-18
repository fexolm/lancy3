# Splice

Splice — язык для CPU-ядер, которые специализируются по данным приложения после его запуска. Функция с атрибутом `#[kernel]` сохраняет обычный вызов Rust. Макрос также создаёт фабрику JIT-кода. Тип в `build::<B>()` определяет, какой компилятор обработает ядро.

```rust
use splice::{Executable, kernel};
use splice_cranelift::Cranelift;
use splice_lancy::Lancy;
use splice_llvm::Llvm;
use splice_text::Text;

enum Price { Retail, Wholesale(i32) }

#[kernel]
fn total(#[comptime] plan: &Price, price: i32, qty: i32) -> i32 {
    match plan {
        Price::Retail => price.wrapping_mul(qty),
        Price::Wholesale(discount) => price.wrapping_sub(*discount).wrapping_mul(qty),
    }
}

fn main() -> Result<(), String> {
    let plan = Price::Wholesale(3);
    let fast_build = total::build::<Lancy>(&plan)?;
    let cranelift = total::build::<Cranelift>(&plan)?;
    let llvm = total::build::<Llvm>(&plan)?;
    let listing: String = total::build::<Text>(&plan)?;
    print!("{listing}");
    assert_eq!(fast_build.call(&[10, 4], &mut [])?, 28);
    assert_eq!(cranelift.call(&[10, 4], &mut [])?, 28);
    assert_eq!(llvm.call(&[10, 4], &mut [])?, 28);
    Ok(())
}
```

Вызов `build::<Lancy>` строит SSA HIR Lancy прямо при обходе кернела. Вызов `build::<Llvm>` таким же образом строит LLVM IR. Вызов `build::<Cranelift>` строит и проверяет переносимый IR, затем компилирует его в машинный код через Cranelift. Хостовый Rust выбирает вариант `match` по плану, вычисляет известные выражения и разворачивает `for` по `#[comptime]`-срезу. Параметры специализации могут содержать заимствования и пользовательские enum.

`build::<Text>` возвращает текст функции после специализации. В примере сигнатура содержит только `price` и `qty`, а скидка 3 появляется в теле перед умножением. В листинге сохраняются имена параметров, операции со срезами и вызовы. Если один помощник вызван с разными планами, его тела получают разные номера. Переходы между метками показывают оставшиеся runtime-ветвления и циклы. Листинг отражает семантику Splice до оптимизаций машинного бэкенда.

Сигнатура фабрики — `build::<T>(...) -> splice::Result<T::Output>`. У `Text` тип результата успешной сборки — `String`, у Lancy, Cranelift и LLVM — исполняемый кернел. Ошибки специализации и кодогенерации возвращаются через `Err(String)`. Для сравнения бэкендов на одном переносимом IR можно вызвать `let ir = total::ir(&plan)?;`, а затем `Lancy::compile(&ir)?`, `Cranelift::compile(&ir)?`, `Llvm::compile(&ir)?` или `Text::compile(&ir)?`, импортировав трейт `splice::Codegen`.

| Крейт | Ответственность |
| --- | --- |
| [splice](splice/src/lib.rs) | Общий фронтенд, построение и проверка переносимого IR, трейты `Codegen` и `Executable`. |
| [splice-macros](splice-macros/src/lib.rs) | Макрос `#[kernel]` разбирает Rust-синтаксис и генерирует обращения к фронтенду. |
| [splice-lancy](splice-lancy/src/lib.rs) | Переводит поток операций Splice или готовую переносимую функцию в SSA HIR и вызывает JIT Lancy. |
| [splice-cranelift](splice-cranelift/src/lib.rs) | Переводит проверенную функцию Splice и её вызываемые функции в Cranelift IR. Cranelift JIT компилирует их для процессора хоста с настройкой `opt_level=speed`. |
| [splice-llvm](splice-llvm/src/lib.rs) | Строит LLVM-модуль через типизированный Rust API Inkwell, оптимизирует его с `default<O3>` для процессора хоста и устанавливает машинный код через ORC LLJIT. `to_llvm_ir(&ir)` печатает модуль для отладки. |
| [splice-text](splice-text/src/lib.rs) | Печатает специализированную функцию и её вызываемые функции. Для использования достаточно зависимостей `splice` и `splice-text`. |
| [lancy](lancy/README.md) | Самостоятельный x64-бэкенд. Проверяет SSA HIR, строит LIR, распределяет регистры и устанавливает машинный код. |

Скалярные аргументы `i32` и срезы `&[i32]` / `&mut [i32]` передаются двумя списками с сохранением порядка внутри каждого списка. Все три компилятора поддерживают вызовы других ядер, циклы, ранний возврат, проверяемые обращения к срезам и целочисленное деление. Обработка колонок выражается обычным циклом над срезами.

## Полный запрос и сравнение

В примере `sql_plan` один исходный кернел обходит `#[comptime]`-дерево из `Scan`, `Filter`, `Project` и бинарных `Join`. При каждом вызове готовая функция читает таблицы, строит хеш-индексы и возвращает все совпадения. [Код ядер](splice/examples/relational/kernels.rs) · [Дерево по пяти таблицам](splice/examples/relational/pricing.rs)

```sh
cargo run -p splice --release --example sql_plan -- lancy
cargo run -p splice --release --example sql_plan -- cranelift
cargo run -p splice --release --example sql_plan -- llvm
cargo run -p splice --release --example sql_plan -- text
cargo bench -p splice --bench backends
cargo test --workspace --release
```

Стенд Criterion.rs измеряет проекцию по 65 536 строкам и SQL-план с четырьмя хеш-соединениями. В таблице продаж 29 360 128 строк, в каждом из четырёх справочников по 1 835 008 товаров. Один вызов SQL-плана возвращает 11 010 048 строк. Данные фиксированы для всех бэкендов и готовятся до измерения.

Для одного SQL-плана выполните `cargo bench -p splice --bench backends -- sql_plan`. Фильтр `sql_plan/execute/build` выбирает вызовы ядер, собранных через `build::<B>`. HTML-отчёт и интервалы оценок доступны в `target/criterion/report/index.html`. [Код стенда](splice/benches/backends.rs) · [Как читать отчёт Criterion](https://criterion-rs.github.io/book/user_guide/command_line_output.html)

Каждый замер `compile/build` включает специализацию и установку нового исполняемого ядра. Для Cranelift этот путь включает построение переносимого IR. Группа `stage_ir` измеряет построение IR отдельно, а `compile/from_ir` передаёт всем трём бэкендам одну готовую функцию. Освобождение JIT-кода происходит после замера сборки. Группы `execute/build` и `execute/from_ir` повторно вызывают готовые ядра на одних входах с заранее выделенным выходным буфером. До и после измерений стенд сверяет длину и содержимое результата с исходной Rust-функцией.

SQL-план на AMD Ryzen 7 9800X3D, Windows x64, профиль Cargo `bench`. В таблице указаны оценки Criterion по 30 выборкам на каждый случай. Каждое время исполнения относится к одному полному запросу.

| Бэкенд | Сборка `build::<B>`, мс | Вызов после `build`, мс | Сборка из IR, мс | Вызов после сборки из IR, мс |
| --- | ---: | ---: | ---: | ---: |
| Lancy (regalloc2 Ion, iced-x86) | 1,63 | 410,08 | 1,69 | 410,32 |
| Cranelift 0.132.3 (`speed`) | 5,25 | 381,70 | 5,03 | 384,16 |
| LLVM 19.1.5 (O3) | 93,31 | 118,09 | 92,06 | 125,93 |

Построение общего IR заняло 57 мкс. Для LLVM после `build::<Llvm>` 95%-й доверительный интервал оценки времени вызова составил 117,61–118,61 мс.

## Сборка

Нужны Rust 1.94+ и x86-64 Windows или Linux. Бэкенды Lancy и Cranelift собираются отдельно от LLVM. Для приложения с Cranelift достаточно зависимостей `splice` и `splice-cranelift`. Команда `cargo test -p splice-cranelift --release` собирает этот бэкенд и запускает его тесты. Пример с выбором бэкенда, общий стенд и полный комплект тестов используют все три компилятора.

* **Windows x64 MSVC.** Для `splice-llvm` выбран LLVM 19.1. Сборка находит LLVM в стандартном каталоге или в компонентах C++ Clang для Visual Studio. Для другой папки установки задайте `LLVM_SYS_191_PREFIX` до сборки. В ней должны быть `lib/LLVM-C.lib` и `bin/LLVM-C.dll`. Cargo подготавливает DLL для `cargo run`, тестов и бенчмарков. Для самостоятельного запуска готового приложения положите соответствующую DLL рядом с исполняемым файлом или добавьте её каталог в `PATH`.
* **Linux x86-64.** Выбран LLVM 21.1 с динамической линковкой. Нужны инструменты разработки, включая `llvm-config-21`, и разделяемая библиотека LLVM. Для отдельной установки укажите префикс через `LLVM_SYS_211_PREFIX`.

Версии Inkwell и LLVM указаны в [манифесте бэкенда](splice-llvm/Cargo.toml). Операции над LLVM-инструкциями, типами, значениями и базовыми блоками используют [Inkwell Builder](https://docs.rs/inkwell/0.11.0/inkwell/builder/struct.Builder.html).
