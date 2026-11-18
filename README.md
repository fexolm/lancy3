# Splice

Splice — язык для CPU-ядер, которые специализируются по данным приложения после его запуска. Функция с атрибутом `#[kernel]` сохраняет обычный вызов Rust. Макрос также создаёт фабрику JIT-кода. Тип в `build::<B>()` определяет, какой компилятор обработает ядро.

```rust
use splice::{Executable, kernel};
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
    let llvm = total::build::<Llvm>(&plan)?;
    let listing: String = total::build::<Text>(&plan)?;
    print!("{listing}");
    assert_eq!(fast_build.call(&[10, 4], &mut [])?, 28);
    assert_eq!(llvm.call(&[10, 4], &mut [])?, 28);
    Ok(())
}
```

Вызов `build::<Lancy>` строит SSA HIR Lancy прямо при обходе кернела. Вызов `build::<Llvm>` таким же образом строит LLVM IR. Хостовый Rust выбирает вариант `match` по плану, вычисляет известные выражения и разворачивает `for` по `#[comptime]`-срезу. Параметры специализации могут содержать заимствования и пользовательские enum.

`build::<Text>` возвращает текст функции после специализации. В примере сигнатура содержит только `price` и `qty`, а скидка 3 появляется в теле перед умножением. В листинге сохраняются имена параметров, операции со срезами и вызовы. Если один помощник вызван с разными планами, его тела получают разные номера. Переходы между метками показывают оставшиеся runtime-ветвления и циклы. Листинг отражает семантику Splice до оптимизаций машинного бэкенда.

Сигнатура фабрики — `build::<T>(...) -> splice::Result<T::Output>`. У `Text` тип результата успешной сборки — `String`, у Lancy и LLVM — исполняемый кернел. Ошибки специализации и кодогенерации возвращаются через `Err(String)`. Для сравнения бэкендов на одном переносимом IR можно вызвать `let ir = total::ir(&plan)?;`, а затем `Lancy::compile(&ir)?`, `Llvm::compile(&ir)?` или `Text::compile(&ir)?`.

| Крейт | Ответственность |
| --- | --- |
| [splice](splice/src/lib.rs) | Общий фронтенд, построение и проверка переносимого IR, трейты `Codegen` и `Executable`. |
| [splice-macros](splice-macros/src/lib.rs) | Макрос `#[kernel]` разбирает Rust-синтаксис и генерирует обращения к фронтенду. |
| [splice-lancy](splice-lancy/src/lib.rs) | Строит SSA HIR Lancy. Доступны оптимизирующий `Lancy` и `LancyBaseline` для быстрой сборки. |
| [splice-llvm](splice-llvm/src/lib.rs) | Строит LLVM-модуль через типизированный Rust API Inkwell, оптимизирует его с `default<O3>` для процессора хоста и устанавливает машинный код через ORC LLJIT. `to_llvm_ir(&ir)` печатает модуль для отладки. |
| [splice-text](splice-text/src/lib.rs) | Печатает специализированную функцию и её вызываемые функции. Для использования достаточно зависимостей `splice` и `splice-text`. |
| [lancy](lancy/README.md) | CPU-компилятор с SSA HIR, LIR, распределением регистров и x64 JIT. С `default-features = false` его можно использовать без макросов. |

Скалярные аргументы `i32` и срезы `&[i32]` / `&mut [i32]` передаются двумя списками с сохранением порядка внутри каждого списка. Оба компилятора поддерживают вызовы других ядер, циклы, ранний возврат, проверяемые обращения к срезам и целочисленное деление. Обработка колонок выражается обычным циклом над срезами.

## Полный запрос и сравнение

В примере `sql_plan` один исходный кернел обходит `#[comptime]`-дерево из `Scan`, `Filter`, `Project` и бинарных `Join`. При каждом вызове готовая функция читает таблицы, строит хеш-индексы и возвращает все совпадения. [Код ядер](splice/examples/relational/kernels.rs) · [Дерево по пяти таблицам](splice/examples/relational/pricing.rs)

```sh
cargo run -p splice --release --example sql_plan -- lancy
cargo run -p splice --release --example sql_plan -- llvm
cargo run -p splice --release --example sql_plan -- text
cargo bench -p splice --bench backends
cargo test --workspace --release
```

Стенд сравнивает цикл по 65 536 строкам и запрос с четырьмя соединениями на 8 192 продажах. Он измеряет прямую сборку из кернела, подготовку переносимого IR, компиляцию из него и повторные вызовы машинного кода. Результат сверяется с исходной Rust-функцией до и после замеров. [Методика и код](splice/benches/backends.rs)

Один запуск на Windows x64 с LLVM 19.1.5. В таблице приведены медианы 15 серий для `build::<B>`, включая специализацию и установку кода. Время вызова усредняется внутри серии из 32 запусков.

| Нагрузка | Lancy сборка, мкс | LLVM сборка, мкс | Lancy вызов, мкс | LLVM вызов, мкс |
| --- | ---: | ---: | ---: | ---: |
| Проекция | 61,70 | 5 465,60 | 98,36 | 2,75 |
| SQL-план | 896,80 | 108 222,50 | 123,17 | 26,94 |

## Сборка

Нужны Rust 1.94+ и x86-64 Windows или Linux. Приложение с зависимостями `splice` и `splice-lancy` собирается отдельно от LLVM. Для сборки только этих библиотек запустите `cargo build -p splice -p splice-lancy`. Пример с выбором бэкенда, общий стенд и полный комплект тестов используют оба компилятора.

* **Windows x64 MSVC.** Для `splice-llvm` выбран LLVM 19.1. Сборка находит LLVM в стандартном каталоге или в компонентах C++ Clang для Visual Studio. Для другой папки установки задайте `LLVM_SYS_191_PREFIX` до сборки. В ней должны быть `lib/LLVM-C.lib` и `bin/LLVM-C.dll`. Cargo подготавливает DLL для `cargo run`, тестов и бенчмарков. Для самостоятельного запуска готового приложения положите соответствующую DLL рядом с исполняемым файлом или добавьте её каталог в `PATH`.
* **Linux x86-64.** Выбран LLVM 21.1 с динамической линковкой. Нужны инструменты разработки, включая `llvm-config-21`, и разделяемая библиотека LLVM. Для отдельной установки укажите префикс через `LLVM_SYS_211_PREFIX`.
* **Сценарии C++.** Для пакета `lancy-customers` нужен компилятор C++17.

Версии Inkwell и LLVM указаны в [манифесте бэкенда](splice-llvm/Cargo.toml). Операции над LLVM-инструкциями, типами, значениями и базовыми блоками используют [Inkwell Builder](https://docs.rs/inkwell/0.11.0/inkwell/builder/struct.Builder.html).
