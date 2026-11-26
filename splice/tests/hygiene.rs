//! Generated code must not depend on names in the user's scope (MAC-04,
//! MAC-05, MAC-22, MAC-23): each module below shadows or hides something the
//! expansion used to rely on, and must still compile and run. Taking `print`
//! asserts that a kernel was lowered rather than kept as a Rust import.
//!
//! A user `mod foo` beside `#[kernel] fn foo` stays a name conflict by design:
//! the kernel's API is the module `foo` (`foo::build`, `foo::BuildArgs`).
use splice_lancy::Lancy;

/// Variants that shadow `Ok` and `Err` from the prelude.
mod status {
    #![allow(dead_code)]
    use splice::kernel;

    pub enum Status {
        Ok,
        Err,
    }
    use Status::*;

    pub fn unused() -> Status {
        if true { Ok } else { Err }
    }

    #[kernel]
    pub fn add(#[comptime] k: i32, x: i32) -> i32 {
        k + x
    }
}

#[test]
fn prelude_variants_shadowed_by_user_enum() {
    let _lowered = status::add::print;
    let code = status::add::build::<Lancy>(status::add::BuildArgs { k: 2 }).unwrap();
    assert_eq!(code.call(status::add::CallArgs { x: 5 }), Ok(7));
}

/// No prelude at all.
mod bare {
    #![no_implicit_prelude]
    use ::splice::kernel;

    #[kernel]
    pub fn scale(#[comptime] k: i32, xs: &[i32]) -> i32 {
        let mut sum = 0;
        for &x in xs {
            sum += k * x;
        }
        sum
    }
}

#[test]
fn module_without_implicit_prelude() {
    let _lowered = bare::scale::print;
    let code = bare::scale::build::<Lancy>(bare::scale::BuildArgs { k: 3 }).unwrap();
    assert_eq!(code.call(bare::scale::CallArgs { xs: &[1, 2] }), Ok(9));
}

/// `self` inside a `comptime!` of a kernel method.
mod receiver {
    use splice::kernel;

    #[derive(splice::KernelType)]
    pub struct Holder {
        pub n: i32,
    }

    #[kernel]
    impl Holder {
        fn get(#[comptime] &self, x: i32) -> i32 {
            comptime!(self.n + 1) + x
        }
    }

    #[kernel]
    pub fn read(#[comptime] holder: &Holder, x: i32) -> i32 {
        Holder::get(holder, x)
    }
}

#[test]
fn comptime_self_in_kernel_method() {
    use receiver::{Holder, read};
    let _lowered = read::print;
    let holder = Holder { n: 10 };
    let code = read::build::<Lancy>(read::BuildArgs { holder: &holder }).unwrap();
    assert_eq!(code.call(read::CallArgs { x: 5 }), Ok(read(&holder, 5)));
    assert_eq!(read(&holder, 5), 16);
}

/// Relative `use` paths in a kernel body.
mod relative {
    use splice::kernel;

    pub fn offset() -> i32 {
        100
    }

    pub mod inner {
        use splice::kernel;

        #[kernel]
        pub fn shifted(x: i32) -> i32 {
            use super::offset;
            x + offset()
        }
    }

    #[kernel]
    pub fn local(x: i32) -> i32 {
        use self::inner::shifted;
        shifted(x) + 1
    }
}

#[test]
fn relative_use_paths_in_kernel_bodies() {
    use relative::{inner::shifted, local};
    let _lowered = (shifted::print, local::print);
    let code = shifted::build::<Lancy>(shifted::BuildArgs {}).unwrap();
    assert_eq!(code.call(shifted::CallArgs { x: 1 }), Ok(101));
    let code = local::build::<Lancy>(local::BuildArgs {}).unwrap();
    assert_eq!(code.call(local::CallArgs { x: 1 }), Ok(102));
}

/// Ordinary attributes on parameters.
mod attributes {
    use splice::kernel;

    #[kernel]
    pub fn ignore(#[allow(unused_variables)] x: i32, #[comptime] k: i32) -> i32 {
        k
    }
}

#[test]
fn parameter_attributes_pass_through() {
    let code = attributes::ignore::build::<Lancy>(attributes::ignore::BuildArgs { k: 9 }).unwrap();
    assert_eq!(code.call(attributes::ignore::CallArgs { x: 1 }), Ok(9));
    assert_eq!(attributes::ignore(1, 9), 9);
}

/// User types named like the generated `BuildArgs`, `CallArgs` and
/// `Signature`, in scalar kernel and `#[kernel(rust)]` signatures.
mod generated_names {
    use splice::kernel;

    #[derive(splice::ComptimeKey, splice::KernelType)]
    pub struct Signature {
        pub scale: i32,
    }
    #[derive(splice::ComptimeKey, splice::KernelType)]
    pub struct BuildArgs(pub i32);
    #[derive(Clone, Copy)]
    pub struct CallArgs(pub i32);

    #[kernel]
    pub fn scaled(#[comptime] s: &Signature, #[comptime] b: BuildArgs, x: i32) -> i32 {
        s.scale * x + b.0
    }

    #[kernel(rust)]
    pub fn opaque(#[comptime] b: BuildArgs, c: CallArgs) -> i32 {
        let local: CallArgs = c;
        local.0 + b.0
    }
}

#[test]
fn user_types_named_like_generated_items() {
    use generated_names::{BuildArgs, CallArgs, Signature, opaque, scaled};
    let _lowered = scaled::print;
    let s = Signature { scale: 3 };
    let code = scaled::build::<Lancy>(scaled::BuildArgs {
        s: &s,
        b: BuildArgs(1),
    })
    .unwrap();
    assert_eq!(code.call(scaled::CallArgs { x: 5 }), Ok(16));
    let code = opaque::build::<Lancy>(opaque::BuildArgs { b: BuildArgs(2) }).unwrap();
    assert_eq!(code.call(opaque::CallArgs { c: CallArgs(4) }), Ok(6));
}
