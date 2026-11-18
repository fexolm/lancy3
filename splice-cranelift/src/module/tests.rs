use super::*;
use crate::Cranelift;
use cranelift_jit::JITBuilder;
use cranelift_module::default_libcall_names;
use splice::{
    Codegen, Executable,
    hir::{Binary, FunctionBuilder},
};

fn increment() -> Function {
    let mut f = FunctionBuilder::new(1);
    let one = f.constant(1);
    let result = f.binary(Binary::Add, f.arg(0), one);
    f.ret(result);
    f.finish().unwrap()
}

fn caller(child: Function, times: usize) -> Function {
    let child = Arc::new(child);
    let mut f = FunctionBuilder::new(1);
    let mut value = f.arg(0);
    for _ in 0..times {
        value = f.call_shared(child.clone(), &[value]);
    }
    f.ret(value);
    f.finish().unwrap()
}

fn emitted_graph(f: &Function, unique_bodies: usize) -> HashMap<FuncId, Vec<FuncId>> {
    let mut jit = JITModule::new(JITBuilder::new(default_libcall_names()).unwrap());
    let kernel = KernelModule::lower(&mut jit, f).unwrap();
    assert_eq!(kernel.bodies.len(), unique_bodies);
    let mut graph = HashMap::new();
    kernel
        .emit(&mut jit.make_context(), |id, ctx| {
            ctx.verify(jit.isa()).unwrap();
            let calls = local_callees(&ctx.func).collect();
            assert!(graph.insert(id, calls).is_none());
            Ok(())
        })
        .unwrap();
    graph
}

#[test]
fn nested_and_shared_calls_inline_across_function_boundaries() {
    let f = caller(caller(increment(), 2), 2);
    let graph = emitted_graph(&f, 3);
    assert_eq!(graph.len(), 1);
    assert!(graph.values().all(Vec::is_empty));
    assert_eq!(
        Cranelift::compile(&f).unwrap().call(&[i32::MAX], &mut []),
        Ok(i32::MIN + 3)
    );
}

#[test]
fn shared_call_dag_keeps_reachable_out_of_line_definitions() {
    let mut f = increment();
    for _ in 0..12 {
        f = caller(f, 2);
    }
    let graph = emitted_graph(&f, 13);
    assert!(graph.values().any(|calls| !calls.is_empty()));
    assert!(graph.values().flatten().all(|id| graph.contains_key(id)));
    assert_eq!(
        Cranelift::compile(&f).unwrap().call(&[-9], &mut []),
        Ok((1 << 12) - 9)
    );
}
