# Clippy (pedantic), частичный прогон без splice-llvm/cranelift; llvm-sys не собрался (нет LLVM в окружении ревью)

```
    348 warning: long literal lacking separators
    159 warning: this method could have a X attribute
     85 warning: casting X to X may truncate the value on targets with 64-bit wide pointers
     61 warning: docs for function returning X missing X section
     59 warning: casting X to X may truncate the value
     51 warning: casting X to X may lose the sign of the value
     38 warning: casting X to X may wrap around the value on targets with 32-bit wide pointers
     35 warning: this argument is passed by value, but not consumed in the function body
     26 warning: missing X attribute on a method returning X
     24 warning: 5 bindings with single-character names in scope
     21 warning: docs for function which may panic missing X section
     21 warning: casting X to X may wrap around the value
     20 warning: these match arms have identical bodies
     18 warning: usage of wildcard import for enum variants
     17 warning: adding items after statements is confusing, since items exist from the start of the scope
     14 warning: item in documentation is missing backticks
     14 warning: casts from X to X can be expressed infallibly using X
     13 warning: consider adding a X to the last statement for consistent formatting
     12 warning: usage of wildcard import
     11 warning: 6 bindings with single-character names in scope
      8 warning: this function could have a X attribute
      8 warning: calling X is more clear than this expression
      8 warning: 9 bindings with single-character names in scope
      6 warning: you seem to be trying to use X for destructuring a single pattern. Consider using X
      6 warning: casting X to X may truncate the value on targets with 32-bit wide pointers
      5 warning: wildcard matches only a single variant and will also match any future added variants
      5 warning: 8 bindings with single-character names in scope
      5 warning: 7 bindings with single-character names in scope
      4 warning: casting X to X may wrap around the value on targets with 64-bit wide pointers
      3 warning: redundant closure
      3 warning: casting X to X may cause a loss of precision (X is 32 bits wide, but X's mantissa is only 23 bits wide)
      3 warning: casting X to X may cause a loss of precision (X can be up to 64 bits wide depending on the target architecture, but X's mantissa is only 52 bits wide)
      3 warning: binding's name is too similar to existing binding
      3 warning: 10 bindings with single-character names in scope
      2 warning: unused X argument
      2 warning: this function has too many lines (199/100)
      2 warning: this function has too many lines (196/100)
      2 warning: this function has too many lines (152/100)
      2 warning: this function has too many lines (137/100)
      2 warning: this function has too many lines (118/100)
```

## Подозрительные касты (truncation/sign), места:
lancy/src/hir/integer.rs:10:5
lancy/src/hir/integer.rs:15:19
lancy/src/hir/integer.rs:15:9
lancy/src/hir/integer.rs:35:16
lancy/src/hir/integer.rs:39:17
lancy/src/hir/integer.rs:73:36
lancy/src/hir/integer.rs:74:34
lancy/src/hir/integer.rs:80:28
lancy/src/hir/integer.rs:84:38
lancy/src/hir/integer.rs:85:38
lancy/src/hir/ssa/function.rs:279:24
lancy/src/hir/transforms/passes/bounds.rs:31:30
lancy/src/hir/transforms/passes/bounds.rs:38:37
lancy/src/hir/transforms/passes/facts.rs:305:58
lancy/src/hir/transforms/passes/facts.rs:326:38
lancy/src/hir/transforms/passes/facts.rs:326:51
lancy/src/hir/transforms/passes/memory.rs:146:40
lancy/src/hir/transforms/passes/memory.rs:159:40
lancy/src/hir/transforms/passes/memory.rs:161:26
lancy/src/hir/transforms/passes/sccp.rs:134:55
lancy/src/hir/transforms/passes/sccp.rs:44:19
lancy/src/jit/kernel.rs:129:24
lancy/src/jit/kernel.rs:132:79
lancy/src/lir/aarch64/instruction.rs:1106:44
lancy/src/lir/aarch64/instruction.rs:1233:29
lancy/src/lir/aarch64/instruction.rs:1248:23
lancy/src/lir/aarch64/instruction.rs:1274:25
lancy/src/lir/aarch64/instruction.rs:1399:27
lancy/src/lir/aarch64/instruction.rs:1416:31
lancy/src/lir/aarch64/instruction.rs:1427:23
lancy/src/lir/aarch64/instruction.rs:1591:23
lancy/src/lir/aarch64/instruction.rs:1610:29
lancy/src/lir/aarch64/instruction.rs:1728:38
lancy/src/lir/aarch64/instruction.rs:1728:57
lancy/src/lir/aarch64/instruction.rs:1742:66
lancy/src/lir/aarch64/instruction.rs:1859:27
lancy/src/lir/aarch64/instruction.rs:1901:24
lancy/src/lir/aarch64/instruction.rs:415:13
lancy/src/lir/schedule.rs:325:9
lancy/src/lir/x64/emit.rs:222:34
lancy/src/lir/x64/emit.rs:226:34
lancy/src/lir/x64/emit.rs:266:78
lancy/src/lir/x64/emit.rs:385:52
lancy/src/lir/x64/emit.rs:393:44
lancy/src/lir/x64/emit.rs:515:33
lancy/src/lir/x64/emit.rs:53:5
lancy/src/lir/x64/emit.rs:566:32
lancy/src/lir/x64/emit.rs:570:33
lancy/src/lir/x64/emit.rs:573:34
lancy/src/lir/x64/emit.rs:574:37
lancy/src/lir/x64/emit.rs:709:36
lancy/src/lir/x64/emit.rs:733:40
lancy/src/lir/x64/emit.rs:930:53
lancy/src/lir/x64/emit.rs:944:55
lancy/src/lir/x64/inst/lanes.rs:111:43
lancy/src/lir/x64/inst/lanes.rs:128:26
lancy/src/lir/x64/inst/lanes.rs:173:13
lancy/src/lir/x64/inst/lanes.rs:175:13
lancy/src/lir/x64/inst/lanes.rs:178:17
lancy/src/lir/x64/inst/lanes.rs:215:24
lancy/src/lir/x64/inst/lanes.rs:219:30
lancy/src/lir/x64/inst/lanes.rs:226:26
lancy/src/lir/x64/inst/lanes.rs:260:28
lancy/src/lir/x64/inst/lanes.rs:261:28
lancy/src/lir/x64/inst/lanes.rs:263:36
lancy/src/lir/x64/inst/lanes.rs:292:23
lancy/src/lir/x64/inst/lanes.rs:296:25
lancy/src/lir/x64/inst/lanes.rs:300:43
lancy/src/lir/x64/inst/lanes.rs:301:34
lancy/src/lir/x64/inst/lanes/convert.rs:131:16
lancy/src/lir/x64/inst/lanes/convert.rs:185:20
lancy/src/lir/x64/inst/lanes/convert.rs:53:15
lancy/src/lir/x64/inst/lanes/convert.rs:54:15
lancy/src/lir/x64/inst/lanes/convert.rs:98:16
lancy/src/lir/x64/inst/lanes/float.rs:115:24
lancy/src/lir/x64/inst/lanes/float.rs:11:15
lancy/src/lir/x64/inst/lanes/float.rs:23:16
lancy/src/lir/x64/inst/lanes/float.rs:361:34
lancy/src/lir/x64/inst/lanes/float.rs:79:24
lancy/src/lir/x64/inst/lanes/integer.rs:104:36
lancy/src/lir/x64/inst/lanes/integer.rs:69:16
lancy/src/lir/x64/inst/simd.rs:1008:56
lancy/src/lir/x64/inst/simd.rs:1046:20
lancy/src/lir/x64/inst/simd.rs:1061:27
lancy/src/lir/x64/inst/simd.rs:1063:44
lancy/src/lir/x64/inst/simd.rs:1073:28
lancy/src/lir/x64/inst/simd.rs:461:34
lancy/src/lir/x64/inst/simd.rs:461:62
lancy/src/lir/x64/inst/simd.rs:479:29
lancy/src/lir/x64/inst/simd.rs:479:52
lancy/src/lir/x64/inst/simd.rs:936:63
lancy/src/lir/x64/inst/simd/opmask.rs:17:25
lancy/src/lir/x64/inst/simd/packed/compact.rs:18:35
lancy/src/lir/x64/inst/simd/packed/compact.rs:42:51
lancy/src/lir/x64/inst/simd/packed/compact.rs:42:66
lancy/src/lir/x64/inst/simd/packed/compact.rs:70:27
lancy/src/lir/x64/inst/simd/packed/float.rs:40:27
lancy/src/lir/x64/inst/simd/packed/rearrange.rs:200:17
lancy/src/lir/x64/inst/simd/packed/rearrange.rs:215:18
lancy/src/lir/x64/inst/simd/packed/rearrange.rs:483:40
lancy/src/lir/x64/inst/simd/packed/rearrange.rs:65:43
lancy/src/lir/x64/isel/masks.rs:37:77
lancy/src/lir/x64/isel/rules/binary.rs:69:63
lancy/src/lir/x64/isel/rules/binary.rs:74:37
lancy/src/lir/x64/isel/rules/division.rs:57:37
lancy/src/lir/x64/isel/rules/division.rs:72:25
lancy/src/lir/x64/isel/rules/division.rs:73:20
lancy/src/lir/x64/isel/rules/experimental.rs:18:25
lancy/src/lir/x64/isel/rules/mask.rs:125:21
lancy/src/lir/x64/isel/rules/mask.rs:127:72
lancy/src/lir/x64/isel/rules/mask.rs:52:52
lancy/src/lir/x64/isel/rules/mod.rs:136:57
lancy/src/lir/x64/isel/rules/mod.rs:164:34
lancy/src/lir/x64/isel/rules/mod.rs:165:62
lancy/src/lir/x64/isel/rules/mod.rs:166:53
lancy/src/lir/x64/isel/rules/mod.rs:83:39
lancy/src/lir/x64/isel/rules/packed_rearrange.rs:112:86
lancy/src/lir/x64/isel/rules/packed_rearrange.rs:136:80
lancy/src/lir/x64/isel/rules/packed_rearrange.rs:28:21
lancy/src/lir/x64/isel/rules/packed_rearrange.rs:84:89
lancy/src/lir/x64/isel/rules/packed_rearrange.rs:95:32
lancy/src/support/float16.rs:63:37
lancy/src/support/float16.rs:73:20
lancy/src/support/float16.rs:74:24
lancy/src/support/slotmap/key.rs:21:22
splice-core/src/debug.rs:78:39
splice-core/src/frontend/scalar.rs:339:31
splice-core/src/hir/builder.rs:58:13
splice-core/src/hir/mod.rs:101:47
splice-core/src/hir/mod.rs:102:47
splice-core/src/hir/mod.rs:96:41
splice-core/src/hir/mod.rs:97:41
splice-core/src/runtime_context.rs:113:21
splice-core/src/runtime_context.rs:122:28
splice-core/src/runtime_context.rs:171:36
splice-core/src/runtime_context.rs:317:24
splice-core/src/runtime_context.rs:318:20
splice-core/src/runtime_context.rs:342:63
splice-core/src/runtime_context.rs:356:78
splice-core/src/runtime_context.rs:427:29
splice-core/src/runtime_context.rs:485:33
splice-core/src/runtime_context.rs:79:25
splice-core/src/simd.rs:100:17
splice-core/src/simd.rs:123:17
splice-core/src/simd.rs:130:17
splice-core/src/simd.rs:319:9
splice-core/src/simd.rs:403:23
splice-core/src/simd.rs:445:17
splice-core/src/simd.rs:449:21
splice-core/src/simd.rs:572:23
splice-core/src/simd.rs:572:31
splice-jit/src/elf.rs:136:34
splice-jit/src/elf.rs:39:43
splice-jit/src/elf.rs:60:31
splice-jit/src/elf.rs:85:18
splice-jit/src/lib.rs:51:68
splice-lancy/src/debug_info.rs:54:21
splice-lancy/src/debug_info.rs:61:13
splice-lancy/src/lib.rs:119:43
splice-lancy/src/lib.rs:127:22
splice-lancy/src/lower/mod.rs:386:39
splice-lancy/src/lower/mod.rs:400:58
splice-lancy/src/lower/mod.rs:400:77
splice-lancy/src/lower/mod.rs:407:60
splice-lancy/src/lower/mod.rs:415:40
splice-lancy/src/lower/mod.rs:419:27
splice-lancy/src/lower/mod.rs:422:49
splice-lancy/src/lower/mod.rs:528:13
splice-lancy/src/lower/mod.rs:87:30
splice-lancy/src/lower/mod.rs:98:37
splice-lancy/src/lower/portable.rs:60:53
splice-lancy/src/lower/simd.rs:174:47
splice-lancy/src/lower/simd.rs:197:47
splice-lancy/src/lower/simd.rs:206:47
splice-macros/src/comptime_key.rs:50:61
splice-macros/src/scalar/lower/context.rs:129:25
splice-macros/src/scalar/lower/context.rs:167:31
splice-macros/src/scalar/lower/context.rs:310:17
splice-macros/src/scalar/lower/context.rs:312:31
splice-macros/src/scalar/lower/context.rs:463:9
splice-macros/src/scalar/lower/context.rs:90:30
splice-macros/src/scalar/lower/context.rs:93:30
splice-macros/src/scalar/lower/mod.rs:441:29
splice-macros/src/scalar/lower/mod.rs:441:51
splice-macros/src/scalar/lower/owned.rs:75:26
splice-macros/src/scalar/lower/owned.rs:76:27
splice-macros/src/scalar/mod.rs:227:29
splice-macros/src/scalar/mod.rs:249:29
splice-macros/src/scalar/mod.rs:513:20
splice-target/src/abi.rs:45:36
splice-target/src/lib.rs:118:20
splice-target/src/lib.rs:36:8
