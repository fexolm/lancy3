/* llvm-sys expects external C entry points for the inline functions in
 * llvm-c/Target.h. The Visual Studio distribution omits the headers, so
 * build.rs derives an equivalent header from its LLVM-C import library. */
#ifdef SPLICE_LLVM_IMPORT_TARGETS
#include "import_targets.h"
#else
#include <llvm-c/Target.h>
#endif

#define ALL(name) void LLVM_InitializeAll##name(void) { LLVMInitializeAll##name(); }
#define NATIVE(name) int LLVM_InitializeNative##name(void) { return LLVMInitializeNative##name(); }

ALL(TargetInfos)
ALL(Targets)
ALL(TargetMCs)
ALL(AsmPrinters)
ALL(AsmParsers)
ALL(Disassemblers)
NATIVE(Target)
NATIVE(AsmParser)
NATIVE(AsmPrinter)
NATIVE(Disassembler)
