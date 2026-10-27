# PTO Kernel Wrapper Generation

When compiling with the PTO backend via `ir.compile()`, a kernel wrapper is automatically generated for each InCore function to bridge the ptoas output to the orchestration calling convention.

## Pipeline

```text
InCore Function → PTOCodegen → .pto → ptoas → .cpp → kernel_wrapper → kernels/aiv/<name>.cpp
```

Each InCore function is compiled independently through ptoas. The final wrapper file combines:

1. **Preprocessed ptoas code** (with `__global__ AICORE` → `static`)
2. **`kernel_entry(__gm__ int64_t* args)`** wrapper that unpacks the args array and forwards to the ptoas function

## Output Structure

When the program contains an Orchestration function, the PTO backend generates the following output structure:

```text
output_dir/
├── passes_dump/                     # IR after each pass
├── ptoas_passes/                    # Optional ptoas IR after each pass
│   └── <kernel-or-group>/            # ptoas/MLIR-managed dump tree
├── ptoas/                           # Intermediates
│   ├── <func_name>.pto              # MLIR from PTOCodegen
│   └── <func_name>.cpp              # C++ from ptoas
├── kernels/aiv/
│   └── <func_name>.cpp              # Final wrapper
├── orchestration/
│   └── <orch_func_name>.cpp         # simpler runtime orchestration code
└── kernel_config.py                 # Runtime/orchestration/kernel config
```

`ptoas_passes/` is emitted only when `ir.compile(...,
dump_ptoas_passes=True)` or `RunConfig(dump_ptoas_passes=True)` is used.

The orchestration codegen generates identical orchestration C++ code using the simpler runtime API (`rt_submit_task`, `make_tensor_external`, etc.).

## Runtime configuration (`kernel_config.py`)

`kernel_config.py` exposes a `RUNTIME_CONFIG` dict that the dispatch path reads to launch the program. Stable keys:

| Key | When emitted | Notes |
| --- | ------------ | ----- |
| `runtime` | Always | `"tensormap_and_ringbuffer"` (default) or `"host_build_graph"` — the wire name of the `RuntimeKind` selected by `ir.compile(runtime=...)`, or by wrapping the call in `PassContext([], runtime=...)`. |
| `aicpu_thread_num` | Always (`0`) | `0` selects the runtime's architecture default (a2a3: 4; a5: 5); callers may explicitly override it. |

The runtime is carried by `PassContext` as an `ir::RuntimeKind`, not by a
codegen-only argument, so passes that must legalize IR for a specific runtime
switch on `PassContext::GetRuntime()` rather than comparing strings. It is an
enum rather than a name because the set is closed — one enumerator per
implementation under `runtime/src/<arch>/runtime/` — so a typo is a compile
error instead of a value that surfaces much later as an opaque CCEC error about
a nonexistent include directory.

The wire name crosses the ABI boundary in exactly one place each way:
`ir::RuntimeKindToName` when writing `kernel_config.py`, and
`ir::RuntimeKindFromName` when reading one back. Both are re-exported to Python
as `passes.runtime_kind_to_name` / `passes.runtime_kind_from_name`.

The runtime is also part of the `@pl.jit` cache key, so a `host_build_graph`
call cannot reuse an artifact compiled for `tensormap_and_ringbuffer`, whose
`kernel_config.py` names a runtime no matching worker would bind.

## Experimental A5 VPTO helpers

An A5 AIV kernel can keep its EmitC C++ body and Tensor wrapper while linking
separately compiled VPTO functions. On a **private, unpublished** generated tree,
add the helper dependencies to its `kernel_config.py` entry before compilation:

```python
KERNELS[0]["vpto_sources"] = [str(_ROOT_DIR / "helpers" / "helper.pto")]
```

Use the generated configuration's root variable, or resolve the path explicitly.
The C++ body must already declare and call the helper with matching external C
linkage and address spaces. Wrapper preprocessing preserves external function
declarations while internalizing embedded definitions. This is a build interface;
there is currently no DSL annotation or automatic VF outlining.

The build compiles the wrapper with CCEC, lowers each helper through PTOAS VPTO
LLVM IR, compiles that IR with the matching Bisheng, and links all device objects
before extracting `.text`. `KernelCompiler.compile_incore(...,
vpto_sources=[...])` exposes the same path directly. PTOAS host fatobjs are not
inputs to this linker. The runtime still builds, loads, and dispatches the kernel.

The initial contract is limited to `platform="a5"`, `core_type="aiv"`, and
self-contained `.pto` helpers without Bridge operations or external helper
calls. Non-intrinsic LLVM declarations are rejected. Linked images must start at
`kernel_entry` in `.text`, with no remaining relocations, undefined symbols, or
other allocated sections. This structural check does not prove arbitrary helper
code position-independent. `a5sim` is a CPU simulator
and cannot execute these device objects; CANN instruction simulation is a
separate validation path. The caller owns UB allocation, lifetime, transfers,
and synchronization; helper arguments must preserve their UB address spaces.

Helper contents, lowering/compiler options, and compiler versions participate
in the final binary cache identity. Published compiler versions are assumed
immutable. Legacy source-adjacent `.o` files are not reused for these builds.
Artifact packaging copies the helper sources and restores their paths after
relocation. Attach dependencies before artifact publication; editing an already
published artifact is unsupported.

The opt-in fixture `tests/ut/backend/vpto_vf_validation/test_validation.py`
checks the actual Tensor wrapper, device linking, and optional CANN simulator
numerics. Set `PYPTO_RUN_VPTO_VF_VALIDATION=1` and `ASCEND_HOME_PATH`; additionally
set `PYPTO_VPTO_VF_SIMULATOR=1` for simulator execution. It requires a built
PyPTO extension and the pinned Simpler binding. It does not validate the
persistent worker or execution on an NPU card.

## Argument Unpacking

The wrapper unpacks `int64_t* args` following the standard convention:

| Parameter Type | Unpacking Pattern |
| -------------- | ----------------- |
| `TensorType` | `ChipTensor*` → `buffer.addr` → typed pointer |
| `ScalarType` | `uint64_t` → union decode → typed value |

## SPMD Identity Parameters

`tile.get_block_idx()`, `tile.get_block_num()`, and `tile.get_subblock_idx()`
lower to synthetic `i32` parameters that PTOCodegen appends at the **end** of
the `func.func` signature using named SSAs (`%__pypto_spmd_block_idx`,
`%__pypto_spmd_block_num`, `%__pypto_spmd_subblock_idx`). The IR contract for
these ops is unchanged — the synthetic params live only in the generated
MLIR / C++ and never appear in `Function.params`. They are appended in the
canonical order `block_idx, block_num, subblock_idx`, each gated
independently on the ops the function actually uses.

```mlir
func.func @spmd_kernel(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
                       %__pypto_spmd_block_idx: i32,
                       %__pypto_spmd_block_num: i32)
                       attributes { ... } {
  %0 = arith.index_cast %__pypto_spmd_block_idx : i32 to index
  // ... use %0 as the block index ...
}
```

The kernel wrapper resolves the runtime values once from
`intrinsic.h::get_block_idx(args)` / `get_block_num(args)` and forwards them
as the trailing two call args:

```cpp
extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args) {
    // Read logical SPMD block identity from runtime dispatch payload
    int32_t __pypto_spmd_block_idx = get_block_idx(args);
    int32_t __pypto_spmd_block_num = get_block_num(args);

    // ... tensor / scalar / dyn-dim unpacking ...

    // Forward to ptoas-generated function (block args at the end)
    spmd_kernel(a, out, __pypto_spmd_block_idx, __pypto_spmd_block_num);
}
```

**subblock_idx (AIV lane).** `tile.get_subblock_idx()` uses the same
synthetic-param channel: the wrapper resolves it from
`intrinsic.h::get_sub_block_id(args)` (the runtime per-core lane id the
scheduler stores in `GlobalContext.sub_block_id`) and appends
`__pypto_spmd_subblock_idx` after any block-identity args. It deliberately
reads the runtime lane id rather than the ccec `get_subblockid()` register,
which returns a stale value under the `tensormap_and_ringbuffer` dispatch.

Split AIV FIFO endpoints use that runtime value even when the tensor program
does not call `tile.get_subblock_idx()`: after PTOAS lowers a split endpoint,
the wrapper backend forwards the lane as the third argument of PTO-ISA's
explicit `TPUSH(pipe, tile, subblock_id)` / `TPOP(...)` overload. If the
function has no synthetic subblock parameter already, the backend adds a
private trailing parameter to the generated function. The explicit overload
derives the byte offset from each call's actual tile type, so one automatic
pipe can safely carry differently sized sequential transfers. Like block
identity, the wrapper resolves the runtime lane unconditionally because
`GlobalContext.sub_block_id` is populated by the scheduler on every platform.
The endpoint argument is build-guarded: device builds pass the lane to the
explicit overload, while CPU simulation and in-core cost-model builds retain
PTO-ISA's normal two-argument endpoint because those implementations already
model their lane context and do not expose the device-only explicit-lane
overload.

**Detection scope.** Both layers detect SPMD usage on a per-function basis:

- `MemRefCollectorVisitor::UsesSpmdBlockOps` / `UsesSubblockOp` (C++,
  `src/codegen/pto/pto_codegen.cpp`) drive whether PTOCodegen appends the
  block / subblock params to a given function's signature.
- `_uses_spmd_block_ops` / `_uses_dynamic_subblock_id` (Python,
  `python/pypto/backend/pto_backend.py`) drive whether the wrapper appends the
  matching locals to the inner call site. Split `TPUSH` / `TPOP` endpoints are
  additionally detected by `_runtime_split_fifo_endpoint_counts`; they reuse
  that subblock argument or request the private one described above.

For non-SPMD sibling functions in an SPMD group (`group_uses_spmd=True` but
the function itself does not call `tile.get_block_*`), the wrapper still
declares the two block locals because the `__gm_pipe_buffer` sharding logic in
`_generate_arg_unpacking` consumes them — but it does **not** append them to
the inner call, matching the function's MLIR signature.

This replaces the earlier macro-shadow + `[[block_local]] static` /
`static thread_local` bridge plus `#pragma push_macro` / `#undef` /
`pop_macro` dance. Block and lane identity now flow through the call graph like
every other per-launch value (tensor pointers, scalar args, dynamic dims).

## Implementation

**Module**: `python/pypto/backend/pto_backend.py`

Key functions:

- `generate()` — entry point: produces all PTO backend files (kernels + orchestration + config)
- `_preprocess_ptoas_output()` — strips duplicate includes, makes functions static
- `_generate_arg_unpacking()` — generates C++ unpacking code from IR parameter types
- `_generate_kernel_wrapper()` — assembles the complete wrapper file

## See Also

- [PTO Codegen](00-pto_codegen.md): Generating the device-kernel MLIR
- [Orchestration Code Generation](01-orchestration_codegen.md): Generating the host-side launch code
