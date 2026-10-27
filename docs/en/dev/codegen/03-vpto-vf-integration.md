# VPTO VF Integration: Status and A5 Handoff

## Objective and current boundary

Compile a VF helper through VPTO while retaining the EmitC kernel body, PyPTO
Tensor wrapper, and runtime loading path. This branch implements the build
interface and validates the wrapper in CANN instruction simulation. It does
**not** yet establish execution through the persistent runtime on an A5 card.

Branch: `investigate/vpto-wrapper`. Base: `cecb2e3dc`.
The runtime gitlink is `d8f997fabd1b6988b69a94feef61ab669dda0eea`.
PTOAS source inspected during investigation: `235f381f3`; executable used for
validation: PTOAS 0.70. Toolchain: CANN 9.0.0 on Linux aarch64.

## Build design

```text
EmitC caller.pto -> caller.cpp -> real Tensor wrapper.cpp -> wrapper.device.o
VF helper.pto   -> PTOAS VPTO LLVM IR -> matching Bisheng -> helper.device.o

wrapper.device.o + helper.device.o -> ld.lld -> kernel.elf -> .text
                                                               |
                                                     runtime loader/dispatch
                                                     (hardware TODO)
```

A raw AI Core device object and a host fatobj are different artifacts. The
production linker consumes device objects. The standalone simulator harness
uses host fatobjs to launch the same wrapper/helper source through CANN; it does
not submit the extracted production `.text` to a persistent worker.

The helper exports `run_vf` with three UB FP32 pointers and an `i32` length. It
contains `pto.vecscope` and vector operations. The EmitC caller owns UB addresses,
GM transfers, and barriers. Its external C declaration must match the helper ABI.
`kernel_entry(__gm__ int64_t* args)` is the device entry for the runtime: it
unpacks runtime Tensor descriptors and scalar slots before calling that body.

See [the wrapper interface](02-kernel_wrapper.md#experimental-a5-vpto-helpers)
for `KERNELS[i]["vpto_sources"]` and `KernelCompiler.compile_incore()` usage.
Attach dependencies to a private generated configuration before publication;
there is no DSL annotation or automatic VF outlining in this branch.

## Implemented

- [x] Preserve declarations for separately compiled functions during wrapper
  preprocessing; internalize definitions and their matching forward declarations.
- [x] Compile one or more `.pto` helpers through VPTO LLVM IR and matching
  Bisheng; link them with the wrapper before extracting `.text`.
- [x] Restrict the interface to A5 AIV. Reject CPU `a5sim`, AIC, Bridge operations,
  and non-intrinsic external calls in helper LLVM IR.
- [x] Validate linked images: entry at `.text` start, no unresolved symbols,
  remaining relocations, or other nonempty allocated sections. This structural
  check alone does not prove arbitrary helper code position-independent.
- [x] Include helper contents, options, and compiler versions in the final binary
  cache identity; separate build-sharing keys and ignore legacy `.o` sidecars.
  Compiler versions are assumed immutable.
- [x] Package helper sources into artifacts and restore their paths on relocation.
- [x] Add compiler/cache/packaging regression tests and an opt-in CANN fixture.

Key files:

| Area | Repository path |
| ---- | --------------- |
| Wrapper preprocessing | `python/pypto/backend/_ptoas_preprocess.py` |
| Device compilation | `python/pypto/runtime/kernel_compiler.py` |
| Build contract and image checks | `python/pypto/runtime/_vpto.py` |
| Runtime build/cache wiring | `python/pypto/runtime/device_runner.py` |
| Source packaging | `python/pypto/runtime/_artifact_sources.py` |
| Reproducible fixture | `tests/ut/backend/vpto_vf_validation/` |

## Evidence recorded locally

- Native PyPTO extension and pinned runtime binding built in this worktree.
- 369 focused unit tests passed. All configured pre-commit checks passed,
  including repository-wide Pyright in the hook-managed dependency environment,
  Ruff, C++ lint/formatting, and documentation checks. The cache tests were
  rerun after the test-harness typing cleanup: seven passed.
- Three opt-in compiler/linker/simulator tests passed with PTOAS 0.70 and
  CANN 9.0.0. The linked production `.text` was 896 bytes, with entry at its start.
  Relinking at image base `0x10000` produced identical `.text` bytes.
- Simulator numerical cases: `n = 0, 1, 63, 64, 65, 127, 128, 1023, 1024`.
  All passed with real 128-byte runtime Tensor descriptors, a 32-element
  `start_offset`, exact FP32 addition, and unchanged prefix/tail sentinels.
- No hardware execution was completed: the investigation host could not
  initialize its NPU driver. Simulator success is not hardware success.

Logs and binaries live under ignored `build/vpto-investigation/`, including
`final-unit.log`, `precommit-final.log`, `commit-validation.log`, and
`commit-validation/simulator.log`. These
machine-local outputs are not committed; the test sources are the reproduction
recipe. Record new A5 results separately and update the checklist below.

## Reproduce the existing checks on the A5 server

First read `AGENTS.md`, `.claude/CLAUDE.md`, and the testing skill. Check out this
branch in an isolated worktree, initialize its pinned submodules, and build there.
Use a Python environment compatible with the server's CANN and torch_npu stack.
Build/install Simpler from this checkout's `runtime/`; do not bypass runtime pin
checks or copy another checkout's `.so` files. Verify imported extension paths
and revision stamps: an editable install can override `PYTHONPATH`.

Record `git rev-parse HEAD`, `git -C runtime rev-parse HEAD`, `ptoas --version`,
Bisheng/CANN versions, Python and torch_npu versions, and `npu-smi info`.
Use a free A5 device and record its ID. Keep compiler and simulator libraries
from one CANN installation. The reference simulator SoC is `Ascend950PR_9599`;
verify compatibility on a different installation rather than relabeling results.

Run the existing focused unit tests from the repository root:

```bash
source .claude/skills/testing/load-env.sh
export PYTHONPATH="$PWD/python:$PWD/runtime/python:$PWD/runtime:$PYTHONPATH"
python -m pytest \
  tests/ut/runtime/test_kernel_compiler.py \
  tests/ut/runtime/test_external_kernel_cache.py \
  tests/ut/runtime/test_prebuilt_artifact.py \
  tests/ut/runtime/test_kernel_binary_sharing.py \
  tests/ut/runtime/test_binary_cache_context.py \
  tests/ut/codegen/test_pto_codegen.py \
  tests/ut/backend/test_ptoas_sentinels_match.py \
  -n "$PYPTO_TEST_JOBS" -q
```

For the compiler/linker checks, activate the matching CANN environment first.
Ensure `ptoas` on `PATH` and `PTOAS_ROOT`, if set, select the same installation.
`ASCEND_HOME_PATH` must identify that CANN installation.

```bash
source .claude/skills/testing/load-env.sh
export PYPTO_SKIP_RUNTIME_PIN_CHECK=0
export PYPTO_RUN_VPTO_VF_VALIDATION=1
export PYPTO_VPTO_VF_OUTPUT="$PWD/build/vpto-investigation/a5-reproduction"
python -m pytest tests/ut/backend/vpto_vf_validation/test_validation.py \
  --confcutdir=tests/ut/backend/vpto_vf_validation -n 0 -q \
  -k 'not simulator'
```

Optional instruction simulation (serial: the fixture shares one output directory):

```bash
PYPTO_VPTO_VF_SIMULATOR=1 python -m pytest \
  tests/ut/backend/vpto_vf_validation/test_validation.py \
  --confcutdir=tests/ut/backend/vpto_vf_validation -n 0 -q
```

The simulator worker needs local IPC and rejects an application directory that
is group/world writable. The fixture adjusts permissions on its own outputs.
It currently links `runtime_camodel`; **do not run its `check_vf` executable as
if it were a hardware validation program**. Hardware validation below requires
additional implementation. The fixture's `a5sim` rejection is intentional:
`a5sim` means CPU simulation, not CANN instruction simulation.

## TODO and acceptance criteria

### P0: A5 execution, before frontend expansion

- [ ] **Environment and control case.** Verify a real A5 device is available and
  run one existing small EmitC-only case through the pinned persistent runtime.
  Select ONBOARD/A5 explicitly. Record device ID, versions, command, and result.
  Do not run the full system-test suite.
- [ ] **Standalone hardware isolation.** Add an explicit hardware mode/test for
  the wrapper/helper fixture using the installed CANN device runtime, without
  simulator injection or `runtime_camodel`. Support an explicit device ID and
  bounded execution. Pass the same nine lengths, offset, and sentinel checks.
- [ ] **Persistent runtime path.** Add a focused hardware-dependent test under
  `tests/st/`, using the repository's `@pl.jit` authoring conventions. Attach the
  manually prepared helper and external call through a private build path for
  this first test; avoid adding a public DSL feature solely for the harness.
  Compile using `vpto_sources`, load the resulting production `.text`, and
  dispatch through the actual Worker/CoreCallable path. Do not substitute the
  standalone ACL launch for this acceptance criterion.
- [ ] **Repeated calls and synchronization.** Pass all nine lengths, repeated
  launches with changed inputs, nonzero Tensor offsets, and untouched prefix/
  tail checks. Confirm UB ownership, VF completion, and MTE dependencies; inspect
  a trace if results depend on synchronization or fail intermittently.
- [ ] **Real cache behavior.** Show a warm binary-cache hit, then change helper
  computation and show a cache miss plus changed numerical output. Repeat after
  artifact relocation. Keep transient binaries/logs in ignored build outputs.
- [ ] **Record the verdict.** Add hardware commands, versions, device ID, pass/
  fail counts, and remaining failures to this document and its translation.
  Classify environment, compilation, linking, loading, dispatch, and numerical
  failures separately. Mark a box only when its evidence exists.

### P1: Product integration after P0

- [ ] Design how a VF function selects VPTO and how helper source/dependencies
  propagate from IR through codegen and artifact identity.
- [ ] Generate the external declaration, matching UB/scalar ABI, and helper
  module automatically; synchronize public APIs, stubs, and documentation.
- [ ] Test multiple helpers, nesting/call graphs, scalar ABI variations, dynamic
  lengths, and more dtypes supported by the selected VPTO release.

### P2: Capability expansion

- [ ] Design Bridge dependencies, auxiliary sections/relocations, and any needed
  loader changes before admitting them to the current `.text`-only path.
- [ ] Evaluate additional targets only with explicit compiler/runtime evidence;
  do not silently fall back to the CPU simulator.

The A5 handoff is complete when P0 has reproducible hardware evidence or a
precise external blocker. P1/P2 are separate work; simulator success alone does
not close P0.
