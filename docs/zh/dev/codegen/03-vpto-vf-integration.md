# VPTO VF 接入：进展与 A5 交接

## 目标和当前边界

让 VF helper 使用 VPTO 编译，同时保留 EmitC kernel 主体、PyPTO Tensor wrapper
和 runtime 加载路径。本分支已实现构建接口，并通过 CANN 指令模拟器验证真实
wrapper；**尚未证明它能通过常驻 runtime 在 A5 真卡上执行**。

分支：`investigate/vpto-wrapper`。基线：`cecb2e3dc`。
runtime gitlink：`d8f997fabd1b6988b69a94feef61ab669dda0eea`。
调查时阅读的 PTOAS 源码版本：`235f381f3`；验证使用 PTOAS 0.70 可执行文件，
工具链为 Linux aarch64 上的 CANN 9.0.0。

## 构建方案

```text
EmitC caller.pto -> caller.cpp -> real Tensor wrapper.cpp -> wrapper.device.o
VF helper.pto   -> PTOAS VPTO LLVM IR -> matching Bisheng -> helper.device.o

wrapper.device.o + helper.device.o -> ld.lld -> kernel.elf -> .text
                                                               |
                                                     runtime loader/dispatch
                                                     (hardware TODO)
```

AI Core 原始设备对象（device object）和主机封装对象（host fatobj）是不同产物。
生产链接器接收设备对象。独立模拟器测试使用 host fatobj，通过 CANN 启动相同
wrapper/helper 源码；它没有将生产构建提取的 `.text` 提交给常驻 worker。

helper 导出 `run_vf`，参数是三个 UB FP32 指针和一个 `i32` 长度，内部包含
`pto.vecscope` 和向量操作。EmitC 调用方负责 UB 地址、GM 搬运和 barrier，
外部 C 声明必须与 helper ABI 一致。`kernel_entry(__gm__ int64_t* args)` 是
runtime 使用的设备入口，先解包 runtime Tensor 描述符及标量槽，再调用主体。

`KERNELS[i]["vpto_sources"]` 和 `KernelCompiler.compile_incore()` 的用法见
[wrapper 接口说明](02-kernel_wrapper.md)。依赖必须在私有
生成配置发布前添加；本分支没有 DSL 注解或自动 VF 拆分（outlining）。

## 已完成

- [x] wrapper 预处理保留独立编译函数的声明；嵌入的定义及对应前置声明使用内部链接。
- [x] 一个或多个 `.pto` helper 经 VPTO LLVM IR 和匹配的 Bisheng 编译，
  与 wrapper 链接后再提取 `.text`。
- [x] 仅允许 A5 AIV，拒绝 CPU `a5sim`、AIC、Bridge 操作及 helper LLVM IR
  中非 intrinsic 的外部调用。
- [x] 检查链接映像：入口在 `.text` 起点，没有未解析符号、残留重定位或其他
  非空可加载 section。仅靠结构检查不能证明任意 helper 都具有位置无关性。
- [x] helper 内容、选项、编译器版本进入最终二进制缓存标识；区分编译共享键，
  不复用旧的源码旁 `.o` 文件。假设已发布的编译器版本不可变。
- [x] artifact 打包 helper 源码，迁移后恢复路径。
- [x] 编译、缓存、打包回归测试，以及显式启用的 CANN 验证用例。

关键文件：

| 部分 | 仓库路径 |
| ---- | -------- |
| wrapper 预处理 | `python/pypto/backend/_ptoas_preprocess.py` |
| 设备编译 | `python/pypto/runtime/kernel_compiler.py` |
| 构建约束和映像检查 | `python/pypto/runtime/_vpto.py` |
| runtime 构建和缓存接线 | `python/pypto/runtime/device_runner.py` |
| 源码打包 | `python/pypto/runtime/_artifact_sources.py` |
| 可复现测试 | `tests/ut/backend/vpto_vf_validation/` |

## 已有验证证据

- 在当前 worktree 构建了 PyPTO 原生扩展及匹配固定版本的 runtime binding。
- 369 项针对性单测通过，全部配置的 pre-commit 检查通过，包括 hook 管理的
  依赖环境中的全仓 Pyright、Ruff、C++ lint/格式和文档检查。修正测试辅助代码
  的类型问题后，重新执行的七项缓存测试也全部通过。
- PTOAS 0.70、CANN 9.0.0 下，三个编译、链接及模拟器测试通过。生产构建的
  `.text` 为 896 字节，入口位于其起点。用 `0x10000` 映像基址重新链接后，
  `.text` 字节保持一致。
- 模拟器覆盖 `n = 0, 1, 63, 64, 65, 127, 128, 1023, 1024`，全部通过。
  使用真实 128 字节 runtime Tensor 描述符，`start_offset` 为 32 个元素，
  验证 FP32 加法精确结果以及前缀、尾部哨兵值不变。
- 尚未完成真卡执行：调查机器无法初始化 NPU 驱动，模拟器成功不能等同于真卡成功。

本地日志和二进制位于已忽略的 `build/vpto-investigation/`，包括
`final-unit.log`、`precommit-final.log`、`commit-validation.log` 和
`commit-validation/simulator.log`。
这些机器相关输出不提交，测试源码是复现入口。A5 上的新结果应单独记录并更新下方清单。

## 在 A5 服务器上复现已有检查

先阅读 `AGENTS.md`、`.claude/CLAUDE.md` 和 testing skill。在独立 worktree 检出
本分支，初始化固定版本的 submodule，并在其中构建。Python 环境应与服务器的
CANN、torch_npu 匹配。使用本 checkout 的 `runtime/` 构建、安装 Simpler；
不要绕过 runtime pin 检查，不要复制其他 checkout 的 `.so`。检查实际导入的
扩展路径和版本标记：editable install 可能覆盖 `PYTHONPATH`。

记录 `git rev-parse HEAD`、`git -C runtime rev-parse HEAD`、`ptoas --version`、
Bisheng/CANN 版本、Python 和 torch_npu 版本以及 `npu-smi info`。
选择空闲 A5 卡并记录 device ID。编译器和模拟器库必须来自同一套 CANN。
参考模拟器 SoC 是 `Ascend950PR_9599`；其他安装需先确认兼容性，不能仅修改标签。

在仓库根目录运行现有针对性单测：

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

编译、链接检查需要先激活匹配的 CANN 环境。确保 `PATH` 中的 `ptoas` 与
`PTOAS_ROOT`（如果设置）选择同一套安装；`ASCEND_HOME_PATH` 指向该 CANN 安装。

```bash
source .claude/skills/testing/load-env.sh
export PYPTO_SKIP_RUNTIME_PIN_CHECK=0
export PYPTO_RUN_VPTO_VF_VALIDATION=1
export PYPTO_VPTO_VF_OUTPUT="$PWD/build/vpto-investigation/a5-reproduction"
python -m pytest tests/ut/backend/vpto_vf_validation/test_validation.py \
  --confcutdir=tests/ut/backend/vpto_vf_validation -n 0 -q \
  -k 'not simulator'
```

可选的指令模拟器验证必须串行运行，因为 fixture 使用共享输出目录：

```bash
PYPTO_VPTO_VF_SIMULATOR=1 python -m pytest \
  tests/ut/backend/vpto_vf_validation/test_validation.py \
  --confcutdir=tests/ut/backend/vpto_vf_validation -n 0 -q
```

模拟器 worker 需要本地 IPC，且拒绝组或其他用户可写的程序目录。fixture 只调整
自己的输出权限。它当前链接 `runtime_camodel`，**不能把生成的 `check_vf`
直接当成真卡测试程序运行**。下方的真卡验证仍需要补充实现。
fixture 拒绝 `a5sim` 是有意的：它表示 CPU 模拟，不是 CANN 指令模拟。

## TODO 与验收标准

### P0：先完成 A5 执行，再扩展前端

- [ ] **环境与对照用例。** 确认真实 A5 卡可用，通过固定版本的常驻 runtime
  跑通一个现有的小型纯 EmitC 用例，显式选择 ONBOARD/A5，记录卡号、版本、
  命令和结果。不要运行全量系统测试。
- [ ] **独立真卡定位。** 为 wrapper/helper fixture 增加显式真卡模式或测试，
  使用本机 CANN 设备 runtime，不注入模拟器，不链接 `runtime_camodel`。
  支持显式 device ID 和有界超时，通过相同九组长度、offset 和哨兵检查。
- [ ] **常驻 runtime 路径。** 在 `tests/st/` 增加针对性硬件测试，遵循仓库
  `@pl.jit` 编写约定。首个测试可以在私有构建路径手动附加 helper 和外部调用，
  不要仅为测试引入公共 DSL 特性。使用 `vpto_sources` 编译，加载生产 `.text`，
  经真实 Worker/CoreCallable 路径调度。独立 ACL launch 不能替代这项验收。
- [ ] **重复调用与同步。** 通过九组长度、多次启动且切换输入、非零 Tensor offset、
  前缀和尾部不变检查。确认 UB 归属、VF 完成和 MTE 依赖；如结果依赖同步或
  存在偶发失败，采集 trace 定位。
- [ ] **真实缓存行为。** 展示一次二进制缓存命中，再修改 helper 计算，证明缓存
  失效并得到变化后的数值输出。迁移 artifact 后重复验证，临时二进制和日志放入
  已忽略的构建目录。
- [ ] **回填结论。** 在本文及英文版记录真卡命令、版本、卡号、通过/失败数量和
  遗留失败。区分环境、编译、链接、加载、调度、数值错误，有证据后才勾选完成。

### P1：P0 之后再做产品接入

- [ ] 设计 VF 函数选择 VPTO 的方式，以及 helper 源码和依赖如何从 IR 传递到
  codegen 与 artifact 标识。
- [ ] 自动生成外部声明、匹配的 UB/标量 ABI 和 helper 模块；同步公共 API、
  stub 与文档。
- [ ] 测试多个 helper、嵌套和调用图、标量 ABI 变化、动态长度，以及所选 VPTO
  版本支持的更多数据类型。

### P2：能力扩展

- [ ] 接纳 Bridge 前，设计其依赖、额外 section/重定位，以及必要的 loader 改动。
- [ ] 有明确编译器和 runtime 证据后再扩展目标平台，不能静默回退到 CPU 模拟器。

A5 交接的完成条件是 P0 获得可复现的真卡证据，或者明确的外部阻塞原因。
P1/P2 属于后续工作，模拟器通过不能关闭 P0。
