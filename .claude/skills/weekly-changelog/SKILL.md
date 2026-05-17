---
name: weekly-changelog
description: Generate a weekly changelog markdown file summarizing external API and feature changes from git commits in a date range. Extracts before/after Python examples per commit, groups by theme (DSL / distributed / runtime / IR deprecations), and attributes each change to its author. Use when the user asks for a weekly report, changelog, commit summary, or interface-change digest.
---

# Weekly Changelog Generator

## Overview

Produces a markdown report of **externally visible** PyPTO changes over a date range (typically one week). Each entry has a one-line summary, before/after Python example, classification (新增 / 替换 / 弃用), and the implementer's name so reviewers can find the owner. Internal refactors / chores / CI / internal fixes are excluded by default.

## Step 1: Collect Parameters

Ask the user with `AskUserQuestion`:

| Question | Header | Options |
| -------- | ------ | ------- |
| Date range? | Range | This week / Last week / Custom (YYYY-MM-DD..YYYY-MM-DD) |
| Output path? | Output | `./weekly-<start>-to-<end>.md` (Recommended) / `/tmp/...` / custom |
| Language? | Lang | 中文 / English |
| Scope? | Scope | External APIs only (Recommended) / All commits |

If the user already provided values in their request, skip the corresponding question.

## Step 2: List Commits in Range

```bash
git log --since="<start> 00:00" --until="<end> 23:59" \
        --pretty=format:"%h | %an | %s" --date=short
```

Capture `<hash> | <author> | <subject>` for every commit.

## Step 3: Classify Commits

For each commit, classify by subject **prefix and content**:

| Prefix / pattern | External? | Action |
| ---------------- | --------- | ------ |
| `feat(language)`, `feat(distributed)`, `feat(runtime)`, `feat(ir)` exposing DSL/IR op | Yes | Include |
| `feat:` with user-visible additions | Yes | Include |
| `fix(runtime)` / `fix(language)` changing public default or signature | Yes | Include |
| `feat`/`fix` strictly inside `src/` or `passes/` with no DSL / bindings / runtime API change | **No** | Skip |
| `refactor`, `chore`, `test`, `docs`, `ci` | **No** | Skip |

When uncertain, inspect `git show --stat <hash>` and look for changes under:

- `python/pypto/language/`
- `python/pypto/distributed/`
- `python/pypto/runtime/`
- `python/pypto/pypto_core/*.pyi`
- new bindings in `python/bindings/`

If the diff only touches `src/` or `include/` without altering any of the above, treat as internal.

## Step 4: Extract Before/After Per External Commit

For each external commit, in parallel batches of ~5, launch **Explore subagents** to gather:

1. One-sentence summary (Chinese or English per Step 1)
2. **Before** Python snippet (5–10 lines). For pure additions, write `无（新增）` / `None (new)` or show the prior workaround.
3. **After** Python snippet (5–10 lines), drawn from the PR description (`gh pr view <num>`) or new tests in `tests/ut/`.
4. Classification: 新增 (new) / 替换 (replace) / 弃用 (deprecate). Mark deprecations explicitly when a `DeprecationWarning` is emitted.

**Agent prompt template** (one agent per 3–5 commits):

```
研究下列 commits 的对外 Python 接口变化，对每个 commit 输出：
- 一句话摘要
- 改前用法（最小 Python 示例）
- 改后用法（最小 Python 示例）
- 性质（新增/替换/弃用）
工作目录: <project root>
命令: git show --stat <hash>; gh pr view <pr>; 查 python/pypto/<area>/ 和 tests/ut/
保持简洁（每个 commit <120 字）。
Commits: <list>
```

## Step 5: Assemble Markdown

Structure of the output file:

```markdown
# PyPTO 周报：<start> ~ <end>（对外功能与接口变更）

> 仅纳入用户可见的 ... 内部 refactor / chore / ci / 内部 fix 不列出。

## 概览
| Commit | PR | 实施人 | 主题 | 性质 |

## 负责人速查
| 负责人 | commit 数 | 涉及主题 |

## 一、Python DSL 与算子
### 1.1 <title> (#<pr>)
- **实施人**：<author>
- **性质**：新增 / 替换 / 弃用
- **摘要**：...
**改前**：```python ... ```
**改后**：```python ... ```

## 二、分布式 pld.* API
## 三、Runtime 配置
## 四、IR 算子 / 弃用提示
## 五、迁移建议（汇总弃用项）
| 旧用法 | 推荐写法 | 备注 |
```

Always include:

- The **实施人 / Author** line per entry (use `git log --pretty=format:"%an"`).
- A **负责人速查 / Owner index** table aggregating commits per author.
- A **迁移建议 / Migration guide** table for any deprecation or default-value change.

Theme buckets — pick the four headings that match your commits; common ones:

| Bucket | Typical commits |
| ------ | --------------- |
| Python DSL & operators | `feat(language)`, `feat(ir)` new ops |
| Distributed `pld.*` | `feat(distributed)` |
| Runtime / RunConfig | `feat(runtime)`, `fix(runtime)` user-visible |
| IR & deprecation | RFC-driven type changes, `pl.*` deprecations |

Omit empty buckets.

## Step 6: Save and Report

Write the file to the agreed output path with `Write`. Confirm in chat: line count, commit count covered, deprecation count.

## Conventions

- **Author names** come from `git log` (`%an`), not from `Co-Authored-By` lines.
- **Language**: produce the entire file in the chosen language; do not mix.
- **Examples must be runnable-shaped** — copy-paste from PR descriptions / tests, not invent.
- **Before for pure additions**: write `无（新增）` (zh) or `None (new)` (en). Do not fabricate a "before".
- **Mark deprecations explicitly** — the migration table at the end is the deliverable that protects users.

## Important Constraints

- **Never invent commits or PR numbers.** Only use what `git log` and `gh pr view` return.
- **Plan mode**: this workflow is read-only until Step 6. Safe to run during planning.
- **Scope discipline**: when scope is "external only", refusing to include a commit is correct behavior — record the skipped count in the final report.
- **Markdown file location policy**: This skill explicitly creates a markdown file outside `docs/`. Honor the user's chosen path; do not move it to `docs/`.

## Checklist

- [ ] Date range, output path, language, scope captured
- [ ] All commits in range listed with author
- [ ] Each commit classified external vs internal
- [ ] Before/after example extracted for every external commit
- [ ] Author + theme bucket assigned per entry
- [ ] Overview table + owner index + migration table all present
- [ ] File written to the requested path; summary reported back
