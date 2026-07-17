# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Parser / IR tests for ``pl.spmd_submit(..., predicate=(tensor[i] > 0))``.

``predicate=`` attaches a dispatch predicate to a ``ir.Submit``: the scheduler
evaluates ``tensor[indices] <op> target`` at the dispatch point and retires the
task inline (never dispatched to a core) when the comparison is false, while
still settling fanin/fanout. The predicate is stored on first-class
``Submit.predicate_operand`` / ``predicate_indices`` / ``predicate_op`` /
``predicate_target`` fields.

The comparison is matched syntactically, never evaluated — in this position
``rc[0, 0] > 0`` is a declarative spec, not a ``tensor.read`` plus a compare.
Only ``tensor[indices] OP int-literal`` is expressible, mirroring the runtime's
single-comparison predicate.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir import python_print
from pypto.language.parser.diagnostics.exceptions import ParserSyntaxError, ParserTypeError


def _flatten(stmt):
    if isinstance(stmt, ir.SeqStmts):
        out = []
        for s in stmt.stmts:
            out.extend(_flatten(s))
        return out
    if isinstance(stmt, ir.RuntimeScopeStmt):
        return _flatten(stmt.body)
    return [stmt]


def _main_submits(prog):
    fn = prog.get_function("main")
    assert fn is not None
    stmts = _flatten(fn.body)
    return [s.value for s in stmts if isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.Submit)]


_FP32_T = "pl.Tensor[[512, 128], pl.FP32]"
_INT32_T = "pl.Tensor[[512, 128], pl.INT32]"


def _program(predicate_src: str, deps_src: str = "[g_tid]"):
    """Build a two-kernel program whose expert submit carries ``predicate_src``.

    ``predicate_src`` is spliced verbatim as the ``predicate=`` argument text,
    ``deps_src`` as the ``deps=`` argument text. The expert submit's predicate
    reads ``rc``, which the gate submit (bound to ``g_tid``) produces — so the
    default ``deps_src`` satisfies the "producer must be in deps=" contract.
    """
    src = f"""
@pl.program
class Prog:
    @pl.function(type=pl.FunctionType.InCore)
    def expert(self, x: {_FP32_T}, out: pl.Out[{_FP32_T}]) -> {_FP32_T}:
        t = pl.load(x, [0, 0], [128, 128])
        out = pl.store(t, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def gate(self, g: pl.Out[{_INT32_T}]) -> {_INT32_T}:
        t = pl.load(g, [0, 0], [128, 128])
        g = pl.store(t, [0, 0], g)
        return g

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: {_FP32_T}, out: pl.Out[{_FP32_T}], rc: pl.Out[{_INT32_T}]) -> {_FP32_T}:
        with pl.manual_scope():
            rc, g_tid = pl.spmd_submit(self.gate, rc, core_num=1)
            out, _ = pl.spmd_submit(
                self.expert, x, out, core_num=1, deps={deps_src}, predicate={predicate_src}
            )
        return out
"""
    return pl.parse_program(src)


def test_predicate_populates_submit_fields():
    prog = _program("rc[0, 0] > 0")
    submits = _main_submits(prog)
    # gate submit has no predicate; expert submit carries the predicate.
    gate_sub, expert_sub = submits[0], submits[1]
    assert gate_sub.predicate_op == 0
    assert gate_sub.predicate_operand is None
    assert expert_sub.predicate_op == int(ir.DispatchPredicateOp.Gt.value)
    assert expert_sub.predicate_target == 0
    assert isinstance(expert_sub.predicate_operand, ir.Var)
    assert len(expert_sub.predicate_indices) == 2
    assert all(isinstance(i, ir.ConstInt) for i in expert_sub.predicate_indices)
    # Contract surface: the operand producer (gate) must be reachable via deps.
    assert len(expert_sub.deps) == 1


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("==", ir.DispatchPredicateOp.Eq),
        ("!=", ir.DispatchPredicateOp.Ne),
        (">", ir.DispatchPredicateOp.Gt),
        ("<", ir.DispatchPredicateOp.Lt),
        (">=", ir.DispatchPredicateOp.Ge),
        ("<=", ir.DispatchPredicateOp.Le),
    ],
)
def test_all_comparison_spellings(spelling, expected):
    prog = _program(f"rc[0, 0] {spelling} 3")
    expert_sub = _main_submits(prog)[1]
    assert expert_sub.predicate_op == int(expected.value)
    assert expert_sub.predicate_target == 3


def test_negative_target_literal():
    prog = _program("rc[0, 0] >= -5")
    assert _main_submits(prog)[1].predicate_target == -5


def test_explicitly_positive_target_literal():
    prog = _program("rc[0, 0] >= +5")
    assert _main_submits(prog)[1].predicate_target == 5


def test_unsupported_comparison_operator_rejected():
    # ``in`` is a Compare op the runtime predicate cannot express.
    with pytest.raises(ParserSyntaxError, match="operator is not supported"):
        _program("rc[0, 0] in x")


def test_chained_comparison_rejected():
    # The runtime evaluates exactly one comparison.
    with pytest.raises(ParserSyntaxError, match="single comparison"):
        _program("0 < rc[0, 0] < 8")


def test_reversed_operand_order_is_normalized():
    # ``0 < rc[e]`` means the same as ``rc[e] > 0``; the tensor must end up as
    # the operand and the operator is flipped to match.
    prog = _program("0 < rc[0, 0]")
    expert_sub = _main_submits(prog)[1]
    assert expert_sub.predicate_op == int(ir.DispatchPredicateOp.Gt.value)
    assert expert_sub.predicate_target == 0
    assert isinstance(expert_sub.predicate_operand, ir.Var)


def test_reversed_operand_order_flips_asymmetric_op():
    # ``5 >= rc[e]`` normalizes to ``rc[e] <= 5``.
    prog = _program("5 >= rc[0, 0]")
    expert_sub = _main_submits(prog)[1]
    assert expert_sub.predicate_op == int(ir.DispatchPredicateOp.Le.value)
    assert expert_sub.predicate_target == 5


def test_non_tensor_operand_rejected():
    # ``g_tid`` is a Scalar[TASK_ID], not a tensor.
    with pytest.raises(ParserTypeError, match="operand must be a tensor"):
        _program("g_tid[0] > 0")


def test_bare_tensor_operand_rejected():
    # The predicate must locate one *element*; a bare tensor has no subscript.
    with pytest.raises(ParserSyntaxError, match="compare a tensor element"):
        _program("rc > 0")


def test_tensor_index_rejected():
    # ``x`` is a tensor param — it would otherwise render into the runtime
    # predicate index array as ``ext_x``, emitting invalid C++.
    with pytest.raises(ParserTypeError, match="indices must be integer scalars"):
        _program("rc[x, 0] > 0")


def test_index_count_must_match_operand_rank():
    # ``rc`` is rank-2; a single index does not locate one element.
    with pytest.raises(ParserTypeError, match="rank-2 tensor, got 1"):
        _program("rc[0] > 0")


def test_predicate_operand_producer_must_be_in_deps():
    # ``rc`` is produced by the gate submit (``g_tid``). Dropping it from deps=
    # would let the scheduler evaluate the predicate against stale data.
    with pytest.raises(ParserSyntaxError, match="not in deps="):
        _program("rc[0, 0] > 0", deps_src="[]")


def test_predicate_operand_without_tracked_producer_allowed():
    # ``x`` is a function parameter, not a submit result — nothing to prove, so
    # the deps= contract check stays out of the way.
    prog = _program("x[0, 0] > 0", deps_src="[]")
    assert _main_submits(prog)[1].predicate_op == int(ir.DispatchPredicateOp.Gt.value)


def test_non_literal_target_rejected():
    with pytest.raises(ParserSyntaxError, match="must compare against an integer literal"):
        _program("rc[0, 0] > 1.5")


def test_non_literal_tensor_rhs_rejected():
    # ``t[i] > u[j]`` — the runtime compares against a constant, not another
    # tensor element.
    with pytest.raises(ParserSyntaxError, match="must compare against an integer literal"):
        _program("rc[0, 0] > x[0, 0]")


def test_predicate_must_be_a_comparison():
    with pytest.raises(ParserSyntaxError, match="must be a comparison expression"):
        _program("rc")


def test_no_predicate_leaves_fields_default():
    prog = _program("rc[0, 0] > 0")
    gate_sub = _main_submits(prog)[0]
    assert not gate_sub.predicate_op  # 0 == no predicate
    assert gate_sub.predicate_indices == []


def test_print_parse_round_trip():
    prog = _program("rc[0, 0] >= 2")
    printed = python_print(prog)
    # The predicate surfaces on the submit line as the comparison expression.
    assert "predicate=(" in printed
    assert ">= 2)" in printed
    reparsed = pl.parse_program(printed)
    ir.assert_structural_equal(reparsed, prog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
