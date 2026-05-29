# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser + printer coverage for selective tensor dump (simpler#844, Layer 1).

Two surfaces feed the same IR attr ``attrs['dump_vars']`` (a ``vector<VarPtr>``),
so the dump target is tracked by Var identity (never by name):

* ``pl.dump(arg)`` — the Call-only wrapper around a kernel-call argument. The
  printer round-trips it by re-wrapping the marked args in ``pl.dump(...)``.
* ``pl.submit(..., dumps=[t, ...])`` — the submit-side kwarg (symmetric with
  ``deps=``); the ``pl.dump(...)`` wrapper is rejected inside a submit's args.
  The printer round-trips it by emitting a ``dumps=[...]`` kwarg.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserSyntaxError, ParserTypeError


def _kernel_calls(program: ir.Program, callee_name: str = "kernel") -> list[ir.Call]:
    """Collect every ``self.<callee_name>(...)`` Call in *program*."""
    found: list[ir.Call] = []

    class _Collector(ir.IRVisitor):
        def visit_call(self, op):
            if op.op.name == callee_name:
                found.append(op)
            super().visit_call(op)

    _Collector().visit_program(program)
    return found


def _submit_nodes(program: ir.Program) -> list[ir.Submit]:
    """Collect every ``ir.Submit`` bound on an AssignStmt RHS in *program*.

    There is no ``visit_submit`` Python hook, so we intercept the AssignStmt
    that binds the submit result (same shape as test_submit_passes.py's
    ``_find_submit_in_function``)."""
    found: list[ir.Submit] = []

    class _Collector(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Submit):
                found.append(op.value)
            super().visit_assign_stmt(op)

    _Collector().visit_program(program)
    return found


def test_dump_records_arg_vars_on_call() -> None:
    """Each ``pl.dump(arg)`` at a kernel call adds the arg Var to that Call's
    ``attrs['dump_vars']`` — by identity, so the entries are the call's own
    arg Vars."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
            r: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
            o: pl.Tensor[[16, 16], pl.FP32] = pl.store(r, [0, 0], output)
            return o

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
            d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            d = self.kernel(pl.dump(a), b, pl.dump(d))
            return d

    calls = _kernel_calls(P)
    assert len(calls) == 1
    assert "dump_vars" in calls[0].attrs
    names = {v.name_hint for v in calls[0].attrs["dump_vars"]}
    assert names == {"a", "d"}


def test_dump_absent_when_unused() -> None:
    """No ``pl.dump`` wrapper -> no ``dump_vars`` attr on the Call."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
            return o

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            d = self.kernel(a, d)
            return d

    calls = _kernel_calls(P)
    assert len(calls) == 1
    assert "dump_vars" not in calls[0].attrs


def test_dump_roundtrips_through_printer() -> None:
    """The printer re-wraps marked args in ``pl.dump(...)`` so the call
    round-trips."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            a_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            b_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(b, [0, 0], [16, 16])
            r: pl.Tile[[16, 16], pl.FP32] = pl.add(a_tile, b_tile)
            o: pl.Tensor[[16, 16], pl.FP32] = pl.store(r, [0, 0], output)
            return o

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
            d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            d = self.kernel(pl.dump(a), b, d)
            return d

    printed = P.as_python()
    assert "pl.dump(a)" in printed
    # The un-marked args are printed bare.
    assert "pl.dump(b)" not in printed


def test_dump_composes_with_no_dep() -> None:
    """``pl.dump(pl.no_dep(a))`` records both the dump_vars entry and the
    no_dep override index on the same arg, in any nesting order."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
            return o

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            d = self.kernel(pl.dump(pl.no_dep(a)), d)
            return d

    calls = _kernel_calls(P)
    assert len(calls) == 1
    names = {v.name_hint for v in calls[0].attrs["dump_vars"]}
    assert names == {"a"}
    assert list(calls[0].attrs["arg_direction_overrides"]) == [0]


def test_dump_rejects_non_var_argument() -> None:
    """``pl.dump(<non-Var>)`` is rejected — only a bound tensor Var can map to
    an ``Arg`` slot."""
    with pytest.raises(ParserSyntaxError, match="dump.*tensor variable"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
                return o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.kernel(pl.dump(5), d)  # type: ignore[arg-type]  # not a Var
                return d

        _ = P


def test_submit_dumps_records_arg_vars_on_submit() -> None:
    """Each ``dumps=`` entry adds the arg Var to that Submit's
    ``attrs['dump_vars']`` — by identity, so the entries are the submit's own
    arg Vars (the submit-side mirror of test_dump_records_arg_vars_on_call)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.submit(self.kernel, a, b, dumps=[a, b])
            return out

    submits = _submit_nodes(P)
    assert len(submits) == 1
    assert "dump_vars" in submits[0].attrs
    names = {v.name_hint for v in submits[0].attrs["dump_vars"]}
    assert names == {"a", "b"}


def test_submit_dumps_absent_when_unused() -> None:
    """No ``dumps=`` kwarg -> no ``dump_vars`` attr on the Submit."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.submit(self.kernel, a)
            return out

    submits = _submit_nodes(P)
    assert len(submits) == 1
    assert "dump_vars" not in submits[0].attrs


def test_submit_dumps_dedups_repeated_arg() -> None:
    """Listing the same arg twice in ``dumps=`` records it once (dedup by
    Var identity)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.submit(self.kernel, a, dumps=[a, a])
            return out

    submits = _submit_nodes(P)
    assert len(submits) == 1
    assert list(submits[0].attrs["dump_vars"]) == [submits[0].args[0]]


def test_submit_dumps_roundtrips_through_printer() -> None:
    """``pl.submit(..., dumps=[x])`` is the submit-side selective dump surface.
    The parser records the listed args on the Submit's ``dump_vars`` and the
    printer round-trips them as a ``dumps=[...]`` kwarg (not pl.dump wrappers)."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            return a

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.submit(self.kernel, a, dumps=[a])
            return out

    printed = P.as_python()
    assert "dumps=[a]" in printed, printed
    # Submit must NOT use the Call-only pl.dump(arg) wrapper surface.
    assert "pl.dump(" not in printed, printed


def test_submit_rejects_dump_wrapper() -> None:
    """The Call-only ``pl.dump(arg)`` wrapper is rejected inside a submit's
    argument list — submits use the ``dumps=[...]`` kwarg instead."""
    with pytest.raises(ParserSyntaxError, match="pl.dump.*not supported inside pl.submit"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
                return a

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
                with pl.manual_scope():
                    out, _ = pl.submit(self.kernel, pl.dump(a))
                return out

        _ = P


def test_submit_dumps_rejects_non_arg() -> None:
    """A ``dumps=`` entry that is not a positional argument of the submit is
    rejected (strict arg-membership validation)."""
    with pytest.raises(ParserTypeError, match="dumps=.*not an argument of this submit"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
                return a

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                z: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                with pl.manual_scope():
                    out, _ = pl.submit(self.kernel, a, dumps=[z])  # z is not an arg of this submit
                return out

        _ = P


def test_dumps_rejected_on_plain_call() -> None:
    """``dumps=`` is only valid on ``pl.submit(...)``; a plain kernel call must
    use the ``pl.dump(arg)`` wrapper instead."""
    with pytest.raises(ParserTypeError, match="does not accept keyword argument 'dumps'"):

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.AIV)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
                o: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], output)
                return o

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                d = self.kernel(a, d, dumps=[a])  # type: ignore[call-arg]  # dumps= invalid on plain call
                return d

        _ = P


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
