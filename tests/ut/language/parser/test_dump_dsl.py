# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser + printer coverage for ``pl.dump(arg)`` — the per-call selective
tensor dump primitive (simpler#844, Layer 1).

``pl.dump(t)`` wraps a kernel-call argument. The parser records the wrapped
argument Var on the IR Call's ``attrs['dump_vars']`` (a ``vector<VarPtr>``),
so the dump target is tracked by Var identity (never by name). The printer
round-trips it by re-wrapping the marked args in ``pl.dump(...)``.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserSyntaxError


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
