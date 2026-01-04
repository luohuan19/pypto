# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for PyPTO IR (Intermediate Representation) module."""

from typing import cast

import pytest
from pypto import DataType, ir


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self):
        """Test creating a Span with valid coordinates."""
        span = ir.Span("test.py", 1, 5, 1, 10)
        assert span.filename == "test.py"
        assert span.begin_line == 1
        assert span.begin_column == 5
        assert span.end_line == 1
        assert span.end_column == 10

    def test_span_to_string(self):
        """Test Span to_string method."""
        span = ir.Span("test.py", 10, 20, 10, 30)
        assert span.to_string() == "test.py:10:20"

    def test_span_str_repr(self):
        """Test Span __str__ and __repr__."""
        span = ir.Span("example.py", 5, 1, 5, 10)
        assert str(span) == "example.py:5:1"
        assert repr(span) == "example.py:5:1"

    def test_span_is_valid(self):
        """Test Span is_valid method."""
        valid_span = ir.Span("test.py", 1, 1, 1, 10)
        assert valid_span.is_valid() is True

        invalid_span = ir.Span("test.py", -1, -1, -1, -1)
        assert invalid_span.is_valid() is False

        # Test multi-line span
        multiline_span = ir.Span("test.py", 1, 5, 3, 10)
        assert multiline_span.is_valid() is True

    def test_span_unknown(self):
        """Test Span.unknown() factory method."""
        span = ir.Span.unknown()
        assert span.filename == ""
        assert span.is_valid() is False

    def test_span_immutability(self):
        """Test that Span attributes are immutable."""
        span = ir.Span.unknown()

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            span.filename = "other.py"  # type: ignore

        with pytest.raises(AttributeError):
            span.begin_line = 5  # type: ignore


class TestOp:
    """Tests for Op class."""

    def test_op_creation(self):
        """Test creating an Op."""
        op = ir.Op("add")
        assert op.name == "add"

    def test_op_name_immutability(self):
        """Test that Op name is immutable."""
        op = ir.Op("multiply")
        with pytest.raises(AttributeError):
            op.name = "divide"  # type: ignore


class TestVar:
    """Tests for Var class."""

    def test_var_creation(self):
        """Test creating a Var expression."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        var = ir.Var("x", span=span)

        assert var.name == "x"
        assert var.span.filename == "test.py"

    def test_var_is_expr(self):
        """Test that Var is an instance of Expr."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        var = ir.Var("x", span=span)

        assert isinstance(var, ir.Expr)
        assert isinstance(var, ir.IRNode)

    def test_var_immutability(self):
        """Test that Var attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        var = ir.Var("x", span=span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            var.name = "y"  # type: ignore


class TestConstInt:
    """Tests for ConstInt class."""

    def test_const_creation(self):
        """Test creating a ConstInt expression."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstInt(42, span=span)

        assert const.value == 42
        assert const.span.filename == "test.py"

    def test_const_integer(self):
        """Test ConstInt with integer values."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstInt(5, span=span)

        assert const.value == 5

    def test_const_is_expr(self):
        """Test that ConstInt is an instance of Expr."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstInt(10, span=span)

        assert isinstance(const, ir.Expr)
        assert isinstance(const, ir.IRNode)

    def test_const_immutability(self):
        """Test that ConstInt attributes are immutable."""
        span = ir.Span("test.py", 1, 1, 1, 5)
        const = ir.ConstInt(42, span=span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            const.value = 100  # type: ignore


class TestBinaryExpressions:
    """Tests for binary expression classes."""

    def test_add_creation(self):
        """Test creating an Add expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)
        add_expr = ir.Add(x, y, span=span)

        assert cast(ir.Var, add_expr.left).name == "x"
        assert cast(ir.Var, add_expr.right).name == "y"

    def test_add_is_binary_expr(self):
        """Test that Add is an instance of BinaryExpr."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)
        add_expr = ir.Add(x, y, span=span)

        assert isinstance(add_expr, ir.BinaryExpr)
        assert isinstance(add_expr, ir.Expr)
        assert isinstance(add_expr, ir.IRNode)

    def test_sub_creation(self):
        """Test creating a Sub expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)
        sub_expr = ir.Sub(x, y, span=span)

        assert cast(ir.Var, sub_expr.left).name == "x"
        assert cast(ir.Var, sub_expr.right).name == "y"

    def test_mul_creation(self):
        """Test creating a Mul expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)
        mul_expr = ir.Mul(x, y, span=span)

        assert cast(ir.Var, mul_expr.left).name == "x"
        assert cast(ir.Var, mul_expr.right).name == "y"

    def test_floatdiv_creation(self):
        """Test creating a FloatDiv expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)
        div_expr = ir.FloatDiv(x, y, span=span)

        assert cast(ir.Var, div_expr.left).name == "x"
        assert cast(ir.Var, div_expr.right).name == "y"

    def test_floormod_creation(self):
        """Test creating a FloorMod expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)
        mod_expr = ir.FloorMod(x, y, span=span)

        assert cast(ir.Var, mod_expr.left).name == "x"
        assert cast(ir.Var, mod_expr.right).name == "y"

    def test_floordiv_creation(self):
        """Test creating a FloorDiv expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)
        floordiv_expr = ir.FloorDiv(x, y, span=span)

        assert cast(ir.Var, floordiv_expr.left).name == "x"
        assert cast(ir.Var, floordiv_expr.right).name == "y"

    def test_comparison_ops(self):
        """Test creating comparison expressions."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)

        eq_expr = ir.Eq(x, y, span=span)
        assert cast(ir.Var, eq_expr.left).name == "x"

        ne_expr = ir.Ne(x, y, span=span)
        assert cast(ir.Var, ne_expr.left).name == "x"

        lt_expr = ir.Lt(x, y, span=span)
        assert cast(ir.Var, lt_expr.left).name == "x"

        le_expr = ir.Le(x, y, span=span)
        assert cast(ir.Var, le_expr.left).name == "x"

        gt_expr = ir.Gt(x, y, span=span)
        assert cast(ir.Var, gt_expr.left).name == "x"

        ge_expr = ir.Ge(x, y, span=span)
        assert cast(ir.Var, ge_expr.left).name == "x"

    def test_logical_ops(self):
        """Test creating logical expressions."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)

        and_expr = ir.And(x, y, span=span)
        assert cast(ir.Var, and_expr.left).name == "x"

        or_expr = ir.Or(x, y, span=span)
        assert cast(ir.Var, or_expr.left).name == "x"

        xor_expr = ir.Xor(x, y, span=span)
        assert cast(ir.Var, xor_expr.left).name == "x"

    def test_bitwise_ops(self):
        """Test creating bitwise expressions."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)

        bitand_expr = ir.BitAnd(x, y, span=span)
        assert cast(ir.Var, bitand_expr.left).name == "x"

        bitor_expr = ir.BitOr(x, y, span=span)
        assert cast(ir.Var, bitor_expr.left).name == "x"

        bitxor_expr = ir.BitXor(x, y, span=span)
        assert cast(ir.Var, bitxor_expr.left).name == "x"

        shl_expr = ir.BitShiftLeft(x, y, span=span)
        assert cast(ir.Var, shl_expr.left).name == "x"

        shr_expr = ir.BitShiftRight(x, y, span=span)
        assert cast(ir.Var, shr_expr.left).name == "x"

    def test_min_max(self):
        """Test creating Min and Max expressions."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)

        min_expr = ir.Min(x, y, span=span)
        assert cast(ir.Var, min_expr.left).name == "x"

        max_expr = ir.Max(x, y, span=span)
        assert cast(ir.Var, max_expr.left).name == "x"

    def test_pow(self):
        """Test creating Pow expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)

        pow_expr = ir.Pow(x, y, span=span)
        assert cast(ir.Var, pow_expr.left).name == "x"
        assert cast(ir.Var, pow_expr.right).name == "y"


class TestUnaryExpressions:
    """Tests for unary expression classes."""

    def test_neg_creation(self):
        """Test creating a Neg expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        neg_expr = ir.Neg(x, span=span)

        assert cast(ir.Var, neg_expr.operand).name == "x"
        assert isinstance(neg_expr, ir.UnaryExpr)
        assert isinstance(neg_expr, ir.Expr)

    def test_abs_creation(self):
        """Test creating an Abs expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        abs_expr = ir.Abs(x, span=span)

        assert cast(ir.Var, abs_expr.operand).name == "x"

    def test_not_creation(self):
        """Test creating a Not expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        not_expr = ir.Not(x, span=span)

        assert cast(ir.Var, not_expr.operand).name == "x"

    def test_bitnot_creation(self):
        """Test creating a BitNot expression."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        bitnot_expr = ir.BitNot(x, span=span)

        assert cast(ir.Var, bitnot_expr.operand).name == "x"


class TestNestedExpressions:
    """Tests for nested expression trees."""

    def test_simple_nested_expression(self):
        """Test building a simple nested expression: (x + 5) * 2."""
        span = ir.Span("test.py", 1, 1, 1, 20)
        x = ir.Var("x", span=span)
        c5 = ir.ConstInt(5, span=span)
        c2 = ir.ConstInt(2, span=span)

        add_expr = ir.Add(x, c5, span=span)
        mul_expr = ir.Mul(add_expr, c2, span=span)

        # Verify structure
        assert isinstance(mul_expr.left, ir.Add)
        assert cast(ir.Var, mul_expr.left.left).name == "x"
        assert cast(ir.ConstInt, mul_expr.left.right).value == 5
        assert cast(ir.ConstInt, mul_expr.right).value == 2

    def test_complex_nested_expression(self):
        """Test building a complex expression: ((a + b) * (c - d))."""
        span = ir.Span("test.py", 1, 1, 1, 30)
        a = ir.Var("a", span=span)
        b = ir.Var("b", span=span)
        c = ir.Var("c", span=span)
        d = ir.Var("d", span=span)

        add_expr = ir.Add(a, b, span=span)
        sub_expr = ir.Sub(c, d, span=span)
        mul_expr = ir.Mul(add_expr, sub_expr, span=span)

        # Verify structure
        assert isinstance(mul_expr.left, ir.Add)
        assert isinstance(mul_expr.right, ir.Sub)
        assert cast(ir.Var, mul_expr.left.left).name == "a"
        assert cast(ir.Var, mul_expr.left.right).name == "b"
        assert cast(ir.Var, mul_expr.right.left).name == "c"
        assert cast(ir.Var, mul_expr.right.right).name == "d"

    def test_deeply_nested_expression(self):
        """Test building a deeply nested expression: (((x + 1) - 2) * 3) / 4."""
        span = ir.Span("test.py", 1, 1, 1, 40)
        x = ir.Var("x", span=span)
        c1 = ir.ConstInt(1, span=span)
        c2 = ir.ConstInt(2, span=span)
        c3 = ir.ConstInt(3, span=span)
        c4 = ir.ConstInt(4, span=span)

        add_expr = ir.Add(x, c1, span=span)
        sub_expr = ir.Sub(add_expr, c2, span=span)
        mul_expr = ir.Mul(sub_expr, c3, span=span)
        div_expr = ir.FloatDiv(mul_expr, c4, span=span)

        # Verify structure depth
        assert isinstance(div_expr.left, ir.Mul)
        assert isinstance(div_expr.left.left, ir.Sub)
        assert isinstance(div_expr.left.left.left, ir.Add)

    def test_all_operations(self):
        """Test expression using multiple operations."""
        span = ir.Span("test.py", 1, 1, 1, 50)
        a = ir.Var("a", span=span)
        b = ir.Var("b", span=span)
        c = ir.Var("c", span=span)
        d = ir.Var("d", span=span)
        e = ir.Var("e", span=span)
        f = ir.Var("f", span=span)

        add_expr = ir.Add(a, b, span=span)
        mul_expr = ir.Mul(c, d, span=span)
        mod_expr = ir.FloorMod(e, f, span=span)
        div_expr = ir.FloatDiv(mul_expr, mod_expr, span=span)
        final_expr = ir.Sub(add_expr, div_expr, span=span)

        # Verify structure
        assert isinstance(final_expr, ir.Sub)
        assert isinstance(final_expr.left, ir.Add)
        assert isinstance(final_expr.right, ir.FloatDiv)

    def test_unary_with_binary(self):
        """Test mixing unary and binary expressions: -(x + 5)."""
        span = ir.Span("test.py", 1, 1, 1, 20)
        x = ir.Var("x", span=span)
        c5 = ir.ConstInt(5, span=span)

        add_expr = ir.Add(x, c5, span=span)
        neg_expr = ir.Neg(add_expr, span=span)

        # Verify structure
        assert isinstance(neg_expr, ir.Neg)
        assert isinstance(neg_expr.operand, ir.Add)
        assert cast(ir.Var, neg_expr.operand.left).name == "x"


class TestImmutability:
    """Tests for immutability of IR nodes."""

    def test_expr_operands_immutable(self):
        """Test that binary expression operands cannot be modified."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)
        y = ir.Var("y", span=span)
        add_expr = ir.Add(x, y, span=span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            add_expr.left = ir.Var("z", span=span)  # type: ignore

    def test_span_immutable_in_node(self):
        """Test that span attribute in IRNode is immutable."""
        span = ir.Span.unknown()
        x = ir.Var("x", span=span)

        # Attempting to modify should raise AttributeError
        with pytest.raises(AttributeError):
            x.span = ir.Span("other.py", 2, 2, 2, 5)  # type: ignore


class TestSpanTracking:
    """Tests for source location tracking via Span."""

    def test_span_preserved_in_tree(self):
        """Test that spans are preserved throughout the expression tree."""
        span1 = ir.Span("file1.py", 1, 1, 1, 5)
        span2 = ir.Span("file2.py", 2, 2, 2, 5)
        span3 = ir.Span("file3.py", 3, 3, 3, 10)

        x = ir.Var("x", span=span1)
        y = ir.Var("y", span=span2)
        add_expr = ir.Add(x, y, span=span3)

        assert x.span.filename == "file1.py"
        assert y.span.filename == "file2.py"
        assert add_expr.span.filename == "file3.py"

    def test_unknown_span(self):
        """Test creating IR nodes with unknown spans."""
        unknown_span = ir.Span.unknown()
        x = ir.Var("x", dtype=DataType.INT64, span=unknown_span)

        assert x.span.is_valid() is False
        assert x.span.filename == ""


if __name__ == "__main__":
    pytest.main(["-v", __file__])
