# mypy: allow-untyped-defs
import functools
import operator
from functools import reduce
from typing import Any, Tuple

import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import has_free_symbols
import copy
from .. import ir
from ..lowering import lowerings as L
from ..lowering import empty_strided
from ..pattern_matcher import (
    Arg,
    Ignored,
    CallFunction,
    filter_nodes,
    get_arg_value,
    KeywordArg,
    ListOf,
    MULTIPLE,
)
from ..virtualized import ops, V
from .freezing_patterns import register_freezing_graph_pattern, register_concat_folding_pattern
from .post_grad import register_lowering_pattern
from .quantization import (
    _register_quantization_lowerings,
    _register_quantization_weight_pack_pass,
    _register_woq_lowerings,
)
from torch._prims_common import (
    make_channels_last_strides_for,
)
from ..utils import pad_listlike

if torch._C._has_mkldnn:
    aten = torch.ops.aten
    mkldnn = torch.ops.mkldnn
    prims = torch.ops.prims

    _conv_args = [Arg() for _ in range(10)]
    _linear_args = [Arg() for _ in range(6)]
    _conv_transpose_args = [Arg() for _ in range(11)]

    def _conv_call(users=1):
        return CallFunction(
            mkldnn._convolution_pointwise.default, *_conv_args, _users=users
        )

    def _relu_fusion(computation_call, users=1):
        return CallFunction(aten.relu, computation_call, _users=users)

    def _conv_cat_call(conv_call, users=1):
        return CallFunction(
            aten.cat.default,
            ListOf(
                conv_call,
            ),
            KeywordArg("dim"),
            _users=users
        )

    def _linear_call(users=1):
        return CallFunction(
            mkldnn._linear_pointwise.default, *_linear_args, _users=users
        )

    def _conv_transpose_call(users=1):
        return CallFunction(
            mkldnn._convolution_transpose_pointwise.default,
            *_conv_transpose_args,
            _users=users,
        )

    def _to_float(input_call, users=1):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_float"),
            _users=users,
        )

    def _to_bf16(input_call):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_bf16"),
            _users=1,
        )

    def _to_fp16(input_call):
        return CallFunction(
            prims.convert_element_type.default,
            input_call,
            KeywordArg("to_fp16"),
            _users=1,
        )

    def _unary_fusion_pattern(unary_fusion, call_fn, users, lowp_dtype):
        # only insert to_dtype if lowp_dtype is True
        computation_call = (
            _to_float(call_fn(), users=users) if lowp_dtype else call_fn(users=users)
        )
        out = unary_fusion(computation_call)
        if lowp_dtype == torch.bfloat16:
            return _to_bf16(out)
        elif lowp_dtype == torch.float16:
            return _to_fp16(out)
        else:
            return out

    def _gelu_fusion_1(computation_call):
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.erf,
                    CallFunction(aten.mul, computation_call, 0.7071067811865476),
                ),
                1,
            ),
        )

    def _gelu_fusion_2(computation_call):
        return CallFunction(
            aten.mul,
            CallFunction(aten.mul, computation_call, 0.5),
            CallFunction(
                aten.add,
                CallFunction(
                    aten.tanh,
                    CallFunction(
                        aten.mul,
                        CallFunction(
                            aten.add,
                            computation_call,
                            CallFunction(
                                aten.mul,
                                CallFunction(
                                    aten.mul,
                                    CallFunction(
                                        aten.mul, computation_call, computation_call
                                    ),
                                    computation_call,
                                ),
                                0.044715,
                            ),
                        ),
                        0.7978845608028654,
                    ),
                ),
                1,
            ),
        )

    def _hardswish_fusion(computation_call):
        return CallFunction(
            aten.div,
            CallFunction(
                aten.mul,
                computation_call,
                CallFunction(
                    aten.clamp_max,
                    CallFunction(
                        aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                    ),
                    6,
                ),
            ),
            6,
        )

    def _silu_fusion(computation_call):
        return CallFunction(
            aten.mul, computation_call, CallFunction(aten.sigmoid, computation_call)
        )

    def _hardsigmoid_fusion(computation_call):
        return CallFunction(
            aten.div,
            CallFunction(
                aten.clamp_max,
                CallFunction(
                    aten.clamp_min, CallFunction(aten.add, computation_call, 3), 0
                ),
                6,
            ),
            6,
        )

    def _leaky_relu_fusion(computation_call):
        return CallFunction(
            aten.where,
            CallFunction(aten.gt, computation_call, 0),
            computation_call,
            CallFunction(aten.mul, computation_call, KeywordArg("negative_slope")),
        )

    def _hardtanh_fusion(computation_call):
        return CallFunction(
            aten.clamp_max,
            CallFunction(aten.clamp_min, computation_call, KeywordArg("min_value")),
            KeywordArg("max_value"),
        )

    def _combined_fusion(computation_call, elementwise_op):
        return CallFunction(elementwise_op, computation_call)

    # binary_op(other, computation_op)
    def _binary_fusion_v1(computation_call, binary_fn):
        return CallFunction(binary_fn, KeywordArg("other"), computation_call)

    # binary_op(computation_op, other)
    def _binary_fusion_v2(computation_call, binary_fn):
        return CallFunction(binary_fn, computation_call, KeywordArg("other"))

    def _is_single_computation_op(computation_op, lowp_dtype=None):
        def fn(match):
            computation_nodes = filter_nodes(match.nodes, computation_op)

            if lowp_dtype:
                output_node_meta = match.output_node().meta.get("val")
                if output_node_meta.dtype != lowp_dtype:
                    return False

            if len(computation_nodes) < 1:
                return False
            if any(n.args[-3] != "none" for n in computation_nodes):
                return False
            return True

        return fn

    def _is_valid_computation_unary_fusion(computation_op, lowp_dtype=None):
        def fn(match):
            matched = _is_single_computation_op(computation_op, lowp_dtype)(match)
            computation_node = filter_nodes(match.nodes, computation_op)[0]
            if lowp_dtype:
                conversion_dtype_nodes = filter_nodes(
                    match.nodes, prims.convert_element_type.default
                )
                if len(conversion_dtype_nodes) != 2:
                    return False
                # fusion pattern is always in the form of computation_op + to_float32 + unary_op + to_bfloat16
                if computation_node == conversion_dtype_nodes[0].args[0]:
                    to_float = conversion_dtype_nodes[0].args[1]
                    to_lp = conversion_dtype_nodes[1].args[1]
                else:
                    to_float = conversion_dtype_nodes[1].args[1]
                    to_lp = conversion_dtype_nodes[0].args[1]
                matched = matched and to_float == torch.float and to_lp == lowp_dtype
            return matched

        return fn

    def _is_valid_conv_concat_fusion(computation_op, lowp_dtype=None):
        def fn(match):
            cat_node = match.output_node()
            assert cat_node.target == aten.cat.default

            # matched = _is_single_computation_op(computation_op, lowp_dtype)(match)
            # computation_node = filter_nodes(match.nodes, computation_op)[0]
            # graph = match.graph
            # for node in computation_node.users:
            #     print("node: ", node)
            #     if node.target == aten.cat.default:
            #         all_conv = True
            #         for arg_node in node.args:
            #             if arg_node.target != mkldnn._convolution_pointwise.default:
            #                 all_conv = False
            #         return len(node.args) > 0 and all_conv
            #     else:
            #         return False


            

            # if lowp_dtype:
            #     conversion_dtype_nodes = filter_nodes(
            #         match.nodes, prims.convert_element_type.default
            #     )
            #     if len(conversion_dtype_nodes) != 2:
            #         return False
            #     # fusion pattern is always in the form of computation_op + to_float32 + unary_op + to_bfloat16
            #     if computation_node == conversion_dtype_nodes[0].args[0]:
            #         to_float = conversion_dtype_nodes[0].args[1]
            #         to_lp = conversion_dtype_nodes[1].args[1]
            #     else:
            #         to_float = conversion_dtype_nodes[1].args[1]
            #         to_lp = conversion_dtype_nodes[0].args[1]
            #     matched = matched and to_float == torch.float and to_lp == lowp_dtype
            # return matched

        return fn

    # reuse pattern of mkldnn fusion, i.e, mkldnn fusion + concat 
    def _register_conv_concat_fusion_lowering(
            pattern, unary_attr, computation_op, lowp_dtype=None
    ):
        # new_pattern = CallFunction(aten.cat, ListOf(Arg(),), Ignored(), _users=1) # ListOf(pattern)
        conv_cat_call = _conv_cat_call(pattern)#_conv_cat_call(_silu_fusion(_conv_call(2)))#_conv_cat_call(_conv_call())
        # conv_call = (_conv_call(users=2))
        # # conv_cat_call=_conv_cat_call(_silu_fusion(conv_call))
        # conv_cat_call = _silu_fusion(conv_call)
        # conv_cat_call = _conv_cat_call(conv_cat_call)
        def comp_conv_out_size(x, w, b, padding, stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
            ):
            with V.graph.fake_mode:
                x_fake = ir.ir_node_to_tensor(x, guard_shape=True)
                weight_fake = ir.ir_node_to_tensor(w, guard_shape=True)
                dims = len(x_fake.size()) - 2
                assert 0 < len(padding) <= dims
                assert 0 < len(dilation) <= dims
                assert 0 < len(stride) <= dims
                padding = pad_listlike(padding, dims)
                dilation = pad_listlike(dilation, dims)
                stride = pad_listlike(stride, dims)
                transposed=False
                bias_fake = (
                    ir.ir_node_to_tensor(b, guard_shape=True) if b is not None else b
                )
                output_padding = pad_listlike([0], dims)
                output = torch.ops.aten.convolution(
                    x_fake,
                    weight_fake,
                    bias_fake,
                    stride,
                    padding,
                    dilation,
                    transposed,
                    output_padding,
                    groups,
                )
                return list(output.size())

        @register_lowering_pattern(
            conv_cat_call,
            # extra_check=_is_valid_conv_concat_fusion(computation_op, lowp_dtype),
            pass_number=0,
        )
        def fn(match, *args, **kwargs):
            # 1: compute cat output size
            # 2: create create cat kernel
            # 3: create conv with output strides
            computation_op=mkldnn.mkldnn_convolution_with_out_stride
            dim = kwargs["dim"]
            conv_out_size = comp_conv_out_size(*(args[0][0]))
            conv_out_buf_size = copy.deepcopy(conv_out_size)
            cat_out_size = conv_out_size#list(args[0][0].get_size())#(8, 16, 30, 30)
            
            device = args[0][0][0].get_device()
            dtype = args[0][0][0].get_dtype()
            
            # compute_op = mkldnn._convolution_pointwise.default
            offsets_start = [0]
            offsets_end = [cat_out_size[dim]]

            for i in range(1, len(args[0])):
                input_size = conv_out_size
                offsets_start.append(cat_out_size[dim])
                assert len(input_size) == len(cat_out_size)
                # assert args[0][i].get_dtype() == dtype
                # assert args[0][i].get_device() == device
                for j in range(len(cat_out_size)):
                    if j == dim:
                        cat_out_size[j] = cat_out_size[j] + input_size[j]
                    else:
                        cat_out_size[j] = V.graph.sizevars.guard_equals(
                            cat_out_size[j], input_size[j]
                        )
                offsets_end.append(cat_out_size[dim])
            conv_out_strides = make_channels_last_strides_for(cat_out_size)
            # data = ir.OperationBuffer(
            #     name=None,
            #     layout=ir.FixedLayout(
            #         device=device,
            #         dtype=dtype,
            #         size=cat_out_size,
            #         stride=conv_out_strides,
            #     ),
            #     # inputs=[],
            # )
            # data.name = V.graph.register_buffer(data)
            # concat_kernel = ir.ComputedBuffer(
            #     name=None,
            #     layout=ir.FixedLayout(
            #         device=data.get_device(),
            #         dtype=data.get_dtype(),
            #         size=data.get_size(),
            #     ),
            #     data=data,
            # )
            concat_kernel = ir.ConcatKernel(
                name=None,
                layout=ir.FixedLayout(
                    device=device,
                    dtype=dtype,
                    size=cat_out_size,
                    stride=conv_out_strides,
                ),
                inputs=[],
            )
            concat_kernel.name = V.graph.register_buffer(concat_kernel)
            kernel = ir.StorageBox(concat_kernel)
            for i in range(len(args[0])):
                # conv_out_buf_stride = make_channels_last_strides_for(conv_out_buf_size)
                # conv_out_buf = empty_strided(conv_out_buf_size, conv_out_buf_stride, dtype=dtype, device=device)
                input_buffer = ir.SliceView.create(
                    kernel, dim, offsets_start[i], offsets_end[i], clamp=False
                )
                # conv_out_buf.data.data.layout = ir.NonOwningLayout(input_buffer)
                # input_buffer.realize()
                # input_buffer.name = V.graph.register_buffer(input_buffer)
                # concat_kernel.inputs.append(input_buffer)
                computation_args = list(args[0][i])[:-3] + [
                    unary_attr.op_name,
                    unary_attr.scalars_attr,
                    unary_attr.algorithm_attr,
                ]
                computation_args.insert(6, conv_out_strides)
                computation_args.insert(0, input_buffer)
                L[computation_op](*computation_args)
            # concat_kernel.inputs = ir.InputsKernel.unwrap_storage(concat_kernel.inputs)
            V.graph.register_operation(concat_kernel)

            return ir.TensorBox(kernel)
            # out = L[emp](size)
            # output_stride = (14400, 900, 30, 1)
            # offset = 57600
            # dim = 0
            # len_dim = size[dim]
            # graph = match.graph
            # for node in graph.nodes:
            #     if node.target == aten.cat.default:
            #         empty_node = graph.call_function(aten.empty, node.args)
            #         for i, arg_node in enumerate(node.args[0]):
            #             print("arg_node.target: ", arg_node.target)
            #             for i, arg_node2 in enumerate(arg_node.args):
            #                 print("arg_node2.target: ", arg_node2.target)
            #                 if arg_node2.target == aten.convolution.default:
            #                     print("--------arg_node2---------: ", arg_node2)
            #                     computation_args = list(node.args)[:-3] + [
            #                         empty_node,
            #                         output_stride,
            #                         i * offset
            #                     ]
            #                     # computation_args[-2] = aten.as_strided(out, small_size, offset=i * offset)
            #                     conv_out_node = graph.call_function(mkldnn.mkldnn_convolution_with_out_stride, computation_args)
            #                     node.replace_all_uses_with(conv_out_node)
            #                     conv_out_node.meta.update(node.meta)
            #                     graph.erase_node(node)
            #     node.replace_all_uses_with(empty_node)
            #     empty_node.meta.update(node.meta)
            #     graph.erase_node(node)
            # print(graph)
            # exit()
            # output_stride = ()
            # out =  L[computation_op](*computation_args)
            # return out
            # return L[computation_op](*computation_args)

        return fn

    def _register_unary_fusion_lowering(
        pattern, unary_attr, computation_op, lowp_dtype=None
    ):
        # @register_lowering_pattern(
        #     pattern,
        #     extra_check=_is_valid_computation_unary_fusion(computation_op, lowp_dtype),
        # )
        def fn(match, *args, **kwargs):
            computation_args = list(args)[:-3] + [
                unary_attr.op_name,
                unary_attr.scalars_attr,
                unary_attr.algorithm_attr,
            ]
            counters["inductor"]["mkldnn_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"] += len(
                match.nodes
            )
            return L[computation_op](*computation_args)

        return fn

    def _register_leaky_relu_fusion_lowering(pattern, computation_op, lowp_dtype=None):
        @register_lowering_pattern(
            pattern, extra_check=_is_single_computation_op(computation_op, lowp_dtype)
        )
        def fn(match, *args, **kwargs):
            negative_slope = kwargs.get("negative_slope")
            if isinstance(negative_slope, ir.TensorBox):
                matched = False
            else:  # inp is a Number
                matched = True
            if lowp_dtype:
                dtype1 = kwargs.get("to_float")
                dtype2 = (
                    kwargs.get("to_bf16")
                    if lowp_dtype == torch.bfloat16
                    else kwargs.get("to_fp16")
                )
                matched = matched and dtype1 == torch.float and dtype2 == lowp_dtype
            computation_args = list(args)
            counters["inductor"]["mkldnn_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"] += len(
                match.nodes
            )
            if matched:
                computation_args = computation_args[:-3] + [
                    "leaky_relu",
                    [negative_slope],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                # computation_args += ["none", [], ""]
                out = L[computation_op](*computation_args)
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=torch.float)
                out = L[aten.where](
                    L[aten.gt](out, 0),
                    out,
                    L[aten.mul](out, negative_slope),
                )
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=dtype2)  # type: ignore[possibly-undefined]
                return out

        return fn

    def _register_hardtanh_fusion_lowering(pattern, computation_op, lowp_dtype=None):
        @register_lowering_pattern(
            pattern, extra_check=_is_single_computation_op(computation_op, lowp_dtype)
        )
        def fn(match, *args, **kwargs):
            min_value = kwargs.get("min_value")
            max_value = kwargs.get("max_value")
            if isinstance(min_value, ir.TensorBox) or isinstance(
                max_value, ir.TensorBox
            ):
                matched = False
            else:  # inp is a Number
                assert max_value is not None
                matched = min_value <= max_value
            if lowp_dtype:
                dtype1 = kwargs.get("to_float")
                dtype2 = (
                    kwargs.get("to_bf16")
                    if lowp_dtype == torch.bfloat16
                    else kwargs.get("to_fp16")
                )
                matched = matched and dtype1 == torch.float and dtype2 == lowp_dtype
            computation_args = list(args)
            counters["inductor"]["mkldnn_unary_fusion_matcher_count"] += 1
            counters["inductor"]["mkldnn_unary_fusion_matcher_nodes"] += len(
                match.nodes
            )
            if matched:
                computation_args = computation_args[:-3] + [
                    "hardtanh",
                    [min_value, max_value],
                    "",
                ]
                return L[computation_op](*computation_args)
            else:
                out = L[computation_op](*computation_args)
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=torch.float)
                out = L[aten.clamp_max](L[aten.clamp_min](out, min_value), max_value)
                if lowp_dtype:
                    out = L[prims.convert_element_type.default](out, dtype=dtype2)  # type: ignore[possibly-undefined]
                return out

        return fn

    _binary_attr = {
        aten.add: "add",
        ops.add: "add",
        aten.sub: "sub",
        ops.sub: "sub",
    }

    def _is_valid_binary(match, fn):
        binary_nodes = filter_nodes(match.nodes, fn)
        if len(binary_nodes) < 1:
            return False

        def get_meta_value(argument: torch.fx.node.Argument):
            # Only torch.fx.Node is expected to have meta.
            if isinstance(argument, torch.fx.Node):
                return argument.meta.get("val", None)
            return None

        if any(
            not isinstance(get_meta_value(n.args[0]), torch.Tensor)
            or not isinstance(get_meta_value(n.args[1]), torch.Tensor)
            for n in binary_nodes
        ):
            return False
        # check alpha is one.
        if any(
            get_arg_value(n, 2, kwarg_name="alpha") != 1.0
            and get_arg_value(n, 2, kwarg_name="alpha") is not None
            for n in binary_nodes
        ):
            return False

        if any(
            get_meta_value(n.args[0]).dim() != get_meta_value(n.args[1]).dim()
            or not all(
                get_meta_value(n.args[0]).size(i) == get_meta_value(n.args[1]).size(i)
                or get_meta_value(match.kwargs["other"]).size(i) == 1
                for i in range(get_meta_value(n.args[0]).dim())
            )
            or get_meta_value(n.args[0]).device != get_meta_value(n.args[1]).device
            or get_meta_value(n.args[0]).dtype != get_meta_value(n.args[1]).dtype
            for n in binary_nodes
        ):
            return False
        # check args[0] and args[1] is not same
        if any(n.args[0] == n.args[1] for n in binary_nodes):
            return False
        return True

    def _is_valid_computation_binary(computation_op, binary_op, other_index=None):
        def fn(match):
            if not _is_single_computation_op(computation_op)(match):
                return False
            if not _is_valid_binary(match, binary_op):
                return False
            return True

        return fn

    def _get_remaining_users(extra_input_node, compute_node):
        # Think about this pattern:
        #      ReLU
        #     /   \
        #  Conv1
        #   /      \
        # Conv2
        #   \      /
        #      Add
        # Although, the extra input node (ReLU) has more than 1 users: Conv1 and Add.
        # The Conv1 is the ancestor node of the current compute node (Conv2).
        # This indicates that the buffer of ReLU has completed all its usage,
        # So we can safely make changes to it now by doing Conv2->Add inplace fusion.
        # Take above case as example:
        # * extra_input_node: ReLU
        # * compute_node: Conv2
        # _get_remaining_users will return the users of extra_input_node which are not
        # ancestor node of compute_node.
        def _is_ancestor_node(_current_node, _ancestor_node):
            # Check whether _ancestor_node is the ancestor node of _current_node
            _node_list = [_current_node]
            _visited_nodes = set()
            while len(_node_list) != 0:
                _current_node = _node_list.pop(0)
                if _current_node not in _visited_nodes:
                    _visited_nodes.add(_current_node)
                    if _current_node == _ancestor_node:
                        return True
                    elif isinstance(
                        _current_node, torch.fx.Node
                    ) and _current_node.op not in ["placeholder", "output", "get_attr"]:
                        for input in _current_node.all_input_nodes:
                            _node_list.append(input)  # noqa: PERF402
            return False

        return [
            user
            for user in list(extra_input_node.users)
            if not _is_ancestor_node(compute_node, user)
        ]

    def _is_valid_computation_binary_inplace(computation_op, binary_op, other_index):
        def fn(match):
            if not _is_valid_computation_binary(computation_op, binary_op)(match):
                return False
            binary_nodes = filter_nodes(match.nodes, binary_op)

            def _get_compute_node(_binary_node, _other_index):
                assert (
                    len(_binary_node.all_input_nodes) == 2
                ), "Binary node should have 2 input nodes."
                _compute_index = 1 if (_other_index == 0) else 0
                return _binary_node.args[_compute_index]

            def _other_input_not_inplaceable(_binary_node, _other_index):
                _compute_node = _get_compute_node(_binary_node, _other_index)
                return (
                    len(
                        _get_remaining_users(
                            _binary_node.args[_other_index], _compute_node
                        )
                    )
                    > 1
                    or _binary_node.args[_other_index] == _compute_node.args[0]
                )

            if any(_other_input_not_inplaceable(n, other_index) for n in binary_nodes):
                return False
            if any(
                n.args[other_index].op in ["placeholder", "output"]
                for n in binary_nodes
            ):
                return False
            return True

        return fn

    def _register_binary_unary_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        fusion_op,
        unary_attr=None,
    ):
        @register_lowering_pattern(
            pattern, extra_check=_is_valid_computation_binary(computation_op, binary_op)
        )
        def fn(match, *args, **kwargs):
            other = kwargs.get("other")
            assert isinstance(other, ir.TensorBox)
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_count"] += 1
            counters["inductor"][
                "mkldnn_conv_binary_unary_fusion_matcher_nodes"
            ] += len(match.nodes)
            return L[fusion_op](*computation_args)

        return fn

    def _can_be_inplace(_other):
        if isinstance(_other.data, ir.View):
            return _can_be_inplace(_other.data)
        else:
            a =  not (
                isinstance(_other.data, ir.ReinterpretView)
                or len(_other.get_inputs_that_alias_output()) > 0
            )
            print("------------------- a: ", a)
            return a

    def _register_unary_maybe_concat_fusion_lowering(
        pattern,
        conv_out_op,
    ):
        @register_lowering_pattern(
            pattern,
            # extra_check=_is_valid_conv_concat_fusion(
            #     conv_out_op
            # ),
            # pass_number=0
        )
        def fn(match, *args, **kwargs):
            args_list = list(args)
            # computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            # computation_args = args_list
            # return L[conv_out_op](*computation_args)
        #     device= args_list[1].get_device()
        #     dtype = args_list[1].get_dtype()
        #     new_size = list(inputs[0].get_size())
        #     offsets_start = [0]
        #     offsets_end = [new_size[dim]]
        #     assert 0 <= dim < len(new_size)
        #     for i in range(1, len(inputs)):
        #         input_size = inputs[i].get_size()
        #         offsets_start.append(new_size[dim])
        #         assert len(input_size) == len(new_size)
        #         assert inputs[i].get_dtype() == dtype
        #         assert inputs[i].get_device() == device
        #         for j in range(len(new_size)):
        #             if j == dim:
        #                 new_size[j] = new_size[j] + input_size[j]
        #             else:
        #                 new_size[j] = V.graph.sizevars.guard_equals(
        #                     new_size[j], input_size[j]
        #                 )
        #         offsets_end.append(new_size[dim])
        #     concat_kernel = ir.ConcatKernel(
        #     name=None,
        #     layout=ir.FixedLayout(
        #         device=device,
        #         dtype=dtype,
        #         size=new_size,
        #         stride=output_stride,
        #     ),
        #     inputs=[],
        # )
        # kernel = StorageBox(concat_kernel)

        return fn

    def _register_binary_unary_maybe_inplace_fusion_lowering(
        pattern,
        computation_op,
        binary_op,
        inplace_fusion_op,
        outplace_fusion_op,
        unary_attr=None,
        other_index=None,
    ):
        @register_lowering_pattern(
            pattern,
            extra_check=_is_valid_computation_binary_inplace(
                computation_op, binary_op, other_index
            ),
        )
        def fn(match, *args, **kwargs):
            other = kwargs.get("other")
            assert isinstance(other, ir.TensorBox)
            binary_attr = _binary_attr[binary_op]
            args_list = list(args)
            computation_args = [args_list[0], other] + args_list[1:-3] + [binary_attr]
            if len(args_list) > 6:
                if unary_attr is not None:
                    computation_args += [
                        1.0,
                        unary_attr.op_name,
                        unary_attr.scalars_attr,
                        unary_attr.algorithm_attr,
                    ]
                else:
                    computation_args += [1.0, None, [], None]
            counters["inductor"]["mkldnn_conv_binary_unary_fusion_matcher_count"] += 1
            counters["inductor"][
                "mkldnn_conv_binary_unary_fusion_matcher_nodes"
            ] += len(match.nodes)
            # Make sure the other is not an alias or mutation(fx side doesn't has such info).
            other.realize()
            if not _can_be_inplace(other) or other.data.shape != list(
                match.nodes[0].meta["val"].size()
            ):
                return L[outplace_fusion_op](*computation_args)
            return L[inplace_fusion_op](*computation_args)

        return fn

    computation_ops = [
        mkldnn._convolution_pointwise.default,
        mkldnn._linear_pointwise.default,
        mkldnn._convolution_transpose_pointwise.default,
    ]

    class UnaryAttr:
        def __init__(
            self, op_name: str, scalars_attr=None, algorithm_attr=None
        ) -> None:
            self.op_name = op_name
            self.scalars_attr = scalars_attr if scalars_attr else []
            self.algorithm_attr = algorithm_attr if algorithm_attr else ""

    def _register_unary_fusion():
        computation_call_fns = [_conv_call, _linear_call, _conv_transpose_call]

        def _unary_fusion_patterns(lowp_dtype):
            replacement_unary_fusion_patterns = {
                UnaryAttr("gelu", algorithm_attr="tanh"): [
                    _unary_fusion_pattern(_gelu_fusion_2, call_fn, 4, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("gelu", algorithm_attr="none"): [
                    _unary_fusion_pattern(_gelu_fusion_1, call_fn, 2, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("hardswish"): [
                    _unary_fusion_pattern(_hardswish_fusion, call_fn, 2, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("hardsigmoid"): [
                    _unary_fusion_pattern(_hardsigmoid_fusion, call_fn, 1, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
                UnaryAttr("swish"): [
                    _unary_fusion_pattern(_silu_fusion, call_fn, 2, lowp_dtype)
                    for call_fn in computation_call_fns
                ],
            }
            if not lowp_dtype:
                call_user1 = [call_fn(users=1) for call_fn in computation_call_fns]
                replacement_unary_fusion_patterns.update(
                    {
                        UnaryAttr("relu"): [
                            _combined_fusion(u, aten.relu) for u in call_user1
                        ],
                        UnaryAttr("sigmoid"): [
                            _combined_fusion(u, aten.sigmoid) for u in call_user1
                        ],
                        UnaryAttr("tanh"): [
                            _combined_fusion(u, aten.tanh) for u in call_user1
                        ],
                    }
                )

            return replacement_unary_fusion_patterns

        for lowp_dtype in [torch.bfloat16, torch.float16, None]:
            replace_patterns = _unary_fusion_patterns(lowp_dtype)
            for unary_attr, patterns in replace_patterns.items():
                _register_unary_fusion_lowering(
                    patterns[0], unary_attr, computation_ops[0], lowp_dtype
                )
                _register_unary_fusion_lowering(
                    patterns[1], unary_attr, computation_ops[1], lowp_dtype
                )
                _register_unary_fusion_lowering(
                    patterns[2], unary_attr, computation_ops[2], lowp_dtype
                )
            _leaky_relu_patterns = [
                _unary_fusion_pattern(_leaky_relu_fusion, call_fn, 3, lowp_dtype)
                for call_fn in computation_call_fns
            ]
            for pattern, computation_op in zip(_leaky_relu_patterns, computation_ops):
                _register_leaky_relu_fusion_lowering(
                    pattern, computation_op, lowp_dtype
                )
            hardtanh_patterns = [
                _unary_fusion_pattern(_hardtanh_fusion, call_fn, 1, lowp_dtype)
                for call_fn in computation_call_fns
            ]
            for pattern, computation_op in zip(hardtanh_patterns, computation_ops):
                _register_hardtanh_fusion_lowering(pattern, computation_op, lowp_dtype)

        for lowp_dtype in [torch.bfloat16, torch.float16, None]:
            replace_patterns = _unary_fusion_patterns(lowp_dtype)
            for unary_attr, patterns in replace_patterns.items():
                _register_conv_concat_fusion_lowering(
                    patterns[0], unary_attr, computation_ops[0], lowp_dtype
                )

    def _register_conv_out_stride_fusion():
        fusion_op = mkldnn.mkldnn_convolution_with_out_stride
        conv_cat_call = _conv_cat_call(_conv_call())
        _register_unary_maybe_concat_fusion_lowering(
            conv_cat_call,
            fusion_op,
        )

    def _register_binary_fusion():
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [
            mkldnn._convolution_pointwise.binary,
            mkldnn._linear_pointwise.binary,
        ]
        _computation_user_1 = [_conv_call(users=1), _linear_call(users=1)]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern = _binary_fusion_v2(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

            for binary_op in [aten.add, ops.add]:
                pattern = _binary_fusion_v1(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

    def _register_inplace_fusion():
        binary_ops = [aten.add, ops.add]
        inplace_fusion_op = mkldnn._convolution_pointwise_.binary
        outplace_fusion_op = mkldnn._convolution_pointwise.binary
        conv_call = _conv_call(users=1)
        conv_op = computation_ops[0]
        for binary_op in binary_ops:
            binary_v1 = _binary_fusion_v1(conv_call, binary_op)
            binary_unary_v1 = _combined_fusion(binary_v1, aten.relu)
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v1,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
                unary_attr=UnaryAttr("relu"),
            )
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v1,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=0,
            )
            binary_v2 = _binary_fusion_v2(conv_call, binary_op)
            binary_unary_v2 = _combined_fusion(binary_v2, aten.relu)
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_unary_v2,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
                unary_attr=UnaryAttr("relu"),
            )
            _register_binary_unary_maybe_inplace_fusion_lowering(
                binary_v2,
                conv_op,
                binary_op,
                inplace_fusion_op,
                outplace_fusion_op,
                other_index=1,
            )

    def _register_binary_fusion():
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [
            mkldnn._convolution_pointwise.binary,
            mkldnn._linear_pointwise.binary,
        ]
        _computation_user_1 = [_conv_call(users=1), _linear_call(users=1)]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern = _binary_fusion_v2(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

            for binary_op in [aten.add, ops.add]:
                pattern = _binary_fusion_v1(computation_call, binary_op)
                _register_binary_unary_fusion_lowering(
                    pattern, computation_op, binary_op, fusion_op
                )

    def _register_binary_unary_fusion():
        binary_ops = [aten.add, ops.add, aten.sub, ops.sub]
        fusion_ops = [mkldnn._convolution_pointwise.binary]
        _computation_user_1 = [_conv_call(users=1)]
        for computation_call, computation_op, fusion_op in zip(
            _computation_user_1, computation_ops[:-1], fusion_ops
        ):
            for binary_op in binary_ops:
                pattern_v1 = _combined_fusion(
                    _binary_fusion_v2(computation_call, binary_op), aten.relu
                )
                _register_binary_unary_fusion_lowering(
                    pattern_v1,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )
            for binary_op in [aten.add, ops.add]:
                pattern_v2 = _combined_fusion(
                    _binary_fusion_v1(computation_call, binary_op), aten.relu
                )
                _register_binary_unary_fusion_lowering(
                    pattern_v2,
                    computation_op,
                    binary_op,
                    fusion_op,
                    unary_attr=UnaryAttr("relu"),
                )

    def _recover_linear():
        # convert reshape+linear+reshape to a single linear for applying fusion path.
        # concat_linear (pass_number=0) -> mkldnn_linear_pack (pass_numer=1) -> _recover_linear(pass_number=2)
        @register_freezing_graph_pattern(
            CallFunction(
                aten.reshape.default,
                CallFunction(
                    mkldnn._linear_pointwise.default,
                    CallFunction(
                        aten.reshape.default,
                        Arg(),
                        KeywordArg("reshape_1"),
                        _users=MULTIPLE,
                    ),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                ),
                KeywordArg("reshape_2"),
            ),
            pass_number=2,
        )
        def reshape_linear_reshape_pattern(match, *args, **kwargs):
            def get_val(val):
                return val if isinstance(val, int) else val.meta.get("val")

            reshape_1 = kwargs.get("reshape_1")
            reshape_2 = kwargs.get("reshape_2")
            assert isinstance(reshape_1, list)
            assert isinstance(reshape_2, list)
            assert len(reshape_1) == 2

            graph = match.graph
            reshape_2_node = match.output_node()
            linear_input_node = reshape_2_node.args[0].args[0].args[0]
            # check linear's input's shape[:-1] == reshape_2[:-1]
            # and check product(reshape_2[:-1]) == reshape_1[0]
            can_remove_reshape = linear_input_node.meta.get("val").shape[
                :-1
            ] == torch.Size([get_val(val) for val in reshape_2[:-1]])
            can_remove_reshape = can_remove_reshape and (
                reduce(
                    operator.mul,
                    [get_val(val) for val in reshape_2[:-1]],
                )
                == get_val(reshape_1[0])
            )

            if can_remove_reshape:
                repl = graph.call_function(mkldnn._linear_pointwise.default, args)
                repl.meta.update(reshape_2_node.meta)
                reshape_2_node.replace_all_uses_with(repl)
                old_linear_node = reshape_2_node.args[0]
                reshape_1_node = old_linear_node.args[0]
                graph.erase_node(reshape_2_node)
                graph.erase_node(old_linear_node)
                if len(reshape_1_node.users) == 0:
                    graph.erase_node(reshape_1_node)
            counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_count"] += 1
            counters["inductor"]["mkldnn_reshape_linear_reshape_matcher_nodes"] += len(
                match.nodes
            )

        def is_linear_add_bias(match):
            add_node = match.output_node()
            linear_node = add_node.args[0]
            packed_weight_node = linear_node.args[1]
            assert packed_weight_node.target == mkldnn._reorder_linear_weight
            transpose_weight_node = packed_weight_node.args[0]
            assert transpose_weight_node.target == aten.permute.default
            weight_meta = transpose_weight_node.args[0].meta.get("val")
            bias_node = add_node.args[1]
            if isinstance(bias_node, int):
                # we only folding bias if it is a constant
                return False
            bias_meta = add_node.args[1].meta.get("val")
            if weight_meta is None or bias_meta is None:
                return False
            assert weight_meta.dtype in (
                torch.bfloat16,
                torch.float16,
            )
            if bias_meta.dtype != weight_meta.dtype:
                return False
            return (
                linear_node.args[2] is None
                and bias_meta.dim() == 1
                and bias_meta.size(0) == weight_meta.size(1)
            )

        # convert linear+bias to a single linear for applying fusion path.
        @register_freezing_graph_pattern(
            CallFunction(
                aten.add.Tensor,
                CallFunction(mkldnn._linear_pointwise.default, *_linear_args),
                Arg(),
            ),
            pass_number=2,
            extra_check=is_linear_add_bias,
        )
        def linear_bias_pattern(match, *args):
            graph = match.graph
            add_node = match.output_node()
            linear_node = add_node.args[0]
            new_args = list(linear_node.args)
            new_args[2] = add_node.args[1]
            repl = graph.call_function(
                mkldnn._linear_pointwise.default, tuple(new_args)
            )
            repl.meta.update(add_node.meta)
            add_node.replace_all_uses_with(repl)
            match.erase_nodes()
            counters["inductor"]["mkldnn_linear_bias_matcher_count"] += 1
            counters["inductor"]["mkldnn_linear_bias_matcher_nodes"] += len(match.nodes)

    def _is_packable_mkldnn_rnn_layer(match):
        lstm_node = match.output_node()
        POS_WEIGHTS = [1, 2]
        POS_INPUTS = [0, 5, 6]
        POS_ARGS = POS_WEIGHTS + POS_INPUTS
        # Weights should be Constant
        if any(
            lstm_node.args[POS_WEIGHT].op != "get_attr" for POS_WEIGHT in POS_WEIGHTS
        ):
            return False

        # Meta info for weights and inputs should be available
        if any(lstm_node.args[POS_ARG].meta.get("val") is None for POS_ARG in POS_ARGS):
            return False

        # Check device
        if any(
            lstm_node.args[POS_ARG].meta.get("val").device.type != "cpu"
            for POS_ARG in POS_ARGS
        ):
            return False

        # Check dtype
        if any(
            lstm_node.args[POS_ARG].meta.get("val").dtype == torch.bfloat16
            and not mkldnn._is_mkldnn_bf16_supported()
            for POS_ARG in POS_ARGS
        ):
            return False
        if any(
            lstm_node.args[POS_ARG].meta.get("val").dtype == torch.float16
            and not mkldnn._is_mkldnn_fp16_supported()
            for POS_ARG in POS_ARGS
        ):
            return False

        return True

    def _is_packable_convolution(match):
        """
        Check if the node is supported for MKLDNN convolution.
        """
        conv_node = match.output_node()
        input_meta_value = conv_node.args[0].meta.get("val")
        weight_meta_value = conv_node.args[1].meta.get("val")
        if input_meta_value is None or weight_meta_value is None:
            return False
        input_size = input_meta_value.shape
        if conv_node.args[1].op != "get_attr":
            return False
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type != "cpu"
                or (meta_value.dim() != 4 and meta_value.dim() != 5)
            ):
                return False
        if (
            input_meta_value.dtype == torch.bfloat16
            or weight_meta_value.dtype == torch.bfloat16
        ):
            if not mkldnn._is_mkldnn_bf16_supported():
                return False
        if (
            input_meta_value.dtype == torch.float16
            or weight_meta_value.dtype == torch.float16
        ):
            if not mkldnn._is_mkldnn_fp16_supported():
                return False
        is_transposed = conv_node.args[-3]
        if is_transposed:
            # TODO: Support dynamic shape case for MKLDNN conv transpose.
            if has_free_symbols(input_size):
                return False
            groups = conv_node.args[-1]
            in_channels = weight_meta_value.size(0)
            # doesn't support group_depthwise_conv_transpose.
            if groups > 1 and groups == in_channels:
                return False
            # Port from: aten/src/ATen/native/Convolution.cpp:is_output_padding_big
            output_paddings = conv_node.args[-2]
            strides = conv_node.args[3]
            if any(
                output_padding >= stride
                for output_padding, stride in zip(output_paddings, strides)
            ):
                return False
        return True

    def _is_packable_linear(match):
        """
        Check if the node is supported for MKLDNN linear.
        """

        def is_const_or_cat_by_const(weight):
            if weight.op == "get_attr":
                return True
            if weight.target != aten.cat.default:
                return False
            return all(arg.op == "get_attr" for arg in weight.args[0])

        linear_node = match.output_node()
        # mkldnn linear only supports beta=1or0 and alpha=1
        if linear_node.target == aten.addmm.default:
            alpha = linear_node.kwargs.get("alpha", 1.0)
            beta = linear_node.kwargs.get("beta", 1.0)
            if (beta != 0.0 and beta != 1.0) or alpha != 1.0:
                return False
        # weight_idx is 1 for aten.mm and is 2 for aten.addmm
        weight_idx = 2 if linear_node.target == aten.addmm.default else 1
        if not is_const_or_cat_by_const(linear_node.args[weight_idx]):
            return False
        input_meta_value = linear_node.args[weight_idx - 1].meta.get("val")
        weight_meta_value = linear_node.args[weight_idx].meta.get("val")
        if input_meta_value is None or weight_meta_value is None:
            return False
        batch_size = input_meta_value.shape[0]
        if (
            input_meta_value.dtype == torch.float64
            or weight_meta_value.dtype == torch.float64
        ):
            return False
        is_lp_weight = weight_meta_value.dtype in (
            torch.bfloat16,
            torch.float16,
        )
        # on x86, for fp32, mkl should be enabled and batch_size should not be a free symbol.
        # on aarch64, use mkldnn op for fp32 as well if acl is enabled
        if (
            not is_lp_weight
            and not mkldnn._is_mkldnn_acl_supported()
            and ((not torch._C.has_mkl) or has_free_symbols(batch_size))
        ):
            return False
        for meta_value in [input_meta_value, weight_meta_value]:
            if (
                meta_value is None
                or meta_value.device.type != "cpu"
                or meta_value.dim() != 2
            ):
                return False
        if weight_idx == 2:
            bias_meta_value = linear_node.args[0].meta.get("val")
            if (
                bias_meta_value is None
                or meta_value.device.type != "cpu"
                or bias_meta_value.dim() != 1
                or bias_meta_value.size(0) != weight_meta_value.size(1)
            ):
                return False

        if (
            input_meta_value.dtype == torch.bfloat16
            or weight_meta_value.dtype == torch.bfloat16
        ):
            if not mkldnn._is_mkldnn_bf16_supported():
                return False
        if (
            input_meta_value.dtype == torch.float16
            or weight_meta_value.dtype == torch.float16
        ):
            if not mkldnn._is_mkldnn_fp16_supported():
                return False
        return True

    _aten_conv_args = (
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        Arg(),
        KeywordArg("is_transposed"),
        Arg(),
        Arg(),
    )

    _aten_mkldnn_rnn_layer_args = (
        Arg(),  # input
        Arg(),  # weight0
        Arg(),  # weight1
        Arg(),  # weight2
        Arg(),  # weight3
        Arg(),  # hx_
        Arg(),  # cx_
        KeywordArg("reverse"),  # reverse
        Arg(),  # batch_sizes
        Arg(),  # mode
        Arg(),  # hidden_size
        Arg(),  # num_layers
        Arg(),  # has_biases
        Arg(),  # bidirectional
        Arg(),  # batch_first
        Arg(),  # train
    )

    def _register_weight_pack_pass():
        @register_freezing_graph_pattern(
            CallFunction(aten.convolution.default, *_aten_conv_args),
            extra_check=_is_packable_convolution,
        )
        def convolution(match, *args, **kwargs):
            is_transposed = kwargs.get("is_transposed")
            assert isinstance(is_transposed, bool)
            graph = match.graph
            conv_node = match.output_node()
            input_size = conv_node.args[0].meta.get("val").shape
            with graph.inserting_before(conv_node):
                constant_args = [args[4], args[3], args[5], args[-1]]
                packed_weight_op = mkldnn._reorder_convolution_weight
                packed_conv_op = mkldnn._convolution_pointwise.default
                if is_transposed:
                    constant_args.insert(1, args[-2])  # output_padding
                    packed_weight_op = mkldnn._reorder_convolution_transpose_weight
                    packed_conv_op = mkldnn._convolution_transpose_pointwise.default
                if not has_free_symbols(input_size):
                    packed_weight_inputs = (
                        (args[1],) + tuple(constant_args) + (input_size,)
                    )
                    packed_weight_node = graph.create_node(
                        "call_function", packed_weight_op, args=packed_weight_inputs
                    )
                else:
                    assert not is_transposed
                    # For dynamic shape case, we need to pack weight in runtime.
                    packed_weight_node = args[1]
                packed_conv_inputs = (
                    (args[0], packed_weight_node, args[2])
                    + tuple(constant_args)
                    + ("none", [], "")
                )
                packed_conv_node = graph.create_node(
                    "call_function", packed_conv_op, tuple(packed_conv_inputs)
                )
                conv_node.replace_all_uses_with(packed_conv_node)
                packed_conv_node.meta.update(conv_node.meta)
                graph.erase_node(conv_node)
            counters["inductor"]["mkldnn_conv_weight_pack_matcher_count"] += 1
            counters["inductor"]["mkldnn_conv_weight_pack_matcher_nodes"] += len(
                match.nodes
            )

        @register_freezing_graph_pattern(
            CallFunction(aten.mkldnn_rnn_layer.default, *_aten_mkldnn_rnn_layer_args),
            extra_check=_is_packable_mkldnn_rnn_layer,
        )
        def mkldnn_rnn_layer(match, *args, **kwargs):
            def get_item(graph, node, index):
                return graph.call_function(operator.getitem, (node, index))

            graph = match.graph
            lstm_node = match.output_node()
            input = args[0]
            weight0, weight1 = args[1:3]
            reverse = kwargs.get("reverse")
            packed_lstm_op = aten.mkldnn_rnn_layer.default
            hidden_size = args[9]
            has_biases = args[11]
            batch_first = args[13]
            with graph.inserting_before(lstm_node):
                packed_weight_op = mkldnn._reorder_mkldnn_rnn_layer_weight.default
                packed_weight_inputs = (
                    weight0,
                    weight1,
                    hidden_size,
                    reverse,
                    has_biases,
                    batch_first,
                )
                packed_weight_node = graph.create_node(
                    "call_function", packed_weight_op, packed_weight_inputs, {}, "name"
                )
                packed_weight_items = [
                    get_item(graph, packed_weight_node, i) for i in range(2)
                ]
                pack_lstm_inputs = (
                    args[0],
                    *packed_weight_items,
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                    reverse,
                    *args[7:],
                )

                packed_lstm_node = graph.create_node(
                    "call_function", packed_lstm_op, args=pack_lstm_inputs
                )
                lstm_node.replace_all_uses_with(packed_lstm_node)
                packed_lstm_node.meta.update(lstm_node.meta)
                graph.erase_node(lstm_node)
            counters["inductor"]["mkldnn_rnn_weight_pack_matcher_count"] += 1
            counters["inductor"]["mkldnn_rnn_weight_pack_matcher_nodes"] += len(
                match.nodes
            )

        @register_freezing_graph_pattern(
            CallFunction(
                aten.addmm.default,
                Arg(),
                Arg(),
                Arg(),
                beta=KeywordArg("beta"),
                alpha=KeywordArg("alpha"),
            ),
            extra_check=_is_packable_linear,
            pass_number=1,
        )
        @register_freezing_graph_pattern(
            CallFunction(aten.mm.default, Arg(), Arg()),
            extra_check=_is_packable_linear,
            pass_number=1,
        )
        def linear(match, *args, **kwargs):
            graph = match.graph
            linear_node = match.output_node()
            input = args[0] if linear_node.target == aten.mm.default else args[1]
            bias = (
                None
                if linear_node.target == aten.mm.default
                or (
                    linear_node.target == aten.addmm.default
                    and linear_node.kwargs.get("beta", 1.0) == 0.0
                )
                else args[0]
            )
            weight = args[1] if linear_node.target == aten.mm.default else args[2]
            with graph.inserting_before(linear_node):
                transpose_weight_node = graph.create_node(
                    "call_function", aten.permute.default, (weight, (1, 0))
                )
                weight_dtype = weight.meta.get("val").dtype
                is_lp_weight = weight_dtype in (
                    torch.bfloat16,
                    torch.float16,
                )
                batch_size = input.meta.get("val").shape[0]
                if has_free_symbols(batch_size):
                    assert (
                        is_lp_weight or mkldnn._is_mkldnn_acl_supported()
                    ), f"only bf16/fp16 weight prepacking supports dynamic shape inputs but got {weight_dtype}"
                # For bfloat16 dynamic shape path, using input size hint to pack weight for a better performance.
                packed_weight_inputs = (
                    transpose_weight_node,
                    batch_size.node.shape_env.size_hint(batch_size.node.expr)
                    if has_free_symbols(batch_size)
                    else batch_size,
                )
                # MKL packed matrix can't be copied to a different address because the internal implementation
                # depends on the alignment of internally-stored metadata.
                # In aot mode, we need to firstly save the packed weight, when loading it,
                # it will be in a different address which doesn't work.
                # Disable MKL prepack linear in AOT mode
                packed_weight_op = (
                    mkldnn._reorder_linear_weight
                    if (
                        is_lp_weight
                        or mkldnn._is_mkldnn_acl_supported()
                        or V.aot_compilation is True
                    )
                    else torch.ops.mkl._mkl_reorder_linear_weight
                )
                packed_weight_node = graph.create_node(
                    "call_function", packed_weight_op, args=packed_weight_inputs
                )

                packed_linear_inputs: Tuple[Any, ...] = (input, packed_weight_node)
                if (
                    is_lp_weight
                    or mkldnn._is_mkldnn_acl_supported()
                    or V.aot_compilation is True
                ):
                    packed_linear_inputs += (bias, "none", [], "")
                    packed_linear_op = mkldnn._linear_pointwise.default
                else:
                    packed_linear_inputs += (transpose_weight_node, bias, batch_size)
                    packed_linear_op = torch.ops.mkl._mkl_linear
                packed_linear_node = graph.create_node(
                    "call_function", packed_linear_op, packed_linear_inputs
                )
                linear_node.replace_all_uses_with(packed_linear_node)
                packed_linear_node.meta.update(linear_node.meta)
                graph.erase_node(linear_node)
            counters["inductor"]["mkldnn_linear_weight_pack_matcher_count"] += 1
            counters["inductor"]["mkldnn_linear_weight_pack_matcher_nodes"] += len(
                match.nodes
            )

    def _eliminate_duplicate_packed_nodes(gm):
        """
        Combine packed weight nodes with the same inputs to reduce memory usage.
        for example:
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(32, 32, bias=True)

            def forward(self, x):
                return self.linear(self.linear(x))

        the above's packed weight nodes are duplicate if two linear calls have same input size.
        """
        if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
            return gm

        packed_weight_ops = [
            torch._C._nn.mkldnn_reorder_conv2d_weight,
            torch._C._nn.mkldnn_reorder_conv3d_weight,
            mkldnn._reorder_convolution_transpose_weight,
            mkldnn._reorder_linear_weight,
            mkldnn._reorder_mkldnn_rnn_layer_weight,
        ]
        if torch._C.has_mkl:
            packed_weight_ops.append(torch.ops.mkl._mkl_reorder_linear_weight)

        for node in gm.graph.nodes:
            if node.target in packed_weight_ops and len(node.args[0].users) > 1:
                for user_node in list(node.args[0].users.keys()):
                    if (
                        user_node.target == node.target
                        and user_node != node
                        and user_node.args == node.args
                    ):
                        user_node.replace_all_uses_with(node)
                        gm.graph.erase_node(user_node)

    @functools.lru_cache(None)
    def _mkldnn_fusion_init():
        # TODO: aarch64: enable op fusion for acl once it supports fused operators. Disabling it for now.
        # Otherwise even the matmul or innerproduct can not be accelerated with acl
        if (
            torch.backends.mkldnn.enabled
            and torch.backends.mkldnn.is_available()
            and not torch.ops.mkldnn._is_mkldnn_acl_supported()
        ):
            _register_unary_fusion()
            _register_inplace_fusion()
            _register_binary_unary_fusion()
            # _register_conv_out_stride_fusion()
            _register_binary_fusion()
            # print("_mkldnn_fusion_init")
            # _register_conv_concat_fusion()
            _register_quantization_lowerings()
            _register_woq_lowerings()
            print("_mkldnn_fusion_init")

    @functools.lru_cache(None)
    def _mkldnn_weight_pack_init():
        if torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available():
            _register_weight_pack_pass()
            _recover_linear()
            _register_quantization_weight_pack_pass()
