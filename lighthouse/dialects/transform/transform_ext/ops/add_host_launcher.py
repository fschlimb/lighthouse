from mlir import ir
from mlir.dialects import ext, transform, gpu, arith
from mlir.dialects.transform import DiagnosedSilenceableFailure

from lighthouse.utils.mlir import func_cif
from lighthouse.dialects.transform.transform_ext import TransformExtensionDialect


def _get_dim_str(dim_attr):
    """Extract dimension string ('x', 'y', or 'z') from a gpu.GPU_DimensionAttr."""
    s = str(dim_attr)
    for d in ("x", "y", "z"):
        if d in s:
            return d
    raise ValueError(f"Could not parse dimension from attribute: {dim_attr}")


def _walk_ops(region):
    """Recursively walk all ops in a region and its nested regions."""
    for block in region.blocks:
        for op in block.operations:
            yield op
            for nested_region in op.regions:
                yield from _walk_ops(nested_region)


def _derive_grid_and_block_sizes(gpu_func_op):
    """Derive grid and block (thread) sizes from a gpu.func body.

    Looks for patterns like:
        %bid = gpu.block_id <dim>
        %offset = arith.muli %bid, %tile_const

    Then traces uses of the offset to memref accesses to determine the total
    dimension size. Grid size = total_dim / tile_size.

    Returns (grid_sizes, block_sizes) as dicts mapping 'x','y','z' to int.
    """
    # Check for known_grid_size and known_block_size attributes first.
    attrs = gpu_func_op.attributes
    if "known_grid_size" in attrs and "known_block_size" in attrs:
        grid_attr = attrs["known_grid_size"]
        block_attr = attrs["known_block_size"]
        grid_sizes = {
            "x": int(grid_attr[0]),
            "y": int(grid_attr[1]),
            "z": int(grid_attr[2]),
        }
        block_sizes = {
            "x": int(block_attr[0]),
            "y": int(block_attr[1]),
            "z": int(block_attr[2]),
        }
        return grid_sizes, block_sizes

    # Use known_block_size independently if available (set by
    # xegpu.set_gpu_launch_threads in the schedule before outlining).
    has_known_block_size = "known_block_size" in attrs
    if has_known_block_size:
        block_attr = attrs["known_block_size"]
        known_block_sizes = {
            "x": int(block_attr[0]),
            "y": int(block_attr[1]),
            "z": int(block_attr[2]),
        }

    # Step 1: Find block_id ops and their muli constants.
    # Map: dimension -> (tile_size, muli_result_value)
    block_id_tiles = {}
    block_id_results = {}  # dim -> block_id result value

    for op in _walk_ops(gpu_func_op.body):
        if op.OPERATION_NAME == "gpu.block_id":
            dim = _get_dim_str(op.attributes["dimension"])
            block_id_results[dim] = op.result

    # Find arith.muli ops that multiply a block_id by a constant.
    for op in _walk_ops(gpu_func_op.body):
        if op.OPERATION_NAME == "arith.muli":
            operands = list(op.operands)
            for i, operand in enumerate(operands):
                other = operands[1 - i]
                # Check if this operand is a block_id result
                for dim, bid_val in block_id_results.items():
                    if operand == bid_val:
                        # The other operand should be a constant
                        other_op = other.owner
                        if other_op.OPERATION_NAME == "arith.constant":
                            tile_size = int(other_op.attributes["value"])
                            block_id_tiles[dim] = (tile_size, op.result)
                        break

    if not block_id_tiles:
        raise ValueError(
            "Could not find arith.muli(gpu.block_id, constant) patterns in gpu.func"
        )

    # Step 2: Find the total dimension sizes by tracing muli results to
    # memref accesses. Look for transfer_write or store_nd that uses the
    # muli offsets as indices.
    dim_total_sizes = {}

    for dim, (tile_size, muli_val) in block_id_tiles.items():
        for use in muli_val.uses:
            user_op = use.owner
            op_name = user_op.OPERATION_NAME
            idx_in_op = use.operand_number

            if op_name == "vector.transfer_write":
                # Operands: value, dest_memref, indices...
                # Index position in memref = operand_number - 2
                memref_dim_idx = idx_in_op - 2
                if memref_dim_idx >= 0:
                    dest_memref = user_op.operands[1]
                    memref_type = ir.MemRefType(dest_memref.type)
                    total_size = memref_type.shape[memref_dim_idx]
                    dim_total_sizes[dim] = total_size
                    break
            elif op_name == "vector.transfer_read":
                # Operands: source_memref, indices..., padding
                # For transfer_read: operand 0 is memref, then indices
                memref_dim_idx = idx_in_op - 1
                if memref_dim_idx >= 0:
                    src_memref = user_op.operands[0]
                    memref_type = ir.MemRefType(src_memref.type)
                    if memref_dim_idx < len(memref_type.shape):
                        total_size = memref_type.shape[memref_dim_idx]
                        dim_total_sizes[dim] = total_size
                        break
            elif op_name == "xegpu.load_nd" or op_name == "xegpu.store_nd":
                # For xegpu ops, offsets come after the desc/value operands.
                # load_nd: desc, [offsets...]  → offset index = operand_number - 1
                # store_nd: value, desc, [offsets...] → offset index = operand_number - 2
                if op_name == "xegpu.load_nd":
                    offset_idx = idx_in_op - 1
                    desc_val = user_op.operands[0]
                elif op_name == "xegpu.store_nd":
                    offset_idx = idx_in_op - 2
                    desc_val = user_op.operands[1]
                else:
                    continue
                # Trace desc back to create_nd_tdesc to get memref shape
                if offset_idx >= 0:
                    desc_op = desc_val.owner
                    if desc_op.OPERATION_NAME == "xegpu.create_nd_tdesc":
                        src_memref = desc_op.operands[0]
                        memref_type = ir.MemRefType(src_memref.type)
                        if offset_idx < len(memref_type.shape):
                            total_size = memref_type.shape[offset_idx]
                            dim_total_sizes[dim] = total_size
                            break

    # Compute grid sizes.
    grid_sizes = {}
    for dim, (tile_size, _) in block_id_tiles.items():
        if dim not in dim_total_sizes:
            raise ValueError(
                f"Could not determine total dimension size for block_id_{dim}. "
                f"Tile size is {tile_size} but no memref access was found."
            )
        total = dim_total_sizes[dim]
        assert total % tile_size == 0, (
            f"Dimension size {total} not divisible by tile size {tile_size} for dim {dim}"
        )
        grid_sizes[dim] = total // tile_size

    # Default grid size for dimensions without block_id usage.
    for d in ("x", "y", "z"):
        if d not in grid_sizes:
            grid_sizes[d] = 1

    # Block (thread) sizes: use known_block_size if available (set by
    # xegpu.set_gpu_launch_threads in the schedule, which computes
    # nb_threads = (WG_M // SG_M) * (WG_N // SG_N) * NB_WORKITEMS).
    if has_known_block_size:
        block_sizes = known_block_sizes
    else:
        block_sizes = {"x": 1, "y": 1, "z": 1}
        if "x" in block_id_tiles:
            block_sizes["x"] = block_id_tiles["x"][0]

    return grid_sizes, block_sizes


def _get_gpu_module_and_func_names(gpu_func_op):
    """Get the gpu.module name and gpu.func name from a gpu.func op."""
    func_name = gpu_func_op.attributes["sym_name"].value
    # Navigate up: gpu.func -> gpu.module (parent op)
    gpu_module = gpu_func_op.parent
    module_name = gpu_module.attributes["sym_name"].value
    return module_name, func_name


class AddHostLauncherOp(TransformExtensionDialect.Operation, name="add_host_launcher"):
    """Create a host function that launches the target gpu.func via gpu.launch_func.

    Analyzes the gpu.func body to determine grid dimensions from
    arith.muli(gpu.block_id, constant) patterns. Creates a func.func with
    the same memref arguments that calls gpu.launch_func.
    """

    target: ext.Operand[transform.AnyOpType]
    launcher_func: ext.Result[transform.AnyOpType[()]] = ext.infer_result()

    @classmethod
    def attach_interface_impls(cls, context=None):
        cls.TransformOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)
        cls.MemoryEffectsOpInterfaceModel.attach(cls.OPERATION_NAME, context=context)

    @staticmethod
    def create_host_launcher(
        gpu_func_op, launcher_name: str, block_size: int | None = None
    ):
        """Create a host function that calls gpu.launch_func on the target gpu.func."""
        module_name, func_name = _get_gpu_module_and_func_names(gpu_func_op)
        grid_sizes, block_sizes = _derive_grid_and_block_sizes(gpu_func_op)
        if block_size is not None:
            block_sizes = {"x": block_size, "y": 1, "z": 1}

        # Get the function argument types from the gpu.func.
        entry_block = gpu_func_op.body.blocks[0]
        arg_types = [arg.type for arg in entry_block.arguments]

        index_t = ir.IndexType.get()

        @func_cif(*arg_types, name=launcher_name)
        def launcher(*args):
            # Create grid and block size constants.
            gx = arith.constant(index_t, grid_sizes["x"])
            gy = arith.constant(index_t, grid_sizes["y"])
            gz = arith.constant(index_t, grid_sizes["z"])
            bx = arith.constant(index_t, block_sizes["x"])
            by = arith.constant(index_t, block_sizes["y"])
            bz = arith.constant(index_t, block_sizes["z"])

            gpu.launch_func(
                kernel=[module_name, func_name],
                grid_size=(gx, gy, gz),
                block_size=(bx, by, bz),
                kernel_operands=list(args),
            )

        return launcher.func_op

    class TransformOpInterfaceModel(transform.TransformOpInterface):
        @staticmethod
        def apply(
            op: "AddHostLauncherOp",
            _rewriter: transform.TransformRewriter,
            results: transform.TransformResults,
            state: transform.TransformState,
        ) -> DiagnosedSilenceableFailure:
            targets = state.get_payload_ops(op.target)
            if len(targets) != 1:
                return DiagnosedSilenceableFailure.SilenceableFailure
            if launcher_name_attr := op.attributes.get("launcher_name"):
                launcher_name = launcher_name_attr.value
            else:
                launcher_name = "payload"

            block_size = None
            if block_size_attr := op.attributes.get("block_size"):
                block_size = int(block_size_attr)

            launcher_funcs = []
            for target in targets:
                if target.OPERATION_NAME != "gpu.func":
                    return DiagnosedSilenceableFailure.SilenceableFailure

                # Insert the launcher in the top-level module (after gpu.module).
                gpu_module = target.parent
                with ir.InsertionPoint(gpu_module), target.location:
                    launcher_func = AddHostLauncherOp.create_host_launcher(
                        target, launcher_name, block_size=block_size
                    )
                    launcher_funcs.append(launcher_func)

            results.set_ops(op.launcher_func, launcher_funcs)
            return DiagnosedSilenceableFailure.Success

        @staticmethod
        def allow_repeated_handle_operands(_op: "AddHostLauncherOp") -> bool:
            return False

    class MemoryEffectsOpInterfaceModel(ir.MemoryEffectsOpInterface):
        @staticmethod
        def get_effects(op: "AddHostLauncherOp", effects):
            transform.only_reads_handle(op.op_operands, effects)
            transform.produces_handle(op.results, effects)
            transform.modifies_payload(effects)


def add_host_launcher(
    target: ir.Value[transform.AnyOpType],
    launcher_name: str | None = None,
    block_size: int | None = None,
) -> ir.Value[transform.AnyOpType]:
    """snake_case wrapper to create an AddHostLauncherOp."""
    op = AddHostLauncherOp(target=target)
    if launcher_name is not None:
        op.attributes["launcher_name"] = ir.StringAttr.get(launcher_name)
    if block_size is not None:
        op.attributes["block_size"] = ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), block_size
        )
    return op.launcher_func
