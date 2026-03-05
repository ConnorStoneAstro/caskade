import torch
import numpy as np


def broadcast_cat_torch(tensors, dim=-1):
    """
    Concatenates tensors with broadcasting.

    It behaves like torch.cat, but first broadcasts the tensors to match
    on all dimensions EXCEPT the concatenation dimension.

    Parameters
    ----------
    tensors : sequence of Tensors
        Tensors to concatenate.
    dim : int
        The dimension along which to concatenate.
        Must be a negative index to ensure consistency across
        tensors of different ranks (e.g., -1 for the last dimension).

    Returns
    -------
    Tensor
        The concatenated tensor.
    """
    if not tensors:
        raise ValueError("tensors argument must be a non-empty sequence")

    # 1. Align Ranks
    # We find the maximum rank (ndim) and pad all smaller tensors with
    # leading singleton dimensions (1s) so everyone aligns to the right.
    max_dim = max(t.ndim for t in tensors)

    # We iterate and unsqueeze; we store them in a new list to avoid mutating inputs
    aligned_tensors = []
    for t in tensors:
        # Prepend 1s until rank matches max_dim
        pads = (1,) * (max_dim - t.ndim)
        if pads:
            # view as (1, 1, ..., *original_shape)
            t = t.view(*pads, *t.shape)
        aligned_tensors.append(t)

    # 2. Normalize dim to positive index based on the new max_dim
    # This allows us to index the shape tuple consistently.
    if dim < 0:
        dim = max_dim + dim

    if dim < 0 or dim >= max_dim:
        raise ValueError(
            f"Dimension {dim - max_dim} is out of bounds for tensors with rank {max_dim}"
        )

    # 3. Determine Broadcast Shape
    # We need a target shape that fits everyone.
    # For the concatenation dimension, we don't care about matching sizes.
    # For all other dimensions, we take the max size (standard broadcasting).
    target_shape = list(aligned_tensors[0].shape)

    for t in aligned_tensors[1:]:
        for i in range(max_dim):
            if i == dim:
                continue  # Skip checks for the concatenation axis

            current_size = t.shape[i]
            target_size = target_shape[i]

            if current_size != target_size:
                if current_size == 1:
                    continue  # This tensor will expand
                elif target_size == 1:
                    target_shape[i] = current_size  # Previous tensors will expand
                else:
                    raise RuntimeError(
                        f"The size of tensor {t.shape} at dimension {i} ({current_size}) "
                        f"does not match the target size ({target_size}) and neither is 1."
                    )

    # 4. Expand and Concatenate
    # We expand every tensor to the target_shape, BUT we must leave the
    # concatenation dimension as-is for that specific tensor.
    expanded_tensors = []
    for t in aligned_tensors:
        # Create the specific target shape for THIS tensor
        # (Global target shape, but with this tensor's specific size at 'dim')
        local_target_shape = list(target_shape)
        local_target_shape[dim] = t.shape[dim]

        # Expand efficienty (returns a view, no data copy)
        expanded_tensors.append(t.expand(*local_target_shape))

    return torch.cat(expanded_tensors, dim=dim)


def broadcast_cat_jax(arrays, dim=-1):
    """
    Concatenates JAX arrays with broadcasting.

    Behaves like jnp.concatenate, but first broadcasts the arrays to match
    on all dimensions EXCEPT the concatenation dimension.

    Parameters
    ----------
    arrays : sequence of jnp.ndarray
        Arrays to concatenate.
    dim : int
        The dimension along which to concatenate.

    Returns
    -------
    jnp.ndarray
        The concatenated array.
    """
    import jax.numpy as jnp

    if not arrays:
        raise ValueError("arrays argument must be a non-empty sequence")

    # 1. Align Ranks
    # Find the maximum rank and left-pad smaller arrays with shape 1
    max_ndim = max(a.ndim for a in arrays)

    aligned_arrays = []
    for a in arrays:
        diff = max_ndim - a.ndim
        if diff > 0:
            # Reshape to (1, 1, ..., *original_shape)
            new_shape = (1,) * diff + a.shape
            aligned_arrays.append(a.reshape(new_shape))
        else:
            aligned_arrays.append(a)

    # 2. Normalize dim
    # Convert negative index to positive based on max_ndim
    if dim < 0:
        dim += max_ndim

    if dim < 0 or dim >= max_ndim:
        raise ValueError(f"Dimension {dim - max_ndim} is out of bounds")

    # 3. Determine Broadcast Shapes
    # We split the shapes into two parts: dimensions *before* the concat axis
    # and dimensions *after* the concat axis.
    shapes_pre = [a.shape[:dim] for a in aligned_arrays]
    shapes_post = [a.shape[dim + 1 :] for a in aligned_arrays]

    try:
        # Calculate the common broadcast shape for the surrounding dimensions
        common_pre = jnp.broadcast_shapes(*shapes_pre)
        common_post = jnp.broadcast_shapes(*shapes_post)
    except ValueError as e:
        raise ValueError("Shapes cannot be broadcast (excluding concatenation axis)") from e

    # 4. Expand and Concatenate
    expanded_arrays = []
    for a in aligned_arrays:
        # Construct the target shape for THIS array:
        # [Broadcasting Pre] + [Original Dim Size] + [Broadcasting Post]
        target_shape = common_pre + (a.shape[dim],) + common_post

        # broadcast_to returns a view (no copy) where possible
        expanded_arrays.append(jnp.broadcast_to(a, target_shape))

    return jnp.concatenate(expanded_arrays, axis=dim)


def broadcast_cat_numpy(arrays, dim=-1):
    """
    Concatenates NumPy arrays with broadcasting.

    Behaves like np.concatenate, but first broadcasts the arrays to match
    on all dimensions EXCEPT the concatenation dimension.

    Parameters
    ----------
    arrays : sequence of np.ndarray
        Arrays to concatenate.
    dim : int
        The dimension along which to concatenate.

    Returns
    -------
    np.ndarray
        The concatenated array.
    """
    if not arrays:
        raise ValueError("arrays argument must be a non-empty sequence")

    # Ensure inputs are actually numpy arrays (handles lists/tuples of numbers)
    arrays = [np.asarray(a) for a in arrays]

    # 1. Align Ranks
    # Find the maximum rank and left-pad smaller arrays with shape 1
    max_ndim = max(a.ndim for a in arrays)

    aligned_arrays = []
    for a in arrays:
        diff = max_ndim - a.ndim
        if diff > 0:
            # Reshape to (1, 1, ..., *original_shape)
            new_shape = (1,) * diff + a.shape
            aligned_arrays.append(a.reshape(new_shape))
        else:
            aligned_arrays.append(a)

    # 2. Normalize dim
    # Convert negative index to positive based on max_ndim
    if dim < 0:
        dim += max_ndim

    if dim < 0 or dim >= max_ndim:
        raise ValueError(f"Dimension {dim - max_ndim} is out of bounds")

    # 3. Determine Broadcast Shapes
    # We split the shapes into two parts: dimensions *before* the concat axis
    # and dimensions *after* the concat axis.
    shapes_pre = [a.shape[:dim] for a in aligned_arrays]
    shapes_post = [a.shape[dim + 1 :] for a in aligned_arrays]

    try:
        # Calculate the common broadcast shape for the surrounding dimensions
        # Note: np.broadcast_shapes was added in NumPy 1.20
        common_pre = np.broadcast_shapes(*shapes_pre)
        common_post = np.broadcast_shapes(*shapes_post)
    except ValueError as e:
        raise ValueError("Shapes cannot be broadcast (excluding concatenation axis)") from e

    # 4. Expand and Concatenate
    expanded_arrays = []
    for a in aligned_arrays:
        # Construct the target shape for THIS array:
        # [Broadcasting Pre] + [Original Dim Size] + [Broadcasting Post]
        target_shape = common_pre + (a.shape[dim],) + common_post

        # broadcast_to returns a read-only view (efficient memory usage)
        expanded_arrays.append(np.broadcast_to(a, target_shape))

    return np.concatenate(expanded_arrays, axis=dim)
