import os
from multiprocessing import cpu_count, Pool

# This has to be set before jax is imported, but we should think of a better way to set it
os.environ["XLA_FLAGS"]= f"--xla_force_host_platform_device_count={max(cpu_count()-2, 1)}" 
os.environ["JAX_PLATFORMS"]='cpu'

import jax
import jax.numpy as jnp
jax.config.update("jax_platforms", "cpu")
jax.config.update('jax_platform_name', 'cpu') # This will be deprecated 


from functools import partial
import numpy as np 
from kasearch.canonical_alignment import all_cdrs_mask, cdr3_mask, reg_def

default_region_masks = np.stack([np.ones(200, dtype = np.uint8), all_cdrs_mask, cdr3_mask])
default_length_matched = np.array([False,True,True], dtype = bool)


@partial(jax.jit, static_argnums=1)
def chunk(array, chunks):
    old_size = array.shape[0]
    if old_size % chunks != 0:
        chunk_size = old_size//chunks + 1
        array = jnp.pad(array, ((0,chunks*(chunk_size) - old_size),(0,0)), constant_values=0)
    else:
        chunk_size = old_size//chunks
    return array.reshape(chunks, chunk_size, array.shape[-1])


@jax.jit
def _calculate_single_sequence_identity_average(abs1, abs2, masks, length_matched):
    comparison = abs1 == abs2

    mask1, mask2 = (abs1 != 0) & (abs1 != 124), (abs2 != 0) & (abs2 != 124)
    overlapping_residues = comparison * mask1 * mask2

    len1, len2 = mask1 @ masks, mask2 @ masks
    region_overlap = overlapping_residues @ masks
    identities = ((region_overlap / len1) + (region_overlap / len2)) * (~length_matched | (len1==len2)) / 2
    return identities


@jax.jit
def _calculate_single_sequence_identity_with_ends(abs1, abs2, masks, length_matched):
    comparison = abs1 == abs2

    mask1, mask2 = (abs1 != 0) & (abs1 != 124), (abs2 != 0) & (abs2 != 124)    
    overlapping_residues = comparison * mask1 * mask2

    length = (mask1 | mask2) @ masks
    region_overlap = overlapping_residues @ masks
    identities = (region_overlap / length) * (~length_matched | ((mask1 @ masks) == (mask2 @ masks)))
    return identities


@jax.jit
def _calculate_single_sequence_identity_without_ends(abs1, abs2, masks, length_matched):
    comparison = abs1 == abs2

    mask1, mask2 = abs1 != 0, abs2 != 0
    not_missing1, not_missing2 = abs1 != 124, abs2 != 124
    exists1, exists2 = mask1 * not_missing1, mask2 * not_missing2
    
    overlapping_residues = comparison * exists1 * exists2

    length = (mask1 | mask2) * (not_missing1 * not_missing2) @ masks
    region_overlap = overlapping_residues @ masks
    identities = (region_overlap / length) * (~length_matched | ((exists1 @ masks) == (exists2 @ masks)))
    return identities


@jax.jit
def _calculate_single_sequence_identity(abs1, abs2, masks, length_matched, include_ends):
    out = jax.lax.cond(
        include_ends, _calculate_single_sequence_identity_with_ends,
        _calculate_single_sequence_identity_without_ends, abs1, abs2,
        masks, length_matched)
    return out


@jax.jit
def _calculate_chunked_sequence_identity(abs1, abs2, masks, length_matched, include_ends):
    
    outs = [_calculate_single_sequence_identity(abs1, ab2, masks, length_matched, include_ends) for ab2 in abs2]
    
    return jax.lax.transpose(jnp.stack(outs, axis = 2),(0,2,1))


calculate_many_sequence_identities = jax.pmap(_calculate_chunked_sequence_identity, in_axes=(0,None,None,None,None))


def calculate_seq_ids_multiquery(array_of_abs1, array_of_abs2, region_masks, length_matched, include_ends):
    masks, length_matched, include_ends = jnp.array(region_masks.T), jnp.array(length_matched), jnp.array(include_ends)
    
    abs1 = chunk(jax.lax.stop_gradient(array_of_abs1), jax.device_count())
    abs2 = jax.lax.stop_gradient(array_of_abs2)

    identities = np.array(calculate_many_sequence_identities(abs1, abs2, masks, length_matched, include_ends))
    
    return identities.reshape(-1, array_of_abs2.shape[0], len(region_masks))[:array_of_abs1.shape[0]]   


def get_n_most_identical_multiquery(
    query, targets, target_ids, n=10,
    region_masks=default_region_masks,
    length_matched=default_length_matched,
    include_ends=True,
):
    
    seq_identity_matrix = calculate_seq_ids_multiquery(targets, query, region_masks, length_matched, include_ends)
    seq_identity_matrix[np.isnan(seq_identity_matrix)] = 0

    position_of_n_best = np.argpartition(-seq_identity_matrix, n-1, axis=0)  # partition by seq_id

    n_highest_identities = np.take_along_axis(seq_identity_matrix, position_of_n_best, axis=0)[:n]

    broadcasted_ids = np.broadcast_to(target_ids[:, None, None], (targets.shape[0], query.shape[0], region_masks.shape[0], 2))

    n_highest_ids = np.take_along_axis(broadcasted_ids, position_of_n_best[:, :, :, None], axis=0)[:n]

    return n_highest_identities.transpose((1,0,2)), n_highest_ids.transpose((1,0,2,3))


def slow_calculate_seq_id(ab1, ab2):
    # Around 1k times slower but doesn't rely on canonical alignment
    ab1_clean = [x for x in ab1 if x[1] != "-"]
    ab2_clean = [x for x in ab2 if x[1] != "-"]

    overlapping_residues = [x for x in ab1_clean if x in ab2_clean]

    # For the whole sequence:
    full_overlap = len(overlapping_residues)
    full_id = (full_overlap / len(ab1_clean) + full_overlap / len(ab2_clean)) / 2

    # For all CDRs
    cdrs_overlap = len([x for x in overlapping_residues if x[0][0] in reg_def["CDR_all"]])
    cdrs_len1 = len([x for x in ab1_clean if x[0][0] in reg_def["CDR_all"]])
    cdrs_len2 = len([x for x in ab2_clean if x[0][0] in reg_def["CDR_all"]])

    if cdrs_len1 and cdrs_len2:
        cdrs_id = (cdrs_overlap / cdrs_len1 + cdrs_overlap / cdrs_len2) / 2
    else:
        cdrs_id = 0

    # For CDR-H3
    h3_overlap = len([x for x in overlapping_residues if x[0][0] in reg_def["CDR3"]])
    h3_len1 = len([x for x in ab1_clean if x[0][0] in reg_def["CDR3"]])
    h3_len2 = len([x for x in ab2_clean if x[0][0] in reg_def["CDR3"]])

    if h3_len1 and h3_len2:
        h3_id = (h3_overlap / h3_len1 + h3_overlap / h3_len2) / 2
    else:
        h3_id = 0

    return full_id, cdrs_id, h3_id


def slow_calculate_many_seq_ids(ab1, list_of_abs, n_jobs=1):
    size = len(list_of_abs)

    with Pool(processes=n_jobs) as pool:
        return pool.starmap(slow_calculate_seq_id, zip(size * [ab1], list_of_abs), chunksize=size // n_jobs)


def slow_get_n_most_identical(query, target, target_ids, n=10, n_jobs=None):
    n_jobs = n_jobs if n_jobs is not None else os.cpu_count() - 1
    seq_identity_matrix = np.array(slow_calculate_many_seq_ids(query, target, n_jobs=n_jobs))

    position_of_n_best = np.argpartition(-seq_identity_matrix, n, axis=0)  # partition by seq_id
    n_highest_identities = np.take_along_axis(seq_identity_matrix, position_of_n_best, axis=0)[:n]

    broadcasted_ids = np.broadcast_to(target_ids[:, None], (len(target), 3, 2))
    n_highest_ids = np.take_along_axis(broadcasted_ids, position_of_n_best[:, :, None], axis=0)[:n]

    return n_highest_identities, n_highest_ids
