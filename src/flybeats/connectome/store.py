"""Where the slice and its controls live under work_dir, and how later phases load them."""
import pyarrow as pa
import pyarrow.parquet as pq


def slice_paths(work_dir):
    d = work_dir / "slice"
    d.mkdir(parents=True, exist_ok=True)
    return d / "slice_neurons.parquet", d / "slice_edges.parquet"


def control_path(work_dir, seed):
    d = work_dir / "slice" / "controls"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"control_seed{seed}_edges.parquet"


def save_table(df, path):
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def load_slice(work_dir, control_seed=None):
    """(neurons, edges). control_seed=None is the real fly; 1-5 load a random control's edges instead."""
    neurons_path, edges_path = slice_paths(work_dir)
    if control_seed is not None:
        edges_path = control_path(work_dir, control_seed)
    return pq.read_table(neurons_path).to_pandas(), pq.read_table(edges_path).to_pandas()
