from matmul import XeGPUMatMul
from mlir import ir

from lighthouse.workload import benchmark

from time import perf_counter
import multiprocessing
from multiprocessing.sharedctypes import Value
from ctypes import c_double
from datetime import timedelta
from itertools import product
import numpy
import logging
import csv
import os


def run_experiment(
    ab_type="f16",
    c_type="f32",
    nruns=100,
    nwarmup=20,
    check_result=False,
    has_relu=False,
    **params,
):
    M = params.pop("M")
    N = params.pop("N")
    K = params.pop("K")

    with ir.Context(), ir.Location.unknown():
        wload = XeGPUMatMul(
            M=M,
            N=N,
            K=K,
            ab_type=ab_type,
            c_type=c_type,
            has_bias=False,
            has_relu=has_relu,
        )

        times = benchmark(
            wload,
            nruns=nruns,
            nwarmup=nwarmup,
            schedule_parameters=params,
            check_correctness=check_result,
            verbose=1,
        )
        times *= 1e6  # convert to microseconds
        elapsed = numpy.mean(times)
        flop_count = wload.get_complexity()[0]
        gflops = flop_count / (elapsed * 1e-6) / 1e9

        return elapsed, gflops


class CSVLogger:
    def __init__(self, filename=None):
        self.filename = filename
        self.header_written = False
        self.fieldnames = None
        self.logger = logging.getLogger("csv_logger")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        if self.filename is not None:
            assert not os.path.exists(self.filename), "CSV file already exist"

    def log(self, data: dict):
        if self.fieldnames is None:
            self.fieldnames = list(data.keys())
        write_header = not os.path.exists(self.filename) or not self.header_written
        if write_header:
            self.logger.info(",".join(self.fieldnames))
        self.logger.info(",".join(str(data[k]) for k in self.fieldnames))
        if self.filename is None:
            return
        with open(self.filename, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(data)


def check_constraints(params, verbose=False):
    def print_reason(msg):
        if verbose:
            print(f"  Invalid: {msg}")

    # hardware constraints
    max_nb_sg_threads = 32
    dpas_tile = [8, 16, 16]
    load_max_rows = 32

    # heuristics
    small_load_tile_elems = 16 * 16  # skip smaller load tiles
    max_nb_unrolled_dpas_ops = 32

    M = params["M"]
    N = params["N"]
    wg_tile_m = params["auto_wg_d0"]
    wg_tile_n = params["auto_wg_d1"]
    sg_tile_m = params["auto_sg_d0"]
    sg_tile_n = params["auto_sg_d1"]
    load_tile_a_m = params["auto_load_a_d0"]
    load_tile_a_k = params["auto_load_a_d1"]
    load_tile_b_k = params["auto_load_b_d0"]
    load_tile_b_n = params["auto_load_b_d1"]
    prefetch_tile_a_m = params["auto_prefetch_a_d0"]
    prefetch_tile_a_k = params["auto_prefetch_a_d1"]
    prefetch_tile_b_k = params["auto_prefetch_b_d0"]
    prefetch_tile_b_n = params["auto_prefetch_b_d1"]
    k_tile = params["auto_k"]

    if M % wg_tile_m != 0:
        print_reason("wg_tile_m does not divide M")
        return False
    if N % wg_tile_n != 0:
        print_reason("wg_tile_n does not divide N")
        return False
    if wg_tile_m % sg_tile_m != 0:
        print_reason("sg_tile_m does not divide wg_tile_m")
        return False
    if wg_tile_n % sg_tile_n != 0:
        print_reason("sg_tile_n does not divide wg_tile_n")
        return False
    # smaller than child tile (DPAS tile sizes are fixed)
    if sg_tile_n % dpas_tile[0] != 0:
        print_reason("sg_tile_n not multiple of dpas_n")
        return False
    if sg_tile_m % dpas_tile[1] != 0:
        print_reason("sg_tile_m not multiple of dpas_m")
        return False
    if k_tile % dpas_tile[2] != 0:
        print_reason("k_tile not multiple of dpas_k")
        return False

    # SG level layout: [nb_sg_threads_m, nb_sg_threads_n]
    nb_sg_threads_m = wg_tile_m // sg_tile_m
    nb_sg_threads_n = wg_tile_n // sg_tile_n
    nb_sg_threads = nb_sg_threads_m * nb_sg_threads_n
    if nb_sg_threads > max_nb_sg_threads:
        # too many SG threads
        print_reason("too many sg threads")
        return False

    if sg_tile_m % load_tile_a_m != 0:
        print_reason("load_tile_a_m does not divide sg_tile_m")
        return False
    if k_tile % load_tile_a_k != 0:
        print_reason("load_tile_a_k does not divide k_tile")
        return False
    if k_tile % load_tile_b_k != 0:
        print_reason("load_tile_b_k does not divide k_tile")
        return False
    if sg_tile_n % load_tile_b_n != 0:
        print_reason("load_tile_b_n does not divide sg_tile_n")
        return False
    if load_tile_a_m > load_max_rows:
        print_reason("too large load_tile_a_m")
        return False
    if load_tile_a_k > load_max_rows:
        print_reason("too large load_tile_a_k")
        return False
    if load_tile_b_k > load_max_rows:
        print_reason("too large load_tile_b_k")
        return False
    if load_tile_b_n > load_max_rows:
        print_reason("too large load_tile_b_n")
        return False
    if sg_tile_m % prefetch_tile_a_m != 0:
        print_reason("prefetch_tile_a_m does not divide sg_tile_m")
        return False
    if k_tile % prefetch_tile_a_k != 0:
        print_reason("prefetch_tile_a_k does not divide k_tile")
        return False
    if k_tile % prefetch_tile_b_k != 0:
        print_reason("prefetch_tile_b_k does not divide k_tile")
        return False
    if sg_tile_n % prefetch_tile_b_n != 0:
        print_reason("prefetch_tile_b_n does not divide sg_tile_n")
        return False
    if prefetch_tile_a_m > load_max_rows:
        print_reason("too large prefetch_tile_a_m")
        return False
    if prefetch_tile_a_k > load_max_rows:
        print_reason("too large prefetch_tile_a_k")
        return False
    if prefetch_tile_b_k > load_max_rows:
        print_reason("too large prefetch_tile_b_k")
        return False
    if prefetch_tile_b_n > load_max_rows:
        print_reason("too large prefetch_tile_b_n")
        return False
    if load_tile_a_m % dpas_tile[0] != 0:
        print_reason("load_tile_a_m not multiple of dpas_m")
        return False
    if load_tile_a_k % dpas_tile[2] != 0:
        print_reason("load_tile_a_k not multiple of dpas_k")
        return False
    if load_tile_b_k % dpas_tile[2] != 0:
        print_reason("load_tile_b_k not multiple of dpas_k")
        return False
    if load_tile_b_n % dpas_tile[1] != 0:
        print_reason("load_tile_b_n not multiple of dpas_n")
        return False

    if load_tile_a_m * load_tile_a_k < small_load_tile_elems:
        # skip small load tiles
        print_reason("too small load_tile_a")
        return False
    if load_tile_b_k * load_tile_b_n < small_load_tile_elems:
        # skip small load tiles
        print_reason("too small load_tile_b")
        return False
    if load_tile_a_m > load_tile_a_k:
        # skip tall and skinny load tiles
        print_reason("invalid load_tile_a shape")
        return False
    if load_tile_b_k > load_tile_b_n:
        # skip tall and skinny load tiles
        print_reason("invalid load_tile_b shape")
        return False

    nb_load_b_n = load_tile_b_n // dpas_tile[1]
    if nb_load_b_n > 1:
        # unsupported VNNI layout, loaded tile can only be row-sliced for vnni
        # NOTE this can plausibly be relaxed
        print_reason("invalid load_tile_b_n for VNNI")
        return False

    # prefetch A layout
    nb_prefetch_a_m = sg_tile_m // prefetch_tile_a_m
    nb_prefetch_a_k = k_tile // prefetch_tile_a_k
    if nb_prefetch_a_m * nb_prefetch_a_k > max_nb_sg_threads:
        # too many prefetch tiles
        print_reason("too many prefetch A tiles")
        return False

    if prefetch_tile_a_m * prefetch_tile_a_k < small_load_tile_elems:
        # skip small prefetch tiles
        print_reason("too small prefetch A tile")
        return False
    if prefetch_tile_a_m > prefetch_tile_a_k:
        # skip column tiles
        print_reason("invalid prefetch_tile_a shape")
        return False

    # prefetch B layout
    nb_prefetch_b_k = k_tile // prefetch_tile_b_k
    nb_prefetch_b_n = sg_tile_n // prefetch_tile_b_n
    if nb_prefetch_b_k * nb_prefetch_b_n > max_nb_sg_threads:
        # too many prefetch tiles
        print_reason("too many prefetch B tiles")
        return False
    if prefetch_tile_b_k * prefetch_tile_b_n < small_load_tile_elems:
        # skip small prefetch tiles
        print_reason("too small prefetch B tile")
        return False
    if prefetch_tile_b_k > prefetch_tile_b_n:
        # skip column tiles
        print_reason("invalid prefetch_tile_b shape")
        return False

    # estimate register usage
    nb_dpas_m = sg_tile_m // dpas_tile[0]
    nb_dpas_n = sg_tile_n // dpas_tile[1]
    nb_dpas_k = k_tile // dpas_tile[2]
    # number of unrolled dpas ops: nb_dpas_m * nb_dpas_n * nb_dpas_k
    if nb_dpas_m * nb_dpas_n * nb_dpas_k > max_nb_unrolled_dpas_ops:
        print_reason("too many unrolled dpas ops")
        return False

    return True


def run_with_timeout(*args, timeout=10, **kwargs):
    """
    Wrapper to execute the experiment with a new thread and a timeout.

    Experiments must be run in a new process to ensure reliable timings.
    Otherwise, IGC compiler may not be invoked.

    Sends kill signal if timeout is reached.
    """
    # wrap return values
    timing = Value(c_double, 0.0)
    gflops = Value(c_double, 0.0)

    def wrapped(timing, gflops, *args, **kwargs):
        res = run_experiment(*args, **kwargs)
        timing.value = res[0]
        gflops.value = res[1]

    all_args = tuple([timing, gflops] + list(args))
    proc = multiprocessing.Process(target=wrapped, args=all_args, kwargs=kwargs)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        print("TIMEOUT")
        proc.kill()
        proc.join()
        return 0, 0
    proc.close()
    return timing.value, gflops.value


def run(
    csv_logger,
    params,
    ab_type,
    c_type,
    add_bias,
    add_relu,
):
    entry = {
        "M": params["M"],
        "N": params["N"],
        "K": params["K"],
        "wg_m": params["auto_wg_d0"],
        "wg_n": params["auto_wg_d1"],
        "sg_m": params["auto_sg_d0"],
        "sg_n": params["auto_sg_d1"],
        "k": params["auto_k"],
        "load_a_m": params["auto_load_a_d0"],
        "load_a_k": params["auto_load_a_d1"],
        "load_b_k": params["auto_load_b_d0"],
        "load_b_n": params["auto_load_b_d1"],
        "pf_a_m": params["auto_prefetch_a_d0"],
        "pf_a_k": params["auto_prefetch_a_d1"],
        "pf_b_k": params["auto_prefetch_b_d0"],
        "pf_b_n": params["auto_prefetch_b_d1"],
    }

    try:
        tic = perf_counter()
        elapsed, gflops = run_with_timeout(
            ab_type=ab_type,
            c_type=c_type,
            nruns=nruns,
            check_result=check_result,
            has_bias=add_bias,
            has_relu=add_relu,
            **params,
        )
        duration = perf_counter() - tic
        entry["time (ms)"] = elapsed
        entry["GFLOPS/s"] = gflops
        csv_logger.log(entry)
        duration_str = f"Duration: {duration:.3f} s"
        print(duration_str)
    except Exception as e:
        print("FAILED")
        print(entry)
        print(f"  Error: {e}")


def get_divisors(n, min_tile=32, max_tile=256):
    p = numpy.ceil(n / max_tile)
    q = n // min_tile

    candidates = n / numpy.arange(max(p, 1), q + 1)
    candidates = [int(v) for v in candidates if v - numpy.round(v) == 0]
    return candidates[::-1]


def divisible_by(a_list, b):
    return [a for a in a_list if a % b == 0]


# --------------------
#  driver
# --------------------

# run loop but do not execute experiments
dry_run = False

# fixed parameters
sizes = [4096, 4096, 4096]
add_bias = False
add_relu = False
M, N, K = sizes
ab_type = "f16"
c_type = "f32"
verbose = False
check_result = True
dump_kernel = False
nruns = 200
dpas_tile = [8, 16, 16]

# hardware constraints
max_nb_sg_threads = 32

# heuristics
small_load_tile_elems = 16 * 16  # skip smaller load tiles

# env
os.environ["NEO_CACHE_PERSISTENT"] = "0"  # disable compiler cache

csv_file = "out_gridsearch.csv"
csv_logger = CSVLogger(csv_file)

wg_tiles_m = divisible_by(get_divisors(M, 64, 256), dpas_tile[0])
wg_tiles_n = divisible_by(get_divisors(N, 64, 256), dpas_tile[1])
sg_tiles_m = divisible_by(get_divisors(M, 32, 100), dpas_tile[0])
sg_tiles_n = divisible_by(get_divisors(N, 32, 100), dpas_tile[1])
k_tiles = divisible_by(get_divisors(K, 16, 50), dpas_tile[2])
load_tiles = [8, 16, 32]
nb_prefetch = 1

print(f"Matmul problem size: {sizes}")
print(f"{ab_type=}")
print(f"{c_type=}")
print(f"{add_bias=}")
print(f"{add_relu=}")

print(f"{wg_tiles_m=}")
print(f"{wg_tiles_n=}")
print(f"{sg_tiles_m=}")
print(f"{sg_tiles_n=}")
print(f"{k_tiles=}")
print(f"{load_tiles=}")

iterables = [
    wg_tiles_m,  # WG n
    wg_tiles_n,  # WG m
    sg_tiles_m,  # SG n
    sg_tiles_n,  # SG m
    k_tiles,  # reduction k
    load_tiles,  # load tile A n
    load_tiles,  # load tile A k
    load_tiles,  # load tile B k
    load_tiles,  # load tile B m
    load_tiles,  # prefetch tile A n
    load_tiles,  # prefetch tile A k
    load_tiles,  # prefetch tile B k
    load_tiles,  # prefetch tile B m
]
total_complexity = numpy.prod([len(x) for x in iterables])
print(f"Total complexity: {total_complexity} configurations")

i = 0
tic = perf_counter()
for (
    wg_tile_m,
    wg_tile_n,
    sg_tile_m,
    sg_tile_n,
    k_tile,
    load_tile_a_m,
    load_tile_a_k,
    load_tile_b_k,
    load_tile_b_n,
    prefetch_tile_a_m,
    prefetch_tile_a_k,
    prefetch_tile_b_k,
    prefetch_tile_b_n,
) in product(*iterables):
    params = {
        "M": M,
        "N": N,
        "K": K,
        "auto_wg_d0": wg_tile_m,
        "auto_wg_d1": wg_tile_n,
        "auto_sg_d0": sg_tile_m,
        "auto_sg_d1": sg_tile_n,
        "auto_k": k_tile,
        "auto_load_a_d0": load_tile_a_m,
        "auto_load_a_d1": load_tile_a_k,
        "auto_load_b_d0": load_tile_b_k,
        "auto_load_b_d1": load_tile_b_n,
        "auto_prefetch_a_d0": prefetch_tile_a_m,
        "auto_prefetch_a_d1": prefetch_tile_a_k,
        "auto_prefetch_b_d0": prefetch_tile_b_k,
        "auto_prefetch_b_d1": prefetch_tile_b_n,
        "auto_nb_prefetch": nb_prefetch,
    }

    if not check_constraints(params, verbose=False):
        continue
    i += 1
    if dry_run:
        continue
    run(
        csv_logger,
        params,
        ab_type,
        c_type,
        add_bias,
        add_relu,
    )

duration = perf_counter() - tic
print(f"Number of executed configurations: {i}")
print(f"Total duration: {timedelta(seconds=duration)}")
