# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build _dsv41_tc_moe_C (SM 8.0) without importing vLLM and without a GPU.

Flags follow csrc/exl3/build_standalone.py at vllm-cmp170hx@18305948ac, plus
-Xptxas -v. TORCH_LIBRARY is not stable ABI, so the .so matches this torch.
"""

import argparse
import os
import shutil
import sys
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "_dsv41_mm_v2try"
SOURCES = ("moe_v2_try.cu",)


def build(out: Path, build_dir: Path, verbose: bool) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get(
        "PATH", ""
    )
    import torch
    from torch.utils import cpp_extension

    print(f"torch {torch.__version__}")
    print(f"torch.version.cuda {torch.version.cuda}")
    print(f"CUDA_HOME {cpp_extension.CUDA_HOME}")
    if torch.version.cuda is None or cpp_extension.CUDA_HOME is None:
        raise RuntimeError("A CUDA-enabled PyTorch and matching CUDA toolkit are required")
    build_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    maps = [
        f"-ffile-prefix-map={HERE}=vllm_exl3/tc_moe",
        f"-ffile-prefix-map={build_dir}=build",
        f"-ffile-prefix-map={Path.home()}=source",
        f"-ffile-prefix-map={Path(torch.__file__).resolve().parent}=torch",
    ]
    defines = ["-DNDEBUG", "-DUSE_CUDA", "-DPy_LIMITED_API=3"]
    cuda_flags = [
        "-O3", "-std=c++20", *defines,
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "-Xptxas=-v",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        *[f"-Xcompiler={flag}" for flag in maps],
    ]
    cpp_extension.load(
        name=NAME,
        sources=[str(HERE / source) for source in SOURCES],
        extra_include_paths=[str(HERE)],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-O3", "-std=gnu++20", *defines, *maps],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=verbose,
        with_cuda=True,
    )
    source = build_dir / f"{NAME}.so"
    destination = out / f"{NAME}.abi3.so"
    with tempfile.NamedTemporaryFile(dir=out, prefix=NAME, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    for suffix in EXTENSION_SUFFIXES:
        obsolete = out / (NAME + suffix)
        if obsolete != destination and obsolete.exists():
            obsolete.unlink()
    plain = out / f"{NAME}.so"
    if plain != destination and plain.exists():
        plain.unlink()
    print(f"{destination} ({destination.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    build(args.out.resolve(), args.build_dir.resolve(), args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
