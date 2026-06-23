# gdrnpp Build Troubleshooting

Fixes for the common build / import errors hit while compiling gdrnpp's native
extensions (`fps`, `flow`, `ransac_voting`, `torch_nndistance`, `egl_renderer`,
`uncertainty_pnp`) and `detectron2`.

## Verified environment

- GPU: NVIDIA **A10G** (Ampere, compute capability `sm_86`); driver 580.x
- Conda env: `gdrnpp_env`, Python 3.10
- CUDA toolkit (inside the env): **11.8** (`$CONDA_PREFIX/bin/nvcc`)
- PyTorch: **2.4.1+cu118**
- System Ceres: **2.2.0**; Eigen3 + glog dev headers installed

## Issue map (TL;DR)

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | Wall of Eigen `-Wignored-attributes` output | Warnings, not errors | Ignore (optionally `-Wno-ignored-attributes`) |
| 2 | `import _ext` -> `undefined symbol: ceres::Problem::AddResidualBlock(...)` | Bundled Ceres 1.x headers vs system Ceres 2.2 lib | Build against system Ceres 2.2 + Eigen, `-std=c++17`, link `ceres`/`glog`/`absl` |
| 3 | `OSError: CUDA_HOME environment variable is not set` | CPU-only PyTorch (`torch.version.cuda == None`) | Install CUDA build `torch 2.4.1+cu118` |
| 4 | torch install: `No such file ... pillow-...dist-info/METADATA` | Corrupted Pillow install | Remove remnants + reinstall Pillow |
| 5 | `import detectron2._C` -> `undefined symbol: at::_ops::zeros_like::call` | `_C` built against a different torch (ABI mismatch) | Rebuild detectron2 against current torch |

Golden rule: **whenever you change the PyTorch version, rebuild every C++/CUDA
extension** (all of `core/csrc/*` and detectron2), or you get `undefined symbol`
errors at import.

---

## 1. Eigen / Ceres compiler warnings are NOT errors

Symptom: thousands of lines like

    warning: ignoring attributes on template argument '__m128' [-Wignored-attributes]

Cause: harmless interaction between old Eigen and GCC SIMD types inside Ceres
templates. The build still succeeds. Confirm by the artifact, not the warnings:

    ls -la *.so      # the .so being present means it compiled

Optional silence: add `-Wno-ignored-attributes` to the compile flags.

---

## 2. uncertainty_pnp: undefined symbol ceres::Problem::AddResidualBlock

Symptom:

    ImportError: .../uncertainty_pnp/_ext....so:
    undefined symbol: _ZN5ceres7Problem16AddResidualBlockEPNS_12CostFunctionEPNS_12LossFunctionEPd
    # demangled: ceres::Problem::AddResidualBlock(ceres::CostFunction*, ceres::LossFunction*, double*)

Cause: that non-template signature is the Ceres 1.x API. In Ceres 2.x it became a
variadic template, so the symbol is absent from libceres.so.4 (Ceres 2.2). The
code compiled against the repo's bundled Ceres 1.x headers but the runtime lib is
system Ceres 2.2 -> mismatch.

Diagnose:

    ldd _ext*.so | grep -iE "ceres|glog"             # shows libceres.so.4 (=2.2)
    grep -E "define CERES_VERSION_(MAJOR|MINOR)" /usr/include/ceres/version.h

Fix: build against the system Ceres 2.2 + Eigen (headers AND lib matched), with
`-std=c++17` (required by Ceres 2.x), linking ceres + glog + abseil. No source
edits needed. Replace `core/csrc/uncertainty_pnp/setup.py` with:

    import os
    import glob
    import re

    ceres_include = "/usr/include"
    eigen_include = "/usr/include/eigen3"
    lib_dir = "/usr/lib/x86_64-linux-gnu"

    # Compile against SYSTEM Ceres 2.2 + Eigen. Ceres 2.x requires C++17.
    ret = os.system(
        "g++ -c src/uncertainty_pnp.cpp -o src/uncertainty_pnp.cpp.o "
        "-fopenmp -fPIC -O2 -std=c++17 -I {} -I {}".format(ceres_include, eigen_include)
    )
    assert ret == 0, "compilation of uncertainty_pnp.cpp failed"

    from cffi import FFI

    ffibuilder = FFI()

    with open(os.path.join(os.path.dirname(__file__), "src/ext.h")) as f:
        ffibuilder.cdef(f.read())

    # Ceres 2.2 depends on abseil + glog; link them so header-instantiated
    # symbols resolve at import time.
    libs = ["ceres", "glog"]
    for so in sorted(glob.glob(os.path.join(lib_dir, "libabsl_*.so"))):
        libs.append(re.sub(r"^lib(.*)\.so$", r"\1", os.path.basename(so)))
    libs.append("stdc++")

    ffibuilder.set_source(
        "_ext",
        '#include "src/ext.h"',
        extra_objects=["src/uncertainty_pnp.cpp.o"],
        library_dirs=[lib_dir],
        libraries=libs,
    )

    if __name__ == "__main__":
        ffibuilder.compile(verbose=True)
        os.system("rm -f src/*.o *.o")

Then force a clean rebuild (cffi caches _ext.c, so it must be deleted):

    cd core/csrc/uncertainty_pnp
    rm -f _ext.c _ext*.so src/*.o *.o
    python setup.py
    python -c "import _ext; print('ceres OK', _ext.__file__)"

---

## 3. CUDA extensions: "CUDA_HOME environment variable is not set"

Symptom: flow / ransac_voting / torch_nndistance fail in CUDAExtension(...) with
`OSError: CUDA_HOME environment variable is not set`, even after exporting
CUDA_HOME and confirming `nvcc --version` works.

Do NOT just chase the env var. Probe the real cause:

    python -c "import os; print('env', os.environ.get('CUDA_HOME')); from torch.utils.cpp_extension import CUDA_HOME; print('torch', CUDA_HOME)"
    python -c "import torch; print('torch.version.cuda =', torch.version.cuda)"

If you see `torch.version.cuda = None`, PyTorch is a CPU-only build. torch only
resolves CUDA_HOME for CUDA builds, and a CPU torch cannot build CUDAExtensions
regardless of env vars.

Fix: install a CUDA PyTorch matching the env CUDA 11.8 toolkit and the A10G:

    nvidia-smi    # confirm GPU + driver
    pip uninstall -y torch torchvision torchaudio
    pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    # want: 2.4.1+cu118 11.8 True NVIDIA A10G

A driver advertising "CUDA 13" is fine; NVIDIA drivers are backward compatible,
so a cu118 runtime runs on it.

---

## 4. torch install fails on a corrupted Pillow

Symptom:

    ERROR: Could not install packages due to an OSError:
    [Errno 2] No such file or directory: '.../pillow-12.2.0.dist-info/METADATA'

Cause: a broken/partial Pillow (leftover pillow_simd + missing METADATA) makes
pip abort while resolving torchvision's Pillow dependency.

Fix:

    SP=$(python -c "import site; print(site.getsitepackages()[0])")
    rm -rf "$SP"/pillow-*.dist-info "$SP"/PIL "$SP"/~*
    pip install --no-cache-dir --force-reinstall pillow
    python -c "import PIL, PIL.Image; print('Pillow', PIL.__version__, 'OK')"
    # then re-run the torch install from section 3

---

## 5. detectron2: undefined symbol at::_ops::zeros_like::call

Symptom:

    ImportError: .../detectron2/_C....so:
    undefined symbol: _ZN2at4_ops10zeros_like4call...   # at::_ops::zeros_like::call(...)

Cause: detectron2's _C extension was compiled against a different torch; the
ATen ABI no longer matches after the torch swap.

Fix (editable source install):

    cd /path/to/src/detectron2
    rm -rf build
    rm -f detectron2/_C*.so
    export CUDA_HOME="$CONDA_PREFIX"
    export TORCH_CUDA_ARCH_LIST="8.6"     # A10G
    python setup.py build_ext --inplace
    python -c "import torch, detectron2; from detectron2 import _C; print('d2 OK', detectron2.__version__)"

---

## Full clean rebuild runbook

    conda activate gdrnpp_env

    # 0. CUDA env (permanent so subshells inherit it)
    conda env config vars set CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="8.6" -n gdrnpp_env
    conda deactivate && conda activate gdrnpp_env
    nvcc --version

    # 1. Ensure CUDA PyTorch (fix Pillow + install cu118 if this asserts)
    python -c "import torch; assert torch.version.cuda, 'CPU-only torch!'; print(torch.__version__, torch.version.cuda)"

    # 2. (one-time) apply the uncertainty_pnp setup.py from section 2

    # 3. Build all gdrnpp native extensions
    cd /path/to/src/gdrnpp
    ( cd core/csrc/uncertainty_pnp && rm -f _ext.c _ext*.so src/*.o *.o )
    sh scripts/compile_all.sh

    # 4. Rebuild detectron2 against the current torch (section 5)

## Verification

    python - <<'PY'
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available())
    import detectron2; from detectron2 import _C
    print("detectron2", detectron2.__version__, "_C OK")
    PY

    cd core/csrc/uncertainty_pnp && python -c "import _ext; print('uncertainty_pnp _ext OK')"

Expected: torch 2.4.1+cu118 cuda 11.8 True, detectron2 0.6 _C OK,
uncertainty_pnp _ext OK.

## Optional cleanups

- Pillow-SIMD: the env originally had it (faster image ops); it was replaced by
  stock Pillow. Reinstall pillow-simd only if you need the throughput.
- `iopath 0.1.10` vs detectron2's `<0.1.10` pin is a harmless resolver warning.
- Cosmetic warnings (`-Wignored-attributes`, pybind11 visibility,
  "no g++-11 version bounds for CUDA 11.8") are safe to ignore.
