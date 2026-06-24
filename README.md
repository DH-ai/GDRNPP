# GDRNPP Modern Compatibility Patch

## Branch

`blender42_py310_fix`

## Purpose

This branch contains compatibility fixes required to run the original GDRNPP + YOLOX codebase on a modern environment:

* Python 3.10+
* PyTorch 2.4+
* NumPy 2.x / recent NumPy releases
* Blender 4.2+ exported PLY models
* Modern CUDA toolchains

The upstream codebase was originally developed and tested with significantly older versions of Python, PyTorch, NumPy, and Blender. Several compatibility issues prevented training and dataset loading on current systems.

---

## Included Fixes

### Python 3.10 Compatibility

#### collections.Sequence removal

Replaced:

```python
from collections import Sequence
```

with:

```python
from collections.abc import Sequence
```

Required because `collections.Sequence` was removed in newer Python versions.

---

### PyTorch Lightning Compatibility

Adjusted dependencies to use a version compatible with the repository.

Repository requirements indicate:

```text
pytorch-lightning ~= 1.6.x
```

Modern Lightning releases removed APIs used by GDRNPP.

---

### Binary File Detection Fix

Patched:

```text
lib/utils/is_binary_file.py
```

Issues:

* Python 2 `unicode()` usage
* Invalid encoding handling when chardet returns `None`

This prevented loading binary Blender-generated PLY files.

---

### NumPy Compatibility

Removed usage of deprecated aliases:

```python
np.float
np.int
np.bool
np.object
np.str
```

Replaced with modern equivalents:

```python
float
int
bool
object
str
```

or explicit NumPy dtypes where appropriate.

These aliases were removed in NumPy >= 2.0.

---

### Blender 4.2 PLY Support

Blender 4.2 exports face indices as:

```text
property list uchar uint vertex_indices
```

The original loader only supported:

```text
int
uchar
float
double
```

Added support for additional PLY datatypes:

```python
formats = {
    "char": ("b", 1),
    "uchar": ("B", 1),
    "short": ("h", 2),
    "ushort": ("H", 2),
    "int": ("i", 4),
    "uint": ("I", 4),
    "float": ("f", 4),
    "double": ("d", 8),
}
```

This enables loading Blender 4.x generated binary PLY meshes.

---

## Dataset Integration

Added support for custom BOP-format datasets:

* Dataset registration
* Dataset metadata
* Custom dataset references
* YOLOX training integration
* GDRNPP pose-estimation integration

Custom dataset structure:

```text
bop/
├── camera.json
├── models/
│   ├── obj_000001.ply
│   ├── obj_000002.ply
│   └── obj_000003.ply
└── train_pbr/
    └── 000000/
        ├── rgb/
        ├── depth/
        ├── mask/
        ├── mask_visib/
        ├── scene_gt.json
        ├── scene_gt_info.json
        ├── scene_gt_coco.json
        └── scene_camera.json
```

---

## Notes

This branch is intended as a modernization layer and does not change model architecture, training logic, or evaluation procedures.

All modifications are compatibility-focused and designed to preserve original behavior while allowing execution on contemporary ML environments.
