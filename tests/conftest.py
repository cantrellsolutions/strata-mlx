import os

# #5: on chips with Neural Accelerators (M5) MLX multiplies float32 matrices at reduced precision unless told not
# to, before its first computation: 7e-4 of the largest value where the float32 tests here allow 2e-5.
os.environ.setdefault("MLX_ENABLE_TF32", "0")
