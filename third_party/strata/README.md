# From Strata

`mtp_fetch.py` is `tools/mtp_fetch.py` of [Strata](https://github.com/Niko1221/Strata), version 0.1.40.3
(commit `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`), unchanged. Its license is the MIT license in `LICENSE`
beside it: Copyright (c) 2026 Niko1221 and the Strata contributors.

It fetches the model's draft head: 31 tensors, 4.9 GB, out of the original 16-bit checkpoint
(Qwen/Qwen3.8-Flash-Next, 360 GB) by HTTP range requests, without downloading the checkpoint, and checks each
against a SHA-256 of a pinned revision. It uses the standard library only.

    python third_party/strata/mtp_fetch.py fetch --out ~/.cache/strata-mlx/mtp      # resumable
    python third_party/strata/mtp_fetch.py verify --out ~/.cache/strata-mlx/mtp     # offline; exit 3 if one is bad

The folder it leaves is what `strata-mlx --mtp <folder>` reads (`strata_mlx/mtp.py`). `HF_ENDPOINT` names a
mirror; the other environment variables are described at the top of the file.
