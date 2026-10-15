---
id: e49
kind: experiment
status: measured
---
# Shared-weight controls on an M5 Pro

Question: do generation-based fusion limits generalize from an 80-core M5 Ultra to a 20-core M5 Pro, and do observed timing changes exceed a duplicate engine's measurement noise?

Setup: real Mac mini M5 Pro, 64GiB, macOS 27.0.1, MLX 0.32.3; pinned main 2842c51019604549329cc5d7bb9846ad8b04ff7d, float16, precision environment unset. Q2_0 first, then IQ3_S, one weight set resident at a time. Workload pause/drain and original serving restoration are handled externally by the lab guard, not by this experiment.

Four engines share weights: current kernel paths at fusion limits 6 and 8, a duplicate current limit-6 control, and a trunk-switch-off reference at limit 6. The reference retains current shared draft-head/thread rules and is not a historical-revision reproduction. Engines have independent KV/recurrent states; only one runs at a time. 20 seeded shuffled rounds per window (1/4/7/8), fixed prefix/input IDs, output-logit hashes/argmax and untimed host copies. Forward caches are trimmed after each pass. Paged expert LRU/OS file caches are shared and adapt across the sequence; this is not an SSD cold-read experiment.

Three shuffled answer rounds, five previously used scored synthetic JSON cases plus English prose/code/Chinese throughput prompts. Completion cap 256, explicit empty thinking block, greedy sampling, raw prompt/output IDs, decoded text, completion-limit and correctness flags retained. Startup, kernel/draft warmup and table-warming decision recorded separately. No forced n-gram warm. Answer text may diverge, so answer rate is not fixed-work throughput; report correctness and completed task elapsed time.

Results: fusion8 is slower than6 by1.336/1.708ms at7/8 tokens for Q2_0, and1.089/1.194ms for IQ3_S. Paired 20-round bootstrap intervals exclude zero, with larger duplicate-control noise on paged IQ3_S. Fusion6/8 bits match at1/4; differ at7/8. IQ3_S argmax differs at7/8 too. Updated trunk switches improve all measured windows versus their disabled reference. See bench/results/e46-m5pro raw samples and RESULTS.md. This is one20-core GPU and one prefix/window fixture, not all M5 chips. Strict scored answers Q2_0 12/15 per arm (Markdown-fenced invoice failures retained), IQ3_S15/15. Answer IDs vary between duplicate controls; cap-limited throughput prompts are not completed-task quality. Table warming skipped both. Q2_0 startup incurred2334.2MiB new systemwide swap-outs, measured micro/answers zero; IQ3_S zero new swap-outs throughout. 466 runtime tests passed, two archive wiki-history checks in initial full-suite attempt failed and remain disclosed. Optional fusion-limit override passed30 targeted tests in scratch source, no runtime deployment.

## Concepts
- [[measurement-noise]]
- [[gpu-generation]]

Integration: this contribution is e49 after upstream allocated e46–e48 to separate memory-pressure experiments. Raw result directories retain their original e46 names and contents for provenance; they are the evidence for this e49 page. The change ID is uvuuktql. Regenerate the wiki graph with repository history; archive-only wiki failures do not qualify wiki integration.

## Cleaned reusable protocol and reproduction

The legacy evidence above belongs to bench/results/e46-m5pro/legacy-harness.py. The cleaned experiment is import-safe, reads actual physical memory, refuses existing output paths, validates inputs, records precision/input/source hashes, and emits per-request start/end events. Its default resets adaptive drafting policy per request (within-request adaptation remains); preserve-history is explicit. These changes have CPU contract coverage and a bounded Q2_0 execution smoke:16 fixed passes and8/8 scored answers, complete-plan analyzer checks passed. This is execution validation, not a new performance study. The full history-backed Mac suite passed505 tests and22 subtests; wiki checks had no errors/warnings. See bench/results/e46-cleaned-smoke. They do not silently regenerate or replace legacy results.

Use the full repository with MLX and the model/draft dependencies installed. First establish an exclusive inference window externally; this script owns no service/workload guard. Example:

```sh
python bench/experiments/e49_shared_weight_controls.py --model /path/to/shard1.gguf --mtp /path/to/draft --tokenizer /path/to/tokenizer.json --fixtures bench/results/e46-m5pro/synthetic-fixtures.json --output /path/to/new-run --micro-rounds 20 --answer-rounds 3 --windows 1,4,7,8
python bench/compare_results.py /path/to/new-run --output /path/to/new-summary.json
```

The experiment is Qwen4Exp-specific, not a universal model loader. Default stop tokens are resolved from the supplied matching tokenizer. Inherited precision variables are recorded; float32 defaults TF32 to0 only if unset. The analyzer rejects incomplete/duplicate samples, mismatched paired prompts, invalid timing, nonfinite recorded logits and invalid completion metadata. It distinguishes JSON booleans from numbers. Raw Markdown-fenced answers remain strict JSON failures. Never compare cap-limited answer rates as completed-task success.

An optional --fusion-limit auto|6|8 serving override preserves automatic defaults, validates before model load and records the effective value. It provides a measured override without hardcoding a rule for every Pro variant. No automatic hardware-policy change is proposed.
