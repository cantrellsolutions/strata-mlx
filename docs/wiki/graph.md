# The relations

> 中文摘要:概念之间的关系图,由 `tools/wiki_check.py --write` 生成,不要手改。箭头从一个概念指向它影响或依赖的概念。

Written by `tools/wiki_check.py --write` from the `## Relations` of every concept page. Do not edit.

```mermaid
flowchart LR
  bandwidth_floor["The memory bandwidth floor"]
  decode_pass["A decode pass"]
  draft_window["The draft window"]
  exactness["Exactness"]
  expert_lookup_kernel["The Q2_0 experts' table kernel"]
  gpu_generation["The GPU's generation"]
  guess_acceptance["Guess acceptance"]
  hyper_connections["Hyper-connections"]
  kernel_fixed_cost["The fixed cost of a kernel"]
  measurement_noise["Measurement noise"]
  ngram_table["The n-gram table"]
  decode_pass -- limited-by --> bandwidth_floor
  decode_pass -- limited-by --> kernel_fixed_cost
  decode_pass -- depends-on --> measurement_noise
  draft_window -- depends-on --> guess_acceptance
  draft_window -- trades-against --> decode_pass
  draft_window -- tuned-by --> gpu_generation
  exactness -- depends-on --> gpu_generation
  expert_lookup_kernel -- part-of --> decode_pass
  expert_lookup_kernel -- tuned-by --> gpu_generation
  expert_lookup_kernel -- limited-by --> kernel_fixed_cost
  expert_lookup_kernel -- depends-on --> exactness
  hyper_connections -- part-of --> decode_pass
  hyper_connections -- limited-by --> kernel_fixed_cost
  hyper_connections -- tuned-by --> gpu_generation
  hyper_connections -- trades-against --> exactness
  kernel_fixed_cost -- tuned-by --> gpu_generation
  measurement_noise -- costs --> decode_pass
  ngram_table -- costs --> decode_pass
  classDef component fill:#dbeafe,stroke:#555
  class expert_lookup_kernel,ngram_table component
  classDef concept fill:#f3f4f6,stroke:#555
  class decode_pass concept
  classDef constraint fill:#fee2e2,stroke:#555
  class bandwidth_floor,exactness,kernel_fixed_cost constraint
  classDef hardware fill:#fef3c7,stroke:#555
  class gpu_generation hardware
  classDef mechanism fill:#dcfce7,stroke:#555
  class draft_window,hyper_connections mechanism
  classDef method fill:#ede9fe,stroke:#555
  class measurement_noise method
  classDef metric fill:#e0f2fe,stroke:#555
  class guess_acceptance metric
```

Colours by kind: component, concept, constraint, hardware, mechanism, method, metric.

| from | relation | to | evidence |
| :--- | :--- | :--- | :--- |
| [A decode pass](concepts/decode-pass.md) | limited-by | [The memory bandwidth floor](concepts/bandwidth-floor.md) | ROADMAP: Faster answers |
| [A decode pass](concepts/decode-pass.md) | limited-by | [The fixed cost of a kernel](concepts/kernel-fixed-cost.md) | ROADMAP: Now: what is left, #9 |
| [A decode pass](concepts/decode-pass.md) | depends-on | [Measurement noise](concepts/measurement-noise.md) | e45 |
| [The draft window](concepts/draft-window.md) | depends-on | [Guess acceptance](concepts/guess-acceptance.md) | e13 |
| [The draft window](concepts/draft-window.md) | trades-against | [A decode pass](concepts/decode-pass.md) | e45 |
| [The draft window](concepts/draft-window.md) | tuned-by | [The GPU's generation](concepts/gpu-generation.md) | e45, zryukqou (5d1a240) |
| [Exactness](concepts/exactness.md) | depends-on | [The GPU's generation](concepts/gpu-generation.md) | #5 |
| [The Q2_0 experts' table kernel](concepts/expert-lookup-kernel.md) | part-of | [A decode pass](concepts/decode-pass.md) | e06 |
| [The Q2_0 experts' table kernel](concepts/expert-lookup-kernel.md) | tuned-by | [The GPU's generation](concepts/gpu-generation.md) | e44, e45, #9 |
| [The Q2_0 experts' table kernel](concepts/expert-lookup-kernel.md) | limited-by | [The fixed cost of a kernel](concepts/kernel-fixed-cost.md) | #9, wlqvqpzr (79ace94) |
| [The Q2_0 experts' table kernel](concepts/expert-lookup-kernel.md) | depends-on | [Exactness](concepts/exactness.md) | e44 |
| [Hyper-connections](concepts/hyper-connections.md) | part-of | [A decode pass](concepts/decode-pass.md) | e06 |
| [Hyper-connections](concepts/hyper-connections.md) | limited-by | [The fixed cost of a kernel](concepts/kernel-fixed-cost.md) | e07, e09, #9 |
| [Hyper-connections](concepts/hyper-connections.md) | tuned-by | [The GPU's generation](concepts/gpu-generation.md) | e45, zryukqou (5d1a240) |
| [Hyper-connections](concepts/hyper-connections.md) | trades-against | [Exactness](concepts/exactness.md) | e18, xtxomuun (9c93d5b) |
| [The fixed cost of a kernel](concepts/kernel-fixed-cost.md) | tuned-by | [The GPU's generation](concepts/gpu-generation.md) | #9, e45 |
| [Measurement noise](concepts/measurement-noise.md) | costs | [A decode pass](concepts/decode-pass.md) | ROADMAP: Faster answers, e45 |
| [The n-gram table](concepts/ngram-table.md) | costs | [A decode pass](concepts/decode-pass.md) | #10, tousqpoo (8fb6d4c) |

| experiment | concepts |
| :--- | :--- |
| [e43: the Q2_0 experts through MLX's own quantized product](experiments/e43-experts-as-qmm.md) | expert-lookup-kernel |
| [e44: threads to a row of the table kernel](experiments/e44-lookup-threads.md) | expert-lookup-kernel, gpu-generation, draft-window, guess-acceptance, measurement-noise |
| [e45: pull requests 9 and 10 checked on the M4 Max](experiments/e45-pull-request-9-check.md) | decode-pass, hyper-connections, expert-lookup-kernel, kernel-fixed-cost, gpu-generation, measurement-noise, draft-window |
