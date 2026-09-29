# Frozen Game24 baseline-adaptation tasks

The 96 `numbers` rows are sampled at evenly spaced indices from the Apache-2.0
`nlile/24-game` train split at dataset revision
`0106b176e4648061d60c32ddb85ac0816699652d`. Sampling matches the index rule in
the pinned InterWhen example (`numpy.linspace(0, len(dataset)-1, N, dtype=int)`).
The committed file intentionally excludes the dataset's `solutions`, difficulty,
and human success-rate columns. Prompts contain only the four numbers and the
task instructions. This is a public development sample, not a held-out test.

SHA-256: `f20fea0807ad1b56acfaee276bd752df95fff9a760156c6f86bd0ebb0632d3a0`.
