# Bounded quality and numerical review

Reviewed all sixteen B/quality-0.json 128-token excerpts and the cross-arm
differences. They remain readable English, discuss the requested code areas,
and show no obvious looping/garbling in these excerpts. This is **not** a
ground-truth code-answer accuracy evaluation: 128-token answers often end mid
sentence, and source-level claims (e.g. interpreting `torch.empty` initialization
as an actual loaded-weight bug) should not be accepted as established facts.

Each arm repeats all 16 continuations exactly over four fresh-cache waves.
Controls A1/A2 also have identical teacher-forced selected-token logprobs at all
1008 available positions. B differs from the control in13/16 continuations,
including two first-token changes, and in26/1008 teacher-forced top1 decisions.
Mean/max selected-token logprob absolute deltas are0.027185/0.448554. France
passes in all three timed processes and the diagnostic process.

Decision: keep the reproducible +5.00% speed result and the implementation as
**default-off**, with the legacy exact pre-mix preserved. Do not call this a
bit-exact optimization or a fix of universal whole-model drift. More extended
code outputs / evaluation are still needed before a broader quality-equivalence
claim or a production-default switch. The measured differences are reproducible
between arithmetic paths, not evidence of within-arm random nondeterminism.
