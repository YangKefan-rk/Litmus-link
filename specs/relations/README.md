# Relations

Scalar generated tests preserve standard litmus relation vocabulary where applicable: `po`, `rf`, `fr`, `co`, `ppo`, fences, acquire/release annotations, and syntactic dependencies.

Vector, CMO, PBMT, `FENCE.I`, and `SFENCE.VMA` tests are not forced into unsupported herd relations. Their metadata marks them as `rvwmo-instruction-level`, `prose-spec`, `hardware-observation`, `platform-specific`, or `negative-exception`.
