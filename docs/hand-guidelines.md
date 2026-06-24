# HAND Guidelines

HAND tests are required when a scenario needs page-table construction, exception recovery, privilege-mode setup, or target-specific memory maps that the automatic generator should not guess.

Every HAND category must include a `TODO.md` with:

- Scenario title.
- Required ISA extensions and privilege features.
- Why the case cannot be generated automatically yet.
- Expected observation type: `rvwmo-herd`, `prose-spec`, `hardware-observation`, `negative-exception`, or `platform-specific`.
- Required synchronization sequence, if any.
