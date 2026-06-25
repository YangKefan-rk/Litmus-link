# PBMT/NC HAND TODO

- PBMT=3 reserved page-fault negative case.
- Cacheable/NC alias with and without `fence iorw,iorw; cbo.flush; fence iorw,iorw`.
- PBMT=NC versus M-mode Bare access to the same PA.
- PBMT=IO platform-specific access and ordering cases.
- AMO on PBMT=NC only after platform atomicity PMA is known.
