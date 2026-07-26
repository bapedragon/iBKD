# Audited LG Table-1 model subset

This directory vendors the seven student model definitions and matching
configuration files needed by the CUB-200 Table-1 extension.

- Source: <https://github.com/lkhl/tiny-transformers>
- Audited commit: `d2165f74049c906b0afc9f957491960fb3c0cc8b`
- Upstream license bundle: [`LICENSE`](LICENSE)

The model source files and YAML values are copied from that commit.  The local
`pycls.models.build` module is deliberately reduced to the model registry
because teacher loading, distillation, optimization, data loading, and result
management are implemented by this repository.  No model layer or Table-1
architecture value is changed.

The upstream files include their original per-model attribution and license
notices.  In particular, the ConViT source records its CC-BY-NC origin; this
repository uses the code only for non-commercial academic experiments.
