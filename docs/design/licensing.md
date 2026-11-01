# Licensing practices

Code remains proprietary pending owner selection of an open-source license. Do not copy third-party code, skills, weights, music, voices, or reference images without reviewing the exact pinned source terms and preserving required notices.

Maintain a rights ledger with source, revision/hash, license evidence, attribution, permitted use, and distribution restrictions. Before public release, audit code dependencies, models, generated outputs, character/voice consent, and assets separately; exclude credentials and private media from the public history.

`cineforge.rights` binds immutable subject/content revisions to operator-reviewed declarations; every use checks the current ledger for revocation, supersession, exact permissions, notices and consent. `RightsRecord` supports strict schema-v1 JSON; `RightsLedger.preflight_export()` returns deterministic attribution metadata only when every supplied request passes.

Callers must supply the complete export inventory and obtain real terms/consent review; declarations are operator assertions, not legal certification or execution/publication approval. The ledger performs no downloads, media export, persistence or cloud calls.
