# Skills catalog

Catalog readiness permits reference selection only; `execution_authorized` remains false. M4.4 owns installation/isolation, and GPU spending, likeness use and publication require separate approvals.

`curated_references()` supplies six metadata-only references: screenplay, camera, sound, editing, official Wan guidance and the approved ComfyUI recipe. See the [pinned source/terms review](../evidence/m4-source-review.json); upstream advice and bounded local observations remain distinct, and suite dependencies/scripts are unadopted.

```python
from cineforge.curated_skills import curated_references
from cineforge.skills import bind_shot

catalog, ledger = curated_references()
pin = next(p for p in catalog.pins if (p.entry_id, p.revision) == ("shot-design", "r1"))
binding = bind_shot(shot, (pin,))  # separate sidecar; Shot v1 stays unchanged
results = catalog.resolve_shot(binding, shot=shot, ledger=ledger)
assert all(result.ready for result in results)
```

Pins bind immutable source bytes, permissions and compatibility; changing declarations requires fresh source and rights reviews. Current revocation is checked on each resolution; review assertions are trusted operator metadata, not authenticated upstream endorsements or legal certification.

`SkillRuntime` installs only the requested reviewed instruction file into a private immutable slot; source/receipt bytes and current rights are checked again when opening a session and before each call. Its broker supports bounded `read_file`/exclusive `write_file` operations on exact paths allowed by both the entry and operator; network, shell and arbitrary code always deny. This is a managed tool boundary, not an OS sandbox or a restriction on host tools used outside the broker.
