# Character registry proposal

A shot references `character_id` + `revision`; a registry entry carries identity description, wardrobe, reference artifact URIs/hashes, optional voice and model-specific conditioning references, and explicit rights/consent evidence.

Store references privately, require approved likeness/voice permissions before execution, and prevent silent replacement of a pinned revision. Test duplicate revisions, missing references, unsupported conditioning, and absent or revoked consent.

`characters.py` implements immutable metadata and review-bound `CharacterPin` values (`id@revision#sha256`) in existing `Shot.character_revision`; `pin_shot` and `CharacterRegistry.resolve_shot` preserve schema version one. Resolution checks current `RightsLedger` permissions for every linked reference/likeness/voice in the requested `Use` context; prepare reviews before registration, since registered revisions cannot be replaced.

Reference URIs are private logical locators, never read by the registry. Byte verification, model capability, and three-shot human identity review belong to M4.2; metadata readiness alone authorizes no execution.
