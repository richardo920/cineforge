"""Curated metadata only; no downloads, dependency traversal or runtime authority."""

from dataclasses import replace

from cineforge.devices import Backend, Precision
from cineforge.rights import EvidenceRef, RightsLedger, RightsRecord, RightsRef, RightsReview, Use
from cineforge.skills import (
    CatalogEntry,
    Compatibility,
    Permissions,
    ProfileEvidence,
    SkillCatalog,
    SourcePin,
    SourceReview,
)

_AUDIT_SHA256 = '2ba1d4fa1f14ba238af0c8d54d6d81902d4e0bbeb0bc270e643ea1a42229a88a'  # pragma: allowlist secret - public provenance hash
_REVIEWER = "m43-curation-4f9ed59e"
_AUDIT_URI = "urn:sha256:" + _AUDIT_SHA256
_AUDIT = EvidenceRef(_AUDIT_URI, _AUDIT_SHA256)

_SOURCES = (
    SourcePin(
        'https://github.com/badgids/Story-Film-Skills',
        '5ec9daf3950a987424a879ab5fef583b013afa8a',  # pragma: allowlist secret - public source revision
        'skills/write-screenplay/SKILL.md',
        'f1a52d598da38ef23dcf5ac93f568a8c0e1747e655474e6604e4b98c892fe0a5',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/badgids/Story-Film-Skills',
        '5ec9daf3950a987424a879ab5fef583b013afa8a',  # pragma: allowlist secret - public source revision
        'LICENSE',
        'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/badgids/Story-Film-Skills',
        '5ec9daf3950a987424a879ab5fef583b013afa8a',  # pragma: allowlist secret - public source revision
        'NOTICE',
        'fe90d2f70dc20379f535c8961bec205faf968429f9a4fd06539ce1994bf97de3',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/badgids/Story-Film-Skills',
        '5ec9daf3950a987424a879ab5fef583b013afa8a',  # pragma: allowlist secret - public source revision
        'skills/shot-design/SKILL.md',
        '29950bfcef4e84d3426d15fba694e63064e4ac1868679099fa5edb0c7662a1f2',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/badgids/Story-Film-Skills',
        '5ec9daf3950a987424a879ab5fef583b013afa8a',  # pragma: allowlist secret - public source revision
        'skills/sound-design/SKILL.md',
        '5c7a65a32f942a67e4f5af190ba0bdb742400bb590cb2606c0919056aa9313a2',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/badgids/Story-Film-Skills',
        '5ec9daf3950a987424a879ab5fef583b013afa8a',  # pragma: allowlist secret - public source revision
        'skills/edit-plan/SKILL.md',
        '47286eec6f1f35638840fe5272ed2214dd208333424545bfd3856ce94ba70850',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B',
        '37ec512624d61f7aa208f7ea8140a131f93afc9a',  # pragma: allowlist secret - public source revision
        'README.md',
        'd19fb6c377090790940f64af3423c87d289abc9fdee4d45963018e18e5d84119',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/Wan-Video/Wan2.1',
        '9737cba9c1c3c4d04b33fcad41c111989865d315',  # pragma: allowlist secret - public source revision
        'LICENSE.txt',
        'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/Wan-Video/Wan2.1',
        '9737cba9c1c3c4d04b33fcad41c111989865d315',  # pragma: allowlist secret - public source revision
        'README.md',
        '3bdaca69aa116222a49859dc21d4a735b60a580af7e1b81531085c87227e4486',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/Wan-Video/Wan2.1',
        '9737cba9c1c3c4d04b33fcad41c111989865d315',  # pragma: allowlist secret - public source revision
        'wan/configs/wan_t2v_1_3B.py',
        '6976ec4c03f737b69ae17afe13339b32905739205d4c19f16c8b329f9f06e986',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/comfyanonymous/ComfyUI_examples',
        'f9431bb000ce792094ff345446e22cac1ea6cef3',  # pragma: allowlist secret - public source revision
        'wan/text_to_video_wan.json',
        '436b9c074b677011b0f50a9feb8c4b4128abbbd9e4acaf34d8eeb80efbd74a64',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/comfyanonymous/ComfyUI_examples',
        'f9431bb000ce792094ff345446e22cac1ea6cef3',  # pragma: allowlist secret - public source revision
        'LICENSE',
        'b43f63a6847e68df94f5757fbca5c9dd83db1db1c7d638689181eae33f58d955',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/comfyanonymous/ComfyUI_examples',
        'f9431bb000ce792094ff345446e22cac1ea6cef3',  # pragma: allowlist secret - public source revision
        'wan/README.md',
        '704c31de4abf39dfeb1db1b6fd208d1ceccaa6dfb8aaa15be5fbb5d5b5d4c4d6',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/richardo920/cineforge',
        '3c70f19772b37f6e61795287cfedc0361ee374ce',  # pragma: allowlist secret - public source revision
        'config/comfy/wan21-t2v-1.3b-api.json',
        '8baab77ae0a01374dfad997d753eba2d7bad84945088dc1f90fdf7f9cf04d529',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/richardo920/cineforge',
        '3c70f19772b37f6e61795287cfedc0361ee374ce',  # pragma: allowlist secret - public source revision
        'config/comfy/wan21-t2v-1.3b.json',
        'f028fc75ba694605e327eb30ba5d411a8342e8841698f51cf9a6e399a0b1f60e',  # pragma: allowlist secret - public source hash
    ),
    SourcePin(
        'https://github.com/richardo920/cineforge',
        '3c70f19772b37f6e61795287cfedc0361ee374ce',  # pragma: allowlist secret - public source revision
        'docs/evidence/m2-live-shot.json',
        '2d0b1ef79034300eff9d2738f4260887f71b58228ccb82158db8c1dc90af1d6c',  # pragma: allowlist secret - public source hash
    ),
)

_FILM_NOTICE = 'Story-Film Skills\nCopyright 2026 Alan Guice (Badgids)\nDeveloped by Alan Guice (Badgids)\n\nThis product includes software originally developed by Alan Guice (Badgids).\n\nAttribution notice for redistributed derivative works:\n"Story-Film Skills was originally developed by Alan Guice (Badgids)."\n\nKeep this NOTICE text with redistributed Derivative Works when Section 4(d) of\nthe Apache License, Version 2.0 applies. This NOTICE is informational and does\nnot change the Apache License, Version 2.0.\n'
_WORKFLOW_NOTICE = 'Permission to use, copy, modify, and/or distribute these workflows for\nany purpose with or without fee is hereby granted.\n\nTHE WORKFLOWS ARE PROVIDED “AS IS” AND THE AUTHOR DISCLAIMS ALL\nWARRANTIES WITH REGARD TO THESE WORKFLOWS INCLUDING ALL IMPLIED WARRANTIES\nOF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE\nFOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY\nDAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN\nAN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT\nOF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THESE WORKFLOWS.\n'

# Fixed assertions from the source/terms review; never recomputed on load.
_REVIEWS = {
    'edit-plan': (
        'a8493aa2d7866e276f7a03b2fd057c675a0ec174e77c412f2282436afd84609c',  # pragma: allowlist secret - fixed public source review subject
        '4070611538f38f9faec2c18bd8239c82129050d6befcef320fca5e2742ff8d38',  # pragma: allowlist secret - fixed public rights review subject
    ),
    'shot-design': (
        '1d4cc641da08bd81fc02e83c68c714c26736c3b74efd89ebb1ae0831b92980be',  # pragma: allowlist secret - fixed public source review subject
        '19b58ecb6677b07ab20691952fd54e9ea6351f40edd8edeb225afe33672b348b',  # pragma: allowlist secret - fixed public rights review subject
    ),
    'sound-design': (
        'b9aacf6a93439eccff1b6c0ae93c626495692422c10119c6b576a7847126e8a9',  # pragma: allowlist secret - fixed public source review subject
        'cd8b02a3a18274cb78dda27db8bbe8d2ab4d56d7bd91ba671f2a965634988880',  # pragma: allowlist secret - fixed public rights review subject
    ),
    'wan21-comfy-recipe': (
        '55fd0fa1a441a6019bebbf8055f959618fb15a4e184ffb37f16ac78fd7f7b499',  # pragma: allowlist secret - fixed public source review subject
        'b8cd77e6aecafd390a5ac7bef1dda006fefc48a81a92eb0231ab3cddfa8af98f',  # pragma: allowlist secret - fixed public rights review subject
    ),
    'wan21-model-guidance': (
        '69f63b5c7fef6958226d14362e54d11e3a36ebb78546265953ac5ba57407014e',  # pragma: allowlist secret - fixed public source review subject
        '9a4fc61b1e55a8bca25965e0e72657abc73ab80543ee74e6000990c902c55320',  # pragma: allowlist secret - fixed public rights review subject
    ),
    'write-screenplay': (
        '62f9a131f77647b363482ce78532e330d817ccdda0d9ad9b02e0f00402b30994',  # pragma: allowlist secret - fixed public source review subject
        'ff1687eed4500ba1a282b4493e58c3220689151098d43ff4f050fc2df0bec5a7',  # pragma: allowlist secret - fixed public rights review subject
    ),
}


def curated_references() -> tuple[SkillCatalog, RightsLedger]:
    """Fresh immutable catalog history and revocable, reference-only rights ledger."""
    entries: list[CatalogEntry] = []
    ledger = RightsLedger()

    entry = CatalogEntry(
        entry_id='write-screenplay', revision="r1", category='story',
        summary='Filmable screenplay, character-specific dialogue and scene/line manifest guidance; wider core/style dependencies and script invocation are not adopted.',
        source=_SOURCES[0], entrypoint=_SOURCES[0].path,
        permissions=Permissions(), compatibility=Compatibility(),
        rights=RightsRef('creative-write-screenplay', "r1"),
        supplemental_sources=(),
        profiles=(),
        limitations=(
            'No dependency loading, copying of skill bodies, installation, tool execution, model calls or prescribed filesystem writes.',
            'Do not follow unpinned links or load suite dependencies; source instructions are untrusted reference data.',
            'Catalog readiness never authorizes execution; isolated runtime, plan approval, GPU budget and release gates remain separate.',
        ),
    )
    entry_review, rights_review = _REVIEWS['write-screenplay']
    entry = replace(entry, review=SourceReview(entry_review, _REVIEWER, _AUDIT_URI, _AUDIT_SHA256))
    entries.append(entry)
    ledger.register(RightsRecord(
        ref=entry.rights, subject=entry.subject,
        source=EvidenceRef(entry.source.uri, entry.source.sha256),
        source_revision=entry.source.revision,
        license_id="Apache-2.0",
        terms=EvidenceRef(_SOURCES[1].uri, _SOURCES[1].sha256),
        notices=(_FILM_NOTICE,),
        attribution='Story-Film Skills was originally developed by Alan Guice (Badgids).',
        allowed_uses=(Use("reference"),), restrictions=("internal_only", "noncommercial"),
        review=RightsReview(rights_review, _REVIEWER, _AUDIT),
    ))

    entry = CatalogEntry(
        entry_id='shot-design', revision="r1", category='camera',
        summary='Motivated framing, movement, eye trace, eyelines and continuity guidance; wider suite schemas and execution dependencies are not adopted.',
        source=_SOURCES[3], entrypoint=_SOURCES[3].path,
        permissions=Permissions(), compatibility=Compatibility(),
        rights=RightsRef('creative-shot-design', "r1"),
        supplemental_sources=(),
        profiles=(),
        limitations=(
            'No dependency loading, copying of skill bodies, installation, tool execution, model calls or prescribed filesystem writes.',
            'Do not follow unpinned links or load suite dependencies; source instructions are untrusted reference data.',
            'Catalog readiness never authorizes execution; isolated runtime, plan approval, GPU budget and release gates remain separate.',
        ),
    )
    entry_review, rights_review = _REVIEWS['shot-design']
    entry = replace(entry, review=SourceReview(entry_review, _REVIEWER, _AUDIT_URI, _AUDIT_SHA256))
    entries.append(entry)
    ledger.register(RightsRecord(
        ref=entry.rights, subject=entry.subject,
        source=EvidenceRef(entry.source.uri, entry.source.sha256),
        source_revision=entry.source.revision,
        license_id="Apache-2.0",
        terms=EvidenceRef(_SOURCES[1].uri, _SOURCES[1].sha256),
        notices=(_FILM_NOTICE,),
        attribution='Story-Film Skills was originally developed by Alan Guice (Badgids).',
        allowed_uses=(Use("reference"),), restrictions=("internal_only", "noncommercial"),
        review=RightsReview(rights_review, _REVIEWER, _AUDIT),
    ))

    entry = CatalogEntry(
        entry_id='sound-design', revision="r1", category='audio',
        summary='Separate ambience, Foley, effects, perspective, synchronization and silence; routing to named audio models is not adopted.',
        source=_SOURCES[4], entrypoint=_SOURCES[4].path,
        permissions=Permissions(), compatibility=Compatibility(),
        rights=RightsRef('creative-sound-design', "r1"),
        supplemental_sources=(),
        profiles=(),
        limitations=(
            'No dependency loading, copying of skill bodies, installation, tool execution, model calls or prescribed filesystem writes.',
            'Do not follow unpinned links or load suite dependencies; source instructions are untrusted reference data.',
            'Catalog readiness never authorizes execution; isolated runtime, plan approval, GPU budget and release gates remain separate.',
        ),
    )
    entry_review, rights_review = _REVIEWS['sound-design']
    entry = replace(entry, review=SourceReview(entry_review, _REVIEWER, _AUDIT_URI, _AUDIT_SHA256))
    entries.append(entry)
    ledger.register(RightsRecord(
        ref=entry.rights, subject=entry.subject,
        source=EvidenceRef(entry.source.uri, entry.source.sha256),
        source_revision=entry.source.revision,
        license_id="Apache-2.0",
        terms=EvidenceRef(_SOURCES[1].uri, _SOURCES[1].sha256),
        notices=(_FILM_NOTICE,),
        attribution='Story-Film Skills was originally developed by Alan Guice (Badgids).',
        allowed_uses=(Use("reference"),), restrictions=("internal_only", "noncommercial"),
        review=RightsReview(rights_review, _REVIEWER, _AUDIT),
    ))

    entry = CatalogEntry(
        entry_id='edit-plan', revision="r1", category='editing',
        summary='Cut order, pacing, transitions, bridges and pickups; file-writing conventions are not adopted.',
        source=_SOURCES[5], entrypoint=_SOURCES[5].path,
        permissions=Permissions(), compatibility=Compatibility(),
        rights=RightsRef('creative-edit-plan', "r1"),
        supplemental_sources=(),
        profiles=(),
        limitations=(
            'No dependency loading, copying of skill bodies, installation, tool execution, model calls or prescribed filesystem writes.',
            'Do not follow unpinned links or load suite dependencies; source instructions are untrusted reference data.',
            'Catalog readiness never authorizes execution; isolated runtime, plan approval, GPU budget and release gates remain separate.',
        ),
    )
    entry_review, rights_review = _REVIEWS['edit-plan']
    entry = replace(entry, review=SourceReview(entry_review, _REVIEWER, _AUDIT_URI, _AUDIT_SHA256))
    entries.append(entry)
    ledger.register(RightsRecord(
        ref=entry.rights, subject=entry.subject,
        source=EvidenceRef(entry.source.uri, entry.source.sha256),
        source_revision=entry.source.revision,
        license_id="Apache-2.0",
        terms=EvidenceRef(_SOURCES[1].uri, _SOURCES[1].sha256),
        notices=(_FILM_NOTICE,),
        attribution='Story-Film Skills was originally developed by Alan Guice (Badgids).',
        allowed_uses=(Use("reference"),), restrictions=("internal_only", "noncommercial"),
        review=RightsReview(rights_review, _REVIEWER, _AUDIT),
    ))

    entry = CatalogEntry(
        entry_id='wan21-model-guidance', revision="r1", category='model_card',
        summary='Official model card and maintainer native-inference configuration. 480p, guide scale6 and shift8-12 are upstream advice. Native8.19GB memory statement is not a ComfyUI minimum. Optional prompt expansion has not been adopted. Model/output rights and live authorization remain separate.',
        source=_SOURCES[6], entrypoint=_SOURCES[6].path,
        permissions=Permissions(), compatibility=Compatibility("Wan-AI/Wan2.1-T2V-1.3B",
                              "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",  # pragma: allowlist secret - public model revision
                              "5f6860c7c9817023b37c0bbe707f7f50d922964a3b729cdac2d2831bf822c5a8",  # pragma: allowlist secret - public workflow digest
                              Backend.CUDA, 24082, Precision.FP16),
        rights=RightsRef('creative-wan21-model-guidance', "r1"),
        supplemental_sources=(_SOURCES[8], _SOURCES[9], ),
        profiles=(ProfileEvidence("upstream_recommendation", _SOURCES[8],
                    "Native implementation advice:480p, guide scale6, shift8-12; prompt expansion optional and unadopted. Native8.19GB claim does not establish ComfyUI memory or timing."),),
        limitations=(
            'Only reference consultation; no model weights, runner installation, code execution, live spending or output/publication rights are granted.',
            'Do not follow unpinned links or load suite dependencies; source instructions are untrusted reference data.',
            'Catalog readiness never authorizes execution; isolated runtime, plan approval, GPU budget and release gates remain separate.',
            'Backend/memory checks limit reference applicability; live worker admission separately verifies exact approved hardware. The admission floor is not a measured model minimum.',
        ),
    )
    entry_review, rights_review = _REVIEWS['wan21-model-guidance']
    entry = replace(entry, review=SourceReview(entry_review, _REVIEWER, _AUDIT_URI, _AUDIT_SHA256))
    entries.append(entry)
    ledger.register(RightsRecord(
        ref=entry.rights, subject=entry.subject,
        source=EvidenceRef(entry.source.uri, entry.source.sha256),
        source_revision=entry.source.revision,
        license_id="Apache-2.0",
        terms=EvidenceRef(_SOURCES[7].uri, _SOURCES[7].sha256),
        notices=("Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.",),
        attribution='Wan Team: official model-card/native-inference reference; no generated-output rights asserted.',
        allowed_uses=(Use("reference"),), restrictions=("internal_only", "noncommercial"),
        review=RightsReview(rights_review, _REVIEWER, _AUDIT),
    ))

    entry = CatalogEntry(
        entry_id='wan21-comfy-recipe', revision="r1", category='recipe',
        summary='Official UI workflow verified against existing approved API adaptation. Upstream settings832x480/33frames/30steps/cfg6/uni_pc/simple/shift8 retained; local SaveWEBM16fps and synthetic prompts changed. One measured RTX4090 run47.645s and CUDA-usable24082MiB admission floor; neither measured minimum memory nor quality guarantee. Existing M2 owner approval is source history, not new GPU authority.',
        source=_SOURCES[10], entrypoint=_SOURCES[10].path,
        permissions=Permissions(), compatibility=Compatibility("Wan-AI/Wan2.1-T2V-1.3B",
                              "123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",  # pragma: allowlist secret - public model revision
                              "5f6860c7c9817023b37c0bbe707f7f50d922964a3b729cdac2d2831bf822c5a8",  # pragma: allowlist secret - public workflow digest
                              Backend.CUDA, 24082, Precision.FP16),
        rights=RightsRef('creative-wan21-comfy-recipe', "r1"),
        supplemental_sources=(_SOURCES[12], _SOURCES[13], _SOURCES[14], _SOURCES[15], ),
        profiles=(ProfileEvidence("upstream_recommendation", _SOURCES[10],
                    "Official core-node graph:832x480,33frames,30steps,cfg6,uni_pc/simple,shift8."),
                  ProfileEvidence("local_observation", _SOURCES[15],
                    "One RTX4090 run:47.645s,33frames at16fps;24082MiB CUDA-usable admission floor. No measured peak/minimum memory, 5090 compatibility, cross-hardware or creative-quality guarantee.")),
        limitations=(
            'Only reference consultation; no model weights, runner installation, code execution, live spending or output/publication rights are granted.',
            'Do not follow unpinned links or load suite dependencies; source instructions are untrusted reference data.',
            'Catalog readiness never authorizes execution; isolated runtime, plan approval, GPU budget and release gates remain separate.',
            'Backend/memory checks limit reference applicability; live worker admission separately verifies exact approved hardware. The admission floor is not a measured model minimum.',
        ),
    )
    entry_review, rights_review = _REVIEWS['wan21-comfy-recipe']
    entry = replace(entry, review=SourceReview(entry_review, _REVIEWER, _AUDIT_URI, _AUDIT_SHA256))
    entries.append(entry)
    ledger.register(RightsRecord(
        ref=entry.rights, subject=entry.subject,
        source=EvidenceRef(entry.source.uri, entry.source.sha256),
        source_revision=entry.source.revision,
        license_id='LicenseRef-ComfyUI-Examples-Permissive',
        terms=EvidenceRef(_SOURCES[11].uri, _SOURCES[11].sha256),
        notices=(_WORKFLOW_NOTICE,),
        attribution='Official ComfyUI_examples workflow; Cineforge adaptation and bounded M2 observation remain distinct.',
        allowed_uses=(Use("reference"),), restrictions=("internal_only", "noncommercial"),
        review=RightsReview(rights_review, _REVIEWER, _AUDIT),
    ))

    return SkillCatalog(tuple(entries)), ledger
