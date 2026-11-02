"""In-memory original PCM16 stem import; no decoder, storage, mixing or approval.

Original-source review is an existing operator assertion, not machine proof of
ownership. This boundary rechecks rights; it cannot issue new rights or export review.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from cineforge.audio import AudioTimebase, Cue, CueSheet, RevisionPin, Stem
from cineforge.audio_pcm import PCMError, inspect_pcm16_wav
from cineforge.characters import RightsBinding
from cineforge.contracts import ContractError, _identifier
from cineforge.rights import EvidenceRef, RightsLedger, Use


class StemImportError(ContractError):
    """Fixed diagnostics without private media, source URI or caller content."""


@dataclass(frozen=True, slots=True)
class ImportedStemArtifact:
    pin: RevisionPin
    sheet: RevisionPin
    cue: Cue
    stem: Stem
    timebase: AudioTimebase
    payload: bytes = field(repr=False)
    permission: RightsBinding
    original_source_review: EvidenceRef = field(repr=False)
    sample_rate: int
    channels: int
    frames: int


def import_original_stem(sheet: CueSheet, cue_id: str, payload: bytes, source: RevisionPin,
                         *, ledger: RightsLedger,
                         original_source_review: EvidenceRef | None = None) -> ImportedStemArtifact:
    """Return unchanged bytes selected as original by trusted operator/application code.

    The supplied evidence selects the exact existing rights review as the operator's
    original-source assertion. Equality cannot prove originality, and creative
    metadata/license/URI must never be used to infer this trusted selection.
    """
    if type(sheet) is not CueSheet or type(source) is not RevisionPin or type(ledger) is not RightsLedger:
        raise StemImportError("invalid stem import record")
    _identifier(cue_id, "cue ID")
    cue = next((item for item in sheet.cues if item.cue_id == cue_id), None)
    if cue is None or cue.kind == "dialogue" or cue.utterance_id is not None:
        raise StemImportError("only a current non-dialogue cue can be imported")
    stem = next((item for item in sheet.stems if item.stem_id == cue.stem_id), None)
    expected_kind = "music" if cue.kind == "music" else "asset"
    if (stem is None or stem.kind != cue.kind or cue.source.subject.kind != expected_kind
            or source.sha256 != cue.source.subject.sha256
            or not isinstance(payload, bytes) or len(payload) > 16 * 1024 * 1024
            or hashlib.sha256(payload).hexdigest() != source.sha256):
        raise StemImportError("stem source or bytes mismatch")
    record = ledger.lookup(cue.source.rights)
    if (type(original_source_review) is not EvidenceRef or record is None
            or record.review is None or record.review.evidence != original_source_review):
        raise StemImportError("explicit original-source review selection required")
    for action in ("reference", "copy"):
        if not ledger.preflight_use(cue.source.subject, cue.source.rights, Use(action)).allowed:
            raise StemImportError("stem source rights failed preflight")
    failed = False
    try:
        wave = inspect_pcm16_wav(payload)
    except PCMError:
        failed = True
    if failed:
        raise StemImportError("unsupported or malformed stem PCM WAV")
    if (wave.frames * sheet.timebase.ticks_per_second
            != cue.duration_ticks * wave.sample_rate):
        raise StemImportError("stem duration does not exactly match cue")
    return ImportedStemArtifact(source, RevisionPin(sheet.sheet_id, sheet.revision, sheet.digest()),
                                cue, stem, sheet.timebase, payload, cue.source,
                                original_source_review, wave.sample_rate, wave.channels, wave.frames)
