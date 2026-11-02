# Audio proposal

Keep dialogue, ambience, Foley/effects, and music as separate stems with start/duration cues on the picture timeline. Retain source, rights, speaker/character revision, and generation provenance for every cue.

`cineforge.audio` implements strict v1 metadata with integer scene ticks, exact text hashes, `en` (English) / `zh` (Mandarin baseline), character/voice pins, and current declaration/rights preflight; synthetic voices need no recording reference. Preflight never verifies media bytes or grants execution; dependency bindings conservatively invalidate the whole scene when script, speech, character, picture or cue-sheet identities change, and do not constitute human approval.

Voice permission subjects match the logical voice ID and `vN` revision. `AudioTimebase.ticks_for_seconds()` bridges legacy shot seconds within one float ULP and strictly less than half a tick; all cue/end bounds remain exact integers (`49/24` seconds at `48000` ticks/second → `98000` ticks).

Non-dialogue cues use explicit `shot_id: null` for scene scope and may cross real shot boundaries; dialogue remains shot-bound. The [owner-selected M5 baseline](https://github.com/richardo920/cineforge/issues/79#issuecomment-6565672917) binds one character pin/voice per canonical character across English/Mandarin and speaker aliases; per-language voice references remain unimplemented.

The voice's recording reference must match the current character declaration, including absence for reference-free synthetic voices. Recording and logical voice-profile permissions are checked independently; legacy recording subject revisions (for example `r1`) remain valid beside a logical profile's `vN` revision.

M5 takes short approved script/台词 lines with text revision, language, speaker/character/voice identity, emotion/delivery, pronunciation and timing; English and Mandarin are the minimum. Full brief-to-screenplay planning remains M6; code-switching and multi-person scenes are outside the M5 baseline.

Keep speech synthesis, lip-sync/talking-video generation, soundtrack mixing and audio/visual evaluation separate. Generate approved speech first, then drive talking video or post-process lip sync; bind both media revisions to the shared timeline, and invalidate affected audio/video, sync checks and approvals when script, speech, character or picture changes.

M5 proves an original actor with a visible mouth speaking at least one short approved clip per language, with preserved identity; the mouthless M4 robot cannot prove lip sync. Unsupported face, pose, style or language must fail explicitly; no universal quality guarantee applies.

Analyze speech alignment, loudness, true peak, clipping, silence and sync; optional audio-model feedback cannot replace human review. Audio timeline alignment alone does not prove visible lip sync: require actual human listening and visual approval of English/Mandarin intelligibility, mouth synchronization, identity, music balance, transitions and final mix against exact media revisions; missing, failed or stale approval blocks export.

Reuse reviewed upstream models/workflows only after pinned model/weights/dependencies, rights/voice consent, hardware and budget approval, before any download, upload, paid test or voice cloning. No model is selected or execution authorized by this proposal; default CI stays offline, and generation, analysis, lip sync and mixing remain planned.
