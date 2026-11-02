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

`audio_analysis.analyze_wav()` measures bounded PCM16 WAV bytes (mono/stereo; 16/24/44.1/48 kHz; RIFF `fmt16` + `data` only), pinning content, settings, picture timing and actual tool identity. Sample peak/RMS energy use dBFS; [FFmpeg loudnorm](https://ffmpeg.org/ffmpeg-filters.html#loudnorm) input statistics provide integrated LUFS and oversampled dBTP under [EBU R128](https://tech.ebu.ch/docs/r/r128.pdf), with explicit undefined/unavailable states and no delivery preset.

Energy checks use frame-aligned windows and declared tick tolerances; partial ends round outward by less than one tick, while duration comparisons stay exact. They recognize no words or visible lip sync; qualitative review remains unavailable, and the default unit suite uses fake tools (`tests_native/` requires an explicitly available FFmpeg and generated synthetic fixtures).

Measured true peak strictly above `0 dBTP` adds `true_peak_overload`, separately from PCM sample-risk counts; exactly zero is not overload, and delivery targets remain M5.4. Process finalization attempts group termination, bounded reap and every descriptor close independently; OS refusal or cleanup failure raises a sanitized error with `cleanup_uncertain=True` and leaves analysis explicitly unavailable, without claiming all descendants were killed/reaped.

`speech.QwenBaseSpeechAdapter` translates exact English/Mandarin lines to pinned Qwen Base request data using explicit canonical voice bindings and current script/picture pins; neutral/default delivery with no pronunciation hints is the supported baseline. Its default runtime returns deterministic silence fixtures; explicitly injected Python runtimes are trusted code. `native_speech.NativeSpeechRuntime` separately opts into a pinned local Linux child and provisional owned receipts; remote native setup/invocation remains separate gated work.

`stem_import.import_original_stem` validates and returns original non-dialogue PCM16 WAV bytes unchanged, with exact rational cue timing and current reference/copy rights. Trusted calling code must select the exact existing rights-review evidence as its original-source assertion; matching metadata cannot prove original/synthetic origin or grant human review/export approval.

`s2v_service.S2VServiceContext` binds a separately checked one-lease RTX 5090 scope (150 GB ephemeral disk, USD 5 ceiling, 120-second cleanup reserve) before speech exists; it grants no graph execution. After guarded hardware/native-speech verification, `s2v.VerifiedS2V` requires current EN/ZH script, character/voice/picture/rights, original RGB PNG and mono PCM16 speech bytes, plus trusted operator proof callbacks; descriptors and JSON flags cannot prove likeness or native execution. Proof responses and nested leaves are strictly revalidated, then pure current rights/metadata checks run after all proof and authorization callbacks.

`SSHWorker.run_guarded_native` calls trusted controller-local code once as `invoke(timeout, absolute_deadline=cap)`; that code forwards the original same-host cap to native execution and owns cleanup. It provides guard admission, not remote transport; the remote bridge must derive its own worker-clock cap.

The exact 14-node Wan2.2 core recipe uses four pinned assets (FP8-scaled diffusion storage; BF16 compute), 832×480, 77 frames at 16 fps, one front-facing fictional adult and one language per clip. Hash-named PNG/WAV pairs transfer through a guarded, bounded once-only protocol; uncertain setup, transfer or submission requires inspecting retained private state before a new request, with no blind retry.

`s2v_run.run_s2v` uses an already prepared/staged owned worker and records original speech timing, lineage, actual output bytes and restricted local VP9 decoding; the silent video stays paired with unmodified speech for later mixing. Offline fixtures and native synthetic-video decoding do not prove model quality, intelligibility, identity or visible lip sync; exact-media human review and export authority remain separate. The original attempt cap includes admission, transport, inspection and publication; late work retains incomplete state, known output identity and no review readiness.

The S2V journal keeps `generated` evidence and `technical_candidate` with `human_review: unavailable`; it intentionally differs from a successful returned record. Only the returned copy becomes `pending_human_review` after real cleanup, fresh admission and the final cap check; retained bytes need safe offline revalidation, never paid regeneration or approval from a stored flag.

`audio_mix.mix_scene` mixes at most 16 exact speech/import cues (32 MiB aggregate; 0.4–60 s), with sample-aligned offsets/fades, stem gains and dialogue peak-driven ambience/music ducking. Attack/hold/release use scene ticks and linear gain; mono duplicates to explicit stereo, while resampling/downmixing/trimming/looping refuse. Actual speech may end before its reserved cue; uncovered scene frames are silence, and current copy/reference/modification rights are checked before native calls and again before return.

Presets measure whole delivered PCM through pinned FFmpeg 9.0.2: `broadcast-linear` targets −23 LUFS / maximum −1 dBTP; project-defined `internal-review-stereo` targets −16 / −2, with ±0.2 LU tolerance. Explicit summed-sample safety attenuation and a linear target gain precede actual post-measurement; a 0.05 dB predicted peak margin is required, and excessive crest factors, undefined/short/silent or unavailable measurements refuse. These are PCM targets, not full broadcaster/codec certification or a universal streaming standard.

Results bind bytes, source metadata, admission snapshot, script/character/picture dependencies, processing and actual tool identity; measured timeout budgets may vary with elapsed work while media bytes remain deterministic. Fake meters return `synthetic_fixture_only`; native technical results grant no human listening/visual approval or export authority.

Reuse reviewed upstream models/workflows only after pinned model/weights/dependencies, rights/voice consent, hardware and budget approval, before any download, upload, paid test or voice cloning. Default CI stays offline; native speech generation and talking-video execution remain separate approval-gated work.

`audio_review.snapshot_for_review()` hashes supplied delivery/mix bytes and paired `en`/`zh` clips, binding script, cue, character, voice/reference and recipe identities. `assess_review()` requires separate current owner decisions plus an injected trusted technical verifier; metadata authenticates no person and grants no export authority.

Checklist: English/Mandarin intelligibility and visible lip sync, shared actor/voice identity, music, artifacts, transitions and balance. Actual owner listening/visual review remains pending until exact M5.6 media exists.

`native_remote.RemoteNativeSpeechRuntime` stages only a complete reviewed 108-wheel runtime, separate curated SDK tree, pinned models and verified first-party package on an already guarded M5 worker; selection requires separately reviewed actual installed manifests. The full runtime tree retains exact SymPy/Numba/Ruff data paths as `0600` bytes, vendored metadata/records and `.pth` bytes; only root dist-info counts toward exact 108-distribution identity, and `-I -B -S` never processes `.pth` startup. Every transfer/invocation shares the original controller cap and a fixed worker-local cap with the lease cleanup reserve; lost acknowledgements consume the attempt, retain uncertainty and require owned Pod cleanup.

Bridge-owned provisional proofs bind completed request/reference/output bytes and native cleanup facts; they remain inspectable offline after Pod deletion and grant no new work, rights, human review or export authority. Staging uses the existing five-second clock bound conservatively; network socket denial allows local UNIX IPC and does not claim a hostile-code sandbox or all-descendant termination proof.

Only the seven pinned Base/Design configuration paths receive strict model JSON checks through an explicit matching model role; runtime/SDK/source resources remain fully hashed opaque data.
