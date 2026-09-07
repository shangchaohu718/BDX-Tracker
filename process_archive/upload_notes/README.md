# Upload notes: history surgery for GitHub (2026-09-07)

The BDX-Tracker GitHub upload required two history rewrites because the
original upstream repository's Git-LFS storage is missing objects
(server-side 404, unrecoverable anywhere):

1. `humanoidverse/data/lafan_29dof.zip` (151 MB): original LFS object could
   not be re-uploaded (S3 EOF on this link) and as a regular git blob it
   exceeds GitHub's 100 MB per-file limit (GH001) — REMOVED from history
   entirely and shipped as a GitHub Release asset instead
   (`lafan_29dof_release_copy.zip`, sha256
   197f918087c37bca690d33d090b42c54f8ce6968ab3eac13da11f0e1b57b62df).
   Place it back at `humanoidverse/data/lafan_29dof.zip` to reproduce the
   human-imitation pipeline.
2. `model/goal_inference/goal.mp4` (930 KB) and
   `model/reward_inference/move-ego-low0.6-0-0.7.mp4` (1.1 MB): original
   media lost upstream — LFS pointer blobs replaced with a one-line
   placeholder note. Sibling .gif versions of both demos survive in the same
   directories.

Consequence: commit SHAs on GitHub differ from the SHAs cited in the older
process records (STATE.md, commit messages, etc.). This directory preserves
the exact mapping:

- `commit-map` — old SHA -> new SHA (71 entries)
- `ref-map`, `changed-refs` — branch-level mapping

The pre-surgery history is preserved verbatim in the local bundle
`~/Desktop/start/BFM-zero-preupload-backup.bundle` (all refs, 182 MB).

GitHub pre-receive hook (GH008) rejects refs that reference LFS objects
absent from LFS storage — placeholders + export were the only way to land
the full history.
