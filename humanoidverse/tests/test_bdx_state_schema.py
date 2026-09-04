"""Unit test for the BDX state schema (Gate 0, external ruling 2026-08-24).

Synthetic perturbation test: perturb ONLY one block at a time and assert
that only the expected split responds; activity uses dof_vel only.
"""

import sys
import torch

sys.path.insert(0, ".")

from humanoidverse.utils.bdx_state import bdx_motion_activity, split_bdx_state


def test_splits_respond_only_to_their_block():
    base = torch.zeros(3, 34)
    dp, dv, gr, av = split_bdx_state(base)
    assert dp.shape == (3, 14) and dv.shape == (3, 14)
    assert gr.shape == (3, 3) and av.shape == (3, 3)

    s = base.clone()
    s[..., 5] = 1.0  # dof_pos channel
    dp2, dv2, gr2, av2 = split_bdx_state(s)
    assert (dp2 - dp).abs().max() > 0 and (dv2 - dv).abs().max() == 0
    assert (gr2 - gr).abs().max() == 0 and (av2 - av).abs().max() == 0

    s = base.clone()
    s[..., 20] = 1.0  # dof_vel channel
    dp2, dv2, gr2, av2 = split_bdx_state(s)
    assert (dv2 - dv).abs().max() > 0 and (dp2 - dp).abs().max() == 0

    s = base.clone()
    s[..., 29] = 1.0  # gravity channel
    _, _, gr2, _ = split_bdx_state(s)
    assert (gr2 - gr).abs().max() > 0

    s = base.clone()
    s[..., 32] = 1.0  # base_ang_vel channel
    _, _, _, av2 = split_bdx_state(s)
    assert (av2 - av).abs().max() > 0


def test_activity_ignores_non_dof_vel_blocks():
    a = torch.zeros(1, 34)
    assert bdx_motion_activity(a).item() == 0.0
    b = a.clone()
    b[..., 0:14] = 5.0   # dof_pos should not count
    c = a.clone()
    c[..., 28:34] = 5.0  # gravity + ang_vel should not count
    d = a.clone()
    d[..., 14:28] = 3.0  # dof_vel of 3 everywhere -> norm = sqrt(14*9)
    assert bdx_motion_activity(b).item() == 0.0
    assert bdx_motion_activity(c).item() == 0.0
    assert abs(bdx_motion_activity(d).item() - (14 * 9) ** 0.5) < 1e-5


if __name__ == "__main__":
    test_splits_respond_only_to_their_block()
    test_activity_ignores_non_dof_vel_blocks()
    print("bdx_state schema tests: OK")


def test_history_builder_matches_handler_contract():
    """Bit-exact comparison against a faithful replica of HistoryHandler
    (add: newest at index 0, older shift right; query[:, :4] newest-first)
    composed with _get_obs_long_history's flatten order."""
    import torch as T
    T_frames = 8
    obs = T.arange(T_frames * 34, dtype=T.float32).view(T_frames, 34)
    act = (T.arange(T_frames * 14, dtype=T.float32) + 1000).view(T_frames, 14)

    # faithful handler replica (per-key rolling buffer, newest-first)
    keys = {"actions": 14, "base_ang_vel": 3, "dof_pos": 14, "dof_vel": 14, "projected_gravity": 3}
    hist = {k: T.zeros(1, 4, d) for k, d in keys.items()}
    t = 5
    for step in range(t + 1):
        hist["actions"][:, 1:] = hist["actions"][:, :-1].clone()
        hist["actions"][:, 0] = act[step]
        for k, sl in (("base_ang_vel", slice(31, 34)), ("dof_pos", slice(0, 14)),
                      ("dof_vel", slice(14, 28)), ("projected_gravity", slice(28, 31))):
            hist[k][:, 1:] = hist[k][:, :-1].clone()
            hist[k][:, 0] = obs[step][sl]
    expected = T.cat([hist[k].reshape(-1) for k in sorted(keys)])

    from humanoidverse.utils.bdx_state import build_bdx_actor_history
    built = build_bdx_actor_history(obs, act, t)
    assert built.shape == (192,)
    assert T.allclose(built, expected), f"history mismatch: {(built - expected).abs().max()}"
    try:
        build_bdx_actor_history(obs, act, 2)
        raise AssertionError("t=2 should be rejected (no full prefix)")
    except AssertionError as e:
        assert "prefix" in str(e)


if __name__ == "__main__":
    test_history_builder_matches_handler_contract()
    print("history contract test: OK")


def test_history_builder_vs_real_handler():
    """End-to-end against the REAL HistoryHandler + production flatten order."""
    import torch as T
    from humanoidverse.envs.env_utils.history_handler import HistoryHandler
    from humanoidverse.utils.bdx_state import build_bdx_actor_history

    # production obs_auxiliary config for history_actor (bdx_zero_obs.yaml)
    cfg = {"base_ang_vel": 4, "projected_gravity": 4, "dof_pos": 4, "dof_vel": 4, "actions": 4}
    dims = {"base_ang_vel": 3, "projected_gravity": 3, "dof_pos": 14, "dof_vel": 14, "actions": 14}
    h = HistoryHandler(1, {"history_actor": cfg}, dims, "cpu")
    T_frames = 8
    obs = T.arange(T_frames * 34, dtype=T.float32).view(T_frames, 34)
    act = (T.arange(T_frames * 14, dtype=T.float32) + 1000).view(T_frames, 14)
    t = 5
    for step in range(t + 1):
        h.add("base_ang_vel", obs[step][31:34].unsqueeze(0))
        h.add("projected_gravity", obs[step][28:31].unsqueeze(0))
        h.add("dof_pos", obs[step][0:14].unsqueeze(0))
        h.add("dof_vel", obs[step][14:28].unsqueeze(0))
        h.add("actions", act[step].unsqueeze(0))
    # production _get_obs_long_history: sorted(keys), each query[:, :L].reshape(N, -1)
    parts = []
    for key in sorted(cfg):
        hh = h.query(key)[:, :4].reshape(1, -1)
        parts.append(hh)
    expected = T.cat(parts, dim=-1).flatten()
    built = build_bdx_actor_history(obs, act, t)
    assert built.shape == expected.shape == (192,)
    assert T.allclose(built, expected), f"max diff {(built - expected).abs().max()}"


if __name__ == "__main__":
    test_history_builder_vs_real_handler()
    print("real-handler contract test: OK")
