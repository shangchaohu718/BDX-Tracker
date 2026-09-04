import torch, json, numpy as np
import tracking_bdx.config.bdx_v4  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from dataclasses import asdict
T='Mjlab-Tracking-Flat-BDX-V4-Mixed-Pool-134-V2'
cfg = load_env_cfg(T, play=True); cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=cfg, device='cpu')
agent_cfg = load_rl_cfg(T)
from mjlab.rl import RslRlVecEnvWrapper
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(T)
runner = runner_cls(env, asdict(agent_cfg), device='cpu')
runner.load('logs/rsl_rl/bdx_v4_tracking/2026-08-27_21-03-24_mixed134_v2_ne1024/model_1999.pt', load_cfg={'actor': True}, strict=True, map_location='cpu')
policy = runner.get_inference_policy(device='cpu')
mc = env.unwrapped.command_manager.get_term("motion")
obs, _ = env.reset()
mc.set_motion_index(100)
def _pin(ids): mc.set_motion_index(100)
mc._assign_motions = _pin
qs, cmds = [], []
with torch.inference_mode():
    for t in range(220):
        qs.append(mc.robot.data.joint_pos[0].cpu().numpy().copy())
        cmds.append(mc.joint_pos[0].cpu().numpy().copy())
        obs, _, _, _ = env.step(policy(obs))
q, c = np.array(qs)[25:], np.array(cmds)[25:]
ref_amp, rob_amp = c.max(0)-c.min(0), q.max(0)-q.min(0)
ratio = rob_amp/np.maximum(ref_amp,1e-6)
cors=[]
for j in range(14):
    a,b = q[:,j]-q[:,j].mean(), c[:,j]-c[:,j].mean()
    d = np.sqrt((a*a).sum()*(b*b).sum()); cors.append(round(float((a*b).sum()/d),3) if d>1e-9 else None)
mv = ref_amp>0.15
print('RESULT ref_amp :', np.round(ref_amp,2).tolist())
print('RESULT rob_amp :', np.round(rob_amp,2).tolist())
print('RESULT ratio   :', np.round(ratio,2).tolist())
print('RESULT corr    :', cors)
print('RESULT jerr_last50:', round(float(np.abs(q[-50:]-c[-50:]).mean()),3))
print('RESULT moving joints:', int(mv.sum()), 'mean ratio:', round(float(ratio[mv].mean()),3))
