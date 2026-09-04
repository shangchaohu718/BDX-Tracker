"""Shared yaw utilities — single implementation for all evaluators
(audit: per-script yaw differencing caused the vyaw misdiagnosis and the
SCALE-era confusions; nobody re-implements angle math anymore)."""
import numpy as np
import torch


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a, b):
    """Smallest signed difference a-b, wrapped to (-pi, pi]."""
    return wrap_pi(np.asarray(a) - np.asarray(b))


def unwrap_yaw(yaw):
    """Sequential unwrap of a yaw sequence (crossing +-pi becomes continuous)."""
    yaw = np.asarray(yaw, dtype=np.float64)
    return yaw[:1].tolist() + (yaw[1:] + np.cumsum(np.round((yaw[:-1] - yaw[1:]) / (2*np.pi)) * 2*np.pi)).tolist()
