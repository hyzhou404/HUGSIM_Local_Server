from logging import raiseExceptions
import numpy as np
import torch
import pdb
from ..utils import geometry_utils as GeoUtils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random

from .forward_sampler import ForwardSampler

STATE_INDEX = [0, 1, 2, 4]

device = "cuda" if torch.cuda.is_available() else "cpu"


def mean_control_effort_coefficients(x0, dx0, xf, dxf):
    """Returns `(c4, c3, c2)` corresponding to `c4 * tf**-4 + c3 * tf**-3 + c2 * tf**-2`."""
    return (12 * (x0 - xf) ** 2, 12 * (dx0 + dxf) * (x0 - xf), 4 * dx0 ** 2 + 4 * dx0 * dxf + 4 * dxf ** 2)


def cubic_spline_coefficients(x0, dx0, xf, dxf, tf):
    return (x0, dx0, -2 * dx0 / tf - dxf / tf - 3 * x0 / tf ** 2 + 3 * xf / tf ** 2,
            dx0 / tf ** 2 + dxf / tf ** 2 + 2 * x0 / tf ** 3 - 2 * xf / tf ** 3)


def compute_interpolating_spline(state_0, state_f, tf):
    dx0, dy0 = state_0[..., 2] * \
               torch.cos(state_0[..., 3]), state_0[..., 2] * \
               torch.sin(state_0[..., 3])
    dxf, dyf = state_f[..., 2] * \
               torch.cos(state_f[..., 3]), state_f[..., 2] * \
               torch.sin(state_f[..., 3])
    tf = tf * torch.ones_like(state_0[..., 0])
    return (
        torch.stack(cubic_spline_coefficients(
            state_0[..., 0], dx0, state_f[..., 0], dxf, tf), -1),
        torch.stack(cubic_spline_coefficients(
            state_0[..., 1], dy0, state_f[..., 1], dyf, tf), -1),
        tf,
    )


def compute_spline_xyvaqrt(x_coefficients, y_coefficients, tf, N=10):
    t = torch.arange(N).unsqueeze(0).to(tf.device) * tf.unsqueeze(-1) / (N - 1)
    tp = t[..., None] ** torch.arange(4).to(tf.device)
    dtp = t[..., None] ** torch.tensor([0, 0, 1, 2]
                                       ).to(tf.device) * torch.arange(4).to(tf.device)
    ddtp = t[..., None] ** torch.tensor([0, 0, 0, 1]).to(
        tf.device) * torch.tensor([0, 0, 2, 6]).to(tf.device)
    x_coefficients = x_coefficients.unsqueeze(-1)
    y_coefficients = y_coefficients.unsqueeze(-1)
    vx = dtp @ x_coefficients
    vy = dtp @ y_coefficients
    v = torch.hypot(vx, vy)
    v_pos = torch.clip(v, min=1e-4)
    ax = ddtp @ x_coefficients
    ay = ddtp @ y_coefficients
    a = (ax * vx + ay * vy) / v_pos
    r = (-ax * vy + ay * vx) / (v_pos ** 2)
    yaw = torch.atan2(vy, vx)
    return torch.cat((
        tp @ x_coefficients,
        tp @ y_coefficients,
        v,
        a,
        yaw,
        r,
        t.unsqueeze(-1),
    ), -1)


def patch_yaw_low_speed(traj):
    idx = traj[...,]


def mean_control_effort_coefficients(x0, dx0, xf, dxf):
    """Returns `(c4, c3, c2)` corresponding to `c4 * tf**-4 + c3 * tf**-3 + c2 * tf**-2`."""
    return (12 * (x0 - xf) ** 2, 12 * (dx0 + dxf) * (x0 - xf), 4 * dx0 ** 2 + 4 * dx0 * dxf + 4 * dxf ** 2)


class SplinePlanner(object):
    def __init__(self, device, dx_grid=None, dy_grid=None, acce_grid=None, dyaw_grid=None, max_steer=0.5, max_rvel=8,
                 acce_bound=[-6, 4], vbound=[-2.0, 30], spline_order=3, N_seg=10, low_speed_threshold=2.0, seed=0):
        self.spline_order = spline_order
        self.device = device
        assert spline_order == 3
        if dx_grid is None:
            # self.dx_grid = torch.tensor([-4., 0, 4.]).to(self.device)
            self.dx_grid = torch.tensor([0.]).to(self.device)
        else:
            self.dx_grid = torch.tensor(dx_grid).to(self.device)
        if dy_grid is None:
            self.dy_grid = torch.tensor([-3., -1.5, 0, 1.5, 3.]).to(self.device)
        else:
            self.dy_grid = torch.tensor(dy_grid).to(self.device)
        self.dy_grid_lane = torch.tensor([-2., 0, 2., ]).to(self.device)
        if acce_grid is None:
            # self.acce_grid = torch.tensor([-1., -0.5, 0., 0.5, 1.]).to(self.device)
            self.acce_grid = torch.tensor([-1., 0., 1.]).to(self.device)
        else:
            self.acce_grid = torch.tensor(acce_grid).to(self.device)
        if dyaw_grid is None:
            self.dyaw_grid = torch.tensor(
                [-np.pi / 6, 0, np.pi / 6]).to(self.device)
        else:
            self.dyaw_grid = torch.tensor(dyaw_grid).to(self.device)
        self.max_steer = max_steer
        self.max_rvel = max_rvel
        self.psi_bound = [-np.pi * 0.75, np.pi * 0.75]
        self.acce_bound = acce_bound
        self.vbound = vbound
        self.N_seg = N_seg
        self.low_speed_threshold = low_speed_threshold
        self.forward_sampler = ForwardSampler(acce_grid=self.acce_grid, dhm_grid=torch.linspace(-0.7, 0.7, 9),
                                              dhf_grid=[-0.4, 0, 0.4], dt=0.1, device=self.device)
        torch.manual_seed(seed)

    def calc_trajectories(self, x0, tf, xf, N=None):
        if N is None:
            N = self.N_seg
        if x0.ndim == 1:
            x0_tile = x0.tile(xf.shape[0], 1)
            xc, yc, tf = compute_interpolating_spline(x0_tile, xf, tf)
        elif x0.ndim == xf.ndim:
            xc, yc, tf = compute_interpolating_spline(x0, xf, tf)
        else:
            raise ValueError("wrong dimension for x0")
        traj = compute_spline_xyvaqrt(xc, yc, tf, N)
        return traj

    def gen_terminals_lane(self, x0, tf, lanes):
        if lanes is None or len(lanes) == 0:
            return self.gen_terminals(x0, tf)

        gs = [self.dx_grid.shape[0], self.dy_grid_lane.shape[0], self.acce_grid.shape[0]]
        dx = self.dx_grid[:, None, None, None].repeat(1, gs[1], gs[2], 1).flatten()
        dy = self.dy_grid_lane[None, :, None, None].repeat(gs[0], 1, gs[2], 1).flatten()
        dv = self.acce_grid[None, None, :, None].repeat(
            gs[0], gs[1], 1, 1).flatten() * tf
        xf = list()
        if x0.ndim == 1:
            for lane in lanes:
                f, p_start = lane
                if isinstance(p_start, np.ndarray):
                    p_start = torch.from_numpy(p_start).to(x0.device)
                elif isinstance(p_start, torch.Tensor):
                    p_start = p_start.to(x0.device)
                offset = x0[:2] - p_start[:2]
                s_offset = offset[0] * \
                           torch.cos(p_start[2]) + offset[1] * torch.sin(p_start[2])
                ds = dx + dv / 2 * tf + x0[2:3] * tf
                ss = ds + s_offset
                xyyaw = torch.from_numpy(f(ss.cpu().numpy())).type(
                    torch.float).to(x0.device)
                xyyaw[..., 1] += dy
                xf.append(
                    torch.cat((xyyaw[:, :2], dv.reshape(-1, 1) + x0[2:3], xyyaw[:, 2:]), -1))
            # adding the end points not fixated on lane
            xf_straight = torch.stack([ds, dy, dv + x0[2], x0[3].tile(ds.shape[0])], -1)
            xf.append(xf_straight)
        elif x0.ndim == 2:
            for lane in lanes:
                f, p_start = lane
                if isinstance(p_start, np.ndarray):
                    p_start = torch.from_numpy(p_start).to(x0.device)
                elif isinstance(p_start, torch.Tensor):
                    p_start = p_start.to(x0.device)
                offset = x0[:, :2] - p_start[None, :2]
                s_offset = offset[:, 0] * torch.cos(p_start[2]) + offset[:, 1] * torch.sin(p_start[2])

                ds = (dx + dv / 2 * tf).unsqueeze(0) + x0[:, 2:3] * tf
                ss = ds + s_offset.unsqueeze(-1)
                xyyaw = torch.from_numpy(f(ss.cpu().numpy())).type(
                    torch.float).to(x0.device)
                xyyaw[..., 1] += dy
                xf.append(torch.cat((xyyaw[..., :2], dv.tile(
                    x0.shape[0], 1).unsqueeze(-1) + x0[:, None, 2:3], xyyaw[..., 2:]), -1))
            # adding the end points not fixated on lane
            xf_straight = torch.stack([ds, dy.tile(x0.shape[0], 1), dv.tile(x0.shape[0], 1) + x0[:, None, 2],
                                       x0[:, None, 3].tile(1, ds.shape[1])], -1)
            xf.append(xf_straight)
        else:
            raise ValueError("x0 must have dimension 1 or 2")
        xf = torch.cat(xf, -2)
        return xf

    def gen_terminals(self, x0, tf):
        gs = [self.dx_grid.shape[0], self.dy_grid.shape[0],
              self.acce_grid.shape[0], self.dyaw_grid.shape[0]]
        dx = self.dx_grid[:, None, None, None].repeat(
            1, gs[1], gs[2], gs[3]).flatten()
        dy = self.dy_grid[None, :, None, None].repeat(
            gs[0], 1, gs[2], gs[3]).flatten()
        dv = tf * self.acce_grid[None, None, :,
                  None].repeat(gs[0], gs[1], 1, gs[3]).flatten()
        dyaw = self.dyaw_grid[None, None, None, :].repeat(
            gs[0], gs[1], gs[2], 1).flatten()
        delta_x = torch.stack([dx, dy, dv, dyaw], -1)

        if x0.ndim == 1:
            xy = torch.cat(
                (delta_x[:, 0:1] + delta_x[:, 2:3] / 2 * tf + x0[2:3] * tf, delta_x[:, 1:2]), -1)
            rotated_delta_xy = GeoUtils.batch_rotate_2D(xy, x0[3])
            refpsi = torch.arctan2(rotated_delta_xy[..., 1], rotated_delta_xy[..., 0])
            rotated_xy = rotated_delta_xy + x0[:2]
            return torch.cat((rotated_xy, delta_x[:, 2:3] + x0[2:3], delta_x[:, 3:] + refpsi.unsqueeze(-1)), -1)
        elif x0.ndim == 2:
            delta_x = torch.tile(delta_x, [x0.shape[0], 1, 1])
            xy = torch.cat(
                (delta_x[:, :, 0:1] + delta_x[:, :, 2:3] / 2 * tf + x0[:, None, 2:3] * tf, delta_x[:, :, 1:2]), -1)
            rotated_delta_xy = GeoUtils.batch_rotate_2D(xy, x0[:, 3:4])
            refpsi = torch.arctan2(rotated_delta_xy[..., 1], rotated_delta_xy[..., 0])
            rotated_xy = rotated_delta_xy + x0[:, None, :2]
            return torch.cat(
                (rotated_xy, delta_x[:, :, 2:3] + x0[:, None, 2:3], delta_x[:, :, 3:] + refpsi.unsqueeze(-1)), -1)
        else:
            raise ValueError("x0 must have dimension 1 or 2")

    def feasible_flag(self, traj, xf):
        diff = traj[..., -1, STATE_INDEX] - xf

        feas_flag = ((traj[..., 2] >= self.vbound[0]) & (traj[..., 2] < self.vbound[1]) &
                     (traj[..., 4] >= self.psi_bound[0]) & (traj[..., 4] < self.psi_bound[1]) &
                     (traj[..., 3] >= self.acce_bound[0]) & (traj[..., 3] <= self.acce_bound[1]) &
                     (torch.abs(traj[..., 5] * traj[..., 2]) <= self.max_rvel) & (
                             torch.clip(torch.abs(traj[..., 2]), min=0.5) * self.max_steer >= torch.abs(
                         traj[..., 5]))).all(1) & (
                            diff.abs() < 5e-3).all(-1)

        return feas_flag

    def gen_trajectories(self, x0, tf, lanes=None, dyn_filter=True, N=None, lane_only=False):
        if N is None:
            N = self.N_seg
        if lanes is not None:
            if isinstance(lanes, torch.Tensor):
                lanes = lanes.cpu().numpy()
            lane_interp = [GeoUtils.interp_lanes(lane) for lane in lanes]
            xf_lane = self.gen_terminals_lane(
                x0, tf, lane_interp)
        else:
            xf_lane = None
        if lane_only:
            assert xf_lane is not None
            xf_set = xf_lane
        else:
            xf_set = self.gen_terminals(x0, tf)
            if xf_lane is not None:
                xf_set = torch.cat((xf_lane, xf_set), 0)
        x0[..., 2] = torch.clip(x0[..., 2], min=1e-3)
        xf_set[..., 2] = torch.clip(xf_set[..., 2], min=1e-3)

        # x, y, v, a, yaw,r, t
        traj = self.calc_trajectories(x0, tf, xf_set, N)
        if dyn_filter:
            feas_flag = self.feasible_flag(traj, xf_set)
            traj = traj[feas_flag]
            xf = xf_set[feas_flag]
        traj = traj[..., 1:, :]  # remove the first time step
        if x0[2] < self.low_speed_threshold:
            # call forward sampler when velocity is low
            extra_traj = self.forward_sampler.sample_trajs(x0.unsqueeze(0), int(tf / self.forward_sampler.dt)).squeeze(
                0)
            f = interp1d(np.arange(1, extra_traj.shape[-2] + 1) * self.forward_sampler.dt, extra_traj.cpu().numpy(),
                         axis=-2)
            extra_traj = torch.from_numpy(f(np.arange(1, N) * tf / N)).to(self.device)
            traj = torch.cat((traj, extra_traj), 0)
        return traj, traj[..., -1, STATE_INDEX]

    @staticmethod
    def get_similarity_flag(x0, x1, thres=[2.0, 0.5, 2.0, np.pi / 12]):
        thres = torch.tensor(thres, device=x0.device)
        diff = x0.unsqueeze(-3) - x1.unsqueeze(-2)
        flag = diff.abs() < thres
        flag = flag.all(-1).any(-1)
        return flag

    def gen_terminals_hardcoded(self, x0_set, tf):
        X0, Y0, v0, psi0 = x0_set[..., 0:1], x0_set[..., 1:2], x0_set[..., 2:3], x0_set[..., 3:]
        xf_set = list()
        # drive straight
        xf_straight = torch.cat((X0 + v0 * tf * torch.cos(psi0), Y0 + v0 * tf * torch.sin(psi0), v0, psi0),
                                -1).unsqueeze(1)
        xf_set.append(xf_straight)
        # hard brake
        decel = torch.clip(-v0 / tf, min=self.acce_bound[0])
        xf_brake = torch.cat((X0 + (v0 + decel * 0.5 * tf) * tf * torch.cos(psi0),
                              Y0 + (v0 + decel * 0.5 * tf) * tf * torch.sin(psi0), v0 + decel * tf, psi0),
                             -1).unsqueeze(1)
        xf_set.append(xf_brake)
        xf_set = torch.cat(xf_set, 1)
        return xf_set

    def gen_trajectory_batch(self, x0_set, tf, lanes=None, dyn_filter=True, N=None, max_children=None):
        if N is None:
            N = self.N_seg
        device = x0_set.device
        xf_set_sample = self.gen_terminals(x0_set, tf)
        importance_score = torch.rand(xf_set_sample.shape[:2], device=device)
        xf_set_hardcoded = self.gen_terminals_hardcoded(x0_set, tf)
        xf_set = torch.cat((xf_set_sample, xf_set_hardcoded), 1)
        importance_score = torch.cat((importance_score, 2 * torch.ones(xf_set_hardcoded.shape[:2], device=device)), 1)
        if lanes is not None:
            lane_interp = [GeoUtils.interp_lanes(lane) for lane in lanes]
            xf_set_lane = self.gen_terminals_lane(x0_set, tf, lane_interp)
            xf_set = torch.cat((xf_set, xf_set_lane), -2)
            importance_score = torch.cat((importance_score, torch.ones(xf_set_lane.shape[:2], device=x0_set.device)), 1)
        x0_set[..., 2] = torch.clip(x0_set[..., 2], min=1e-3)
        xf_set[..., 2] = torch.clip(xf_set[..., 2], min=1e-3)
        num_node = x0_set.shape[0]
        num = xf_set.shape[1]
        x0_tiled = x0_set.repeat_interleave(num, 0)
        xf_tiled = xf_set.reshape(-1, xf_set.shape[-1])
        traj = self.calc_trajectories(x0_tiled, tf, xf_tiled, N)
        if dyn_filter:
            feas_flag = self.feasible_flag(traj, xf_tiled)
        else:
            feas_flag = torch.ones(
                num * num_node, dtype=torch.bool).to(x0_set.device)
        feas_flag = feas_flag.reshape(num_node, num)
        traj = traj.reshape(num_node, num, *traj.shape[1:])
        if (x0_set[:, 2] < self.low_speed_threshold).any():
            extra_traj = self.forward_sampler.sample_trajs(x0_set, int(tf / self.forward_sampler.dt))
            f = interp1d(np.arange(1, extra_traj.shape[-2] + 1) * self.forward_sampler.dt, extra_traj.cpu().numpy(),
                         axis=-2, bounds_error=False, fill_value="extrapolate")
            extra_traj = torch.from_numpy(f(np.arange(0, N) * tf / N)).to(self.device)
            traj = torch.cat((traj, extra_traj), 1)
            extra_importance_score = torch.rand(extra_traj.shape[:2], device=device)
            importance_score = torch.cat((importance_score, extra_importance_score), 1)
            feas_flag = torch.cat((feas_flag, torch.ones(extra_traj.shape[:2], device=device)), 1)

        importance_score = importance_score * feas_flag
        chosen_idx = [torch.where(importance_score[i])[0].tolist() for i in range(num_node)]
        if max_children is not None:
            chosen_idx = [idx if len(idx) <= max_children else torch.topk(importance_score[i], max_children)[1] for
                          i, idx in enumerate(chosen_idx)]
        traj_batch = [traj[i, chosen_idx[i], 1:] for i in range(num_node)]
        return traj_batch

    def gen_trajectory_tree(self, x0, tf, n_layers, dyn_filter=True, N=None):
        if N is None:
            N = self.N_seg
        trajs = list()
        nodes = [x0[None, :]]
        for i in range(n_layers):
            xf = self.gen_terminals(nodes[i], tf)
            x0i = torch.tile(nodes[i], [xf.shape[1], 1])
            xf = xf.reshape(-1, xf.shape[-1])

            traj = self.calc_trajectories(x0i, tf, xf, N)
            if dyn_filter:
                feas_flag = self.feasible_flag(traj, xf)
                traj = traj[feas_flag]
                xf = xf[feas_flag]

            trajs.append(traj)

            nodes.append(xf.reshape(-1, xf.shape[-1]))
        return trajs, nodes[1:]


if __name__ == "__main__":
    planner = SplinePlanner("cuda")
    x0 = torch.tensor([1., 2., 1., 0.]).cuda()
    tf = 5
    traj, xf = planner.gen_trajectories(x0, tf)
    trajs = planner.gen_trajectory_batch(xf, tf)

    # x, y, v, a, yaw,r, t = traj
    msize = 12
    trajs, nodes = planner.gen_trajectory_tree(x0, tf, 2)
    x0 = x0.cpu().numpy()
    traj = traj.cpu().numpy()
    plt.figure(figsize=(20, 10))
    plt.plot(x0[0], x0[1], marker="o", color="b", markersize=msize)
    for node, traj in zip(nodes, trajs):
        node = node.cpu().numpy()
        traj = traj.cpu().numpy()
        x = traj[..., 0]
        y = traj[..., 1]
        plt.plot(x.T, y.T, color="k")
        for p in node:
            plt.plot(p[0], p[1], marker="o", color="b", markersize=msize)
    plt.show()
