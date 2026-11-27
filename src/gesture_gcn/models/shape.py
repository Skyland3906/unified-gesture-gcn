"""Relative Shape Branch: local normalized contour/swept-area grids + geometric descriptors.

Extracts a complementary spatial-shape representation directly from the window-normalized
skeleton coordinates `x[:, :3, :, :]` (no raw pixels or frontend protocol changes required):
  1. Soft-binned local GxG relative grids (4 channels):
     - Ch 0: Per-frame hand contour occupancy (translation & scale invariant silhouette)
     - Ch 1: Per-frame fingertip contour occupancy (finger-spread silhouette)
     - Ch 2: Window swept-trajectory density (horizontal/vertical stripe, ring, dot)
     - Ch 3: Signed Motion History Grid (temporal gradient t in [-1, 1] preserving direction)
  2. Explicit 20-D relative geometry vector combining:
     - Axis-aligned aspect ratios & signed displacement (distinguishes H/V & L/R/U/D)
     - Rotation-invariant 2D PCA elongation, compactness/solidity proxy, fingertip spread,
       scale/area delta, trajectory hollowness, tortuosity, signed winding angle & consistency.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from gesture_gcn.data.skeleton import HandSkeletonSpec


def _resolve_key_indices(spec: HandSkeletonSpec) -> tuple[int, int, tuple[int, ...]]:
    """Return (root_idx, palm_anchor_idx, fingertip_indices) for supported hand topologies."""
    if spec.num_joints == 21:
        return spec.root_joint, 9, (4, 8, 12, 16, 20)
    if spec.num_joints == 22:
        return spec.root_joint, 1, (5, 9, 13, 17, 21)
    # Generic fallback for custom topologies: leaf nodes as fingertips, mid joint as anchor
    parent_set = set(spec.parents)
    leaves = tuple(i for i in range(spec.num_joints) if i not in parent_set)
    tips = leaves if leaves else (spec.num_joints - 1,)
    palm_anchor = min(1, spec.num_joints - 1)
    return spec.root_joint, palm_anchor, tips


def _pca_2d_eigenvalues(
    pts: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form 2D covariance eigenvalues (lam1 >= lam2 >= 0), 100% TorchScript-safe.

    Args:
        pts: Centered 2D points of shape (..., 2, K).

    Returns:
        (lam1, lam2) each of shape (...,).
    """
    x = pts[..., 0, :]
    y = pts[..., 1, :]
    a = (x * x).mean(dim=-1)
    c = (y * y).mean(dim=-1)
    b = (x * y).mean(dim=-1)
    trace = a + c
    disc = torch.sqrt((a - c).square() + 4.0 * b.square() + eps * eps)
    lam1 = 0.5 * (trace + disc)
    lam2 = (0.5 * (trace - disc)).clamp(min=0.0)
    return lam1, lam2


class RelativeShapeBranch(nn.Module):
    """Lightweight relative contour & swept-area branch fused via residual gating."""

    GEOM_DIM: int = 21

    def __init__(
        self,
        spec: HandSkeletonSpec,
        out_channels: int,
        grid_size: int = 8,
        hidden_channels: int = 32,
        sigma: float = 0.28,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.sigma = sigma
        root_idx, palm_idx, tip_indices = _resolve_key_indices(spec)
        self.root_idx = root_idx
        self.palm_idx = palm_idx

        self.register_buffer(
            "tip_indices",
            torch.tensor(tip_indices, dtype=torch.long),
            persistent=False,
        )
        centers = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32)
        self.register_buffer("grid_centers", centers, persistent=False)

        # Border-vs-core mask for grid ring hollowness descriptor
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        rad = torch.sqrt(xx * xx + yy * yy)
        core_mask = (rad <= 0.48).to(torch.float32)
        ring_mask = ((rad > 0.48) & (rad <= 1.15)).to(torch.float32)
        self.register_buffer(
            "core_mask",
            core_mask / core_mask.sum().clamp(min=1.0),
            persistent=False,
        )
        self.register_buffer(
            "ring_mask",
            ring_mask / ring_mask.sum().clamp(min=1.0),
            persistent=False,
        )

        # Lightweight 2-layer 2D CNN for the 4-channel GxG relative shape grid
        mid_ch = max(16, hidden_channels // 2)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(4, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Compact MLP for explicit axis-aligned + rotation-invariant + directional descriptors
        self.geom_encoder = nn.Sequential(
            nn.LayerNorm(self.GEOM_DIM),
            nn.Linear(self.GEOM_DIM, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
        )

        # Project fused shape representation to match GCN pooled feature width
        self.proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

        # Sample-adaptive residual gate: lets the network trust GCN on fine poses and boost shape on trajectories
        self.gate = nn.Linear(out_channels * 2, out_channels)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -0.5)

    def _soft_splat(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vectorized Gaussian soft-binning onto a (B, G, G) grid via matmul.

        Args:
            u: Normalized X coordinates in [-1, 1], shape (B, M).
            v: Normalized Y coordinates in [-1, 1], shape (B, M).
            weights: Optional per-point weights of shape (B, M) or (1, M).

        Returns:
            Grid tensor of shape (B, G, G).
        """
        centers = self.grid_centers  # (G,)
        inv_2s2 = -0.5 / (self.sigma * self.sigma)
        wx = torch.exp((u.clamp(-1.1, 1.1).unsqueeze(-1) - centers).square() * inv_2s2)  # (B, M, G)
        wy = torch.exp((v.clamp(-1.1, 1.1).unsqueeze(-1) - centers).square() * inv_2s2)  # (B, M, G)
        if weights is not None:
            wy = wy * weights.unsqueeze(-1)
        # (B, G, M) @ (B, M, G) -> (B, G, G), row=Y, col=X
        return torch.bmm(wy.transpose(1, 2), wx)

    def extract_shape_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract 4-channel relative shape grid (B, 4, G, G) and 20-D geometry vector (B, 20).

        Args:
            x: Preprocessed skeleton features (B, C_in, T, N), where x[:, :3] are
               isotropically window-normalized 3D coordinates.
        """
        eps = 1e-5
        bsz, _, t_len, num_joints = x.shape
        xy = x[:, :2, :, :]  # (B, 2, T, N)

        # ------------------------------------------------------------------
        # 1. Per-frame hand contour (translation & scale invariant per frame)
        # ------------------------------------------------------------------
        frame_center = xy.mean(dim=3, keepdim=True)  # (B, 2, T, 1)
        rel_pose = xy - frame_center                 # (B, 2, T, N)

        pose_span_x = rel_pose[:, 0].amax(dim=-1) - rel_pose[:, 0].amin(dim=-1)  # (B, T)
        pose_span_y = rel_pose[:, 1].amax(dim=-1) - rel_pose[:, 1].amin(dim=-1)  # (B, T)
        pose_scale = torch.maximum(pose_span_x, pose_span_y).clamp(min=0.08)     # (B, T)

        # Normalized into [-1, 1] preserving hand contour aspect ratio
        pose_u = (2.0 * rel_pose[:, 0] / pose_scale.unsqueeze(-1)).reshape(bsz, t_len * num_joints)
        pose_v = (2.0 * rel_pose[:, 1] / pose_scale.unsqueeze(-1)).reshape(bsz, t_len * num_joints)
        contour_grid = self._soft_splat(pose_u, pose_v) / float(t_len * num_joints)

        # Fingertip-only contour grid in the same hand-local normalized frame
        tips_rel = rel_pose.index_select(dim=3, index=self.tip_indices)  # (B, 2, T, F)
        num_tips = tips_rel.shape[3]
        tip_u = (2.0 * tips_rel[:, 0] / pose_scale.unsqueeze(-1)).reshape(bsz, t_len * num_tips)
        tip_v = (2.0 * tips_rel[:, 1] / pose_scale.unsqueeze(-1)).reshape(bsz, t_len * num_tips)
        tip_grid = self._soft_splat(tip_u, tip_v) / float(t_len * num_tips)

        # Rotation-invariant per-frame hand shape descriptors
        # Permute rel_pose to (B, T, 2, N) for closed-form 2D PCA eigenvalues
        lam1_p, lam2_p = _pca_2d_eigenvalues(rel_pose.permute(0, 2, 1, 3), eps=eps)  # (B, T)
        pose_elong = torch.sqrt(lam2_p / (lam1_p + eps))                              # (B, T)
        pose_rms = torch.sqrt((lam1_p + lam2_p).clamp(min=eps))                       # (B, T)
        pose_area = torch.sqrt((lam1_p * lam2_p).clamp(min=eps))                      # (B, T)

        joint_r = torch.sqrt((rel_pose * rel_pose).sum(dim=1) + eps)  # (B, T, N)
        max_joint_r = joint_r.amax(dim=-1).clamp(min=0.05)            # (B, T)
        # Compactness / solidity proxy: compact fist -> high, splayed palm -> lower
        solidity_proxy = pose_area / (max_joint_r * max_joint_r + eps)

        # Rigid palm scale (wrist -> middle MCP / palm): constant across finger curl/open, rotation-invariant
        palm_vec = xy[:, :, :, self.palm_idx] - xy[:, :, :, self.root_idx]  # (B, 2, T)
        palm_scale = torch.sqrt((palm_vec * palm_vec).sum(dim=1) + eps).clamp(min=0.05)  # (B, T)

        # Fingertip radial openness & dispersion relative to wrist normalized by rigid palm scale
        tips_from_root = xy.index_select(dim=3, index=self.tip_indices) - xy[:, :, :, self.root_idx : self.root_idx + 1]
        tip_r = torch.sqrt((tips_from_root * tips_from_root).sum(dim=1) + eps)  # (B, T, F)
        tip_compact_t = tip_r.mean(dim=-1) / palm_scale                          # (B, T)
        tip_disp_t = tip_r.std(dim=-1, unbiased=False) / palm_scale              # (B, T)

        # Pairwise fingertip spread normalized by rigid palm scale (rotation-invariant openness)
        tip_diff = tips_rel.unsqueeze(4) - tips_rel.unsqueeze(3)                 # (B, 2, T, F, F)
        tip_pair_dist = torch.sqrt((tip_diff * tip_diff).sum(dim=1) + eps).mean(dim=(2, 3))  # (B, T)
        tip_spread_t = tip_pair_dist / palm_scale                                # (B, T)

        # Start vs end segments (first quarter vs last quarter for smooth noise-resistant deltas)
        q = max(1, t_len // 4)
        pose_aspect = torch.log((pose_span_x + 0.02) / (pose_span_y + 0.02)).mean(dim=1)
        pose_elong_mean = pose_elong.mean(dim=1)
        scale_log_delta = torch.log(
            (pose_rms[:, -q:].mean(dim=1) + 0.02) / (pose_rms[:, :q].mean(dim=1) + 0.02)
        )
        tip_compact_mean = tip_compact_t.mean(dim=1)
        tip_compact_delta = tip_compact_t[:, -q:].mean(dim=1) - tip_compact_t[:, :q].mean(dim=1)
        tip_disp_mean = tip_disp_t.mean(dim=1)
        tip_spread_mean = tip_spread_t.mean(dim=1)
        solidity_mean = solidity_proxy.mean(dim=1)

        # ------------------------------------------------------------------
        # 2. Window-level swept trajectory region (density + signed MHI)
        # ------------------------------------------------------------------
        # Palm center trajectory (B, 2, T)
        palm_traj = 0.5 * (xy[:, :, :, self.root_idx] + xy[:, :, :, self.palm_idx])
        traj_center = palm_traj.mean(dim=2, keepdim=True)  # (B, 2, 1)
        rel_traj = palm_traj - traj_center                 # (B, 2, T)

        traj_span_x = rel_traj[:, 0].amax(dim=-1) - rel_traj[:, 0].amin(dim=-1)  # (B,)
        traj_span_y = rel_traj[:, 1].amax(dim=-1) - rel_traj[:, 1].amin(dim=-1)  # (B,)
        # Floor keeps stationary hand centered as a dot rather than amplifying jitter to fill the grid
        traj_scale = torch.maximum(traj_span_x, traj_span_y).clamp(min=0.18)     # (B,)

        traj_u = 2.0 * rel_traj[:, 0] / traj_scale.unsqueeze(-1)  # (B, T)
        traj_v = 2.0 * rel_traj[:, 1] / traj_scale.unsqueeze(-1)  # (B, T)

        swept_density = self._soft_splat(traj_u, traj_v) / float(t_len)  # (B, G, G)
        # Signed temporal progression in [-1, 1]: encodes start -> end direction inside the grid
        t_prog = torch.linspace(-1.0, 1.0, t_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        swept_mhi = self._soft_splat(traj_u, traj_v, weights=t_prog) / (
            swept_density * float(t_len) + 0.25
        )

        # Normalize positive occupancy grids to comparable [0, 1] scale per sample
        contour_grid = contour_grid / (contour_grid.amax(dim=(1, 2), keepdim=True) + eps)
        tip_grid = tip_grid / (tip_grid.amax(dim=(1, 2), keepdim=True) + eps)
        swept_grid = swept_density / (swept_density.amax(dim=(1, 2), keepdim=True) + eps)

        grids = torch.stack([contour_grid, tip_grid, swept_grid, swept_mhi], dim=1)  # (B, 4, G, G)

        # Grid-derived occupancy ratio & ring hollowness (outer ring density minus inner core density)
        grid_occ_ratio = torch.sigmoid((swept_grid - 0.20) * 15.0).mean(dim=(1, 2))
        core_energy = (swept_grid * self.core_mask).sum(dim=(1, 2))
        ring_energy = (swept_grid * self.ring_mask).sum(dim=(1, 2))
        grid_hollowness = ring_energy - core_energy

        # Axis-aligned + rotation-invariant + directional swept trajectory descriptors
        traj_aspect = torch.log((traj_span_x + 0.04) / (traj_span_y + 0.04))
        lam1_t, lam2_t = _pca_2d_eigenvalues(rel_traj, eps=eps)
        traj_elong = torch.sqrt(lam2_t / (lam1_t + eps))

        # Step lengths, path length, net displacement, inverse tortuosity
        step_diff = rel_traj[:, :, 1:] - rel_traj[:, :, :-1]                  # (B, 2, T-1)
        path_len = torch.sqrt((step_diff * step_diff).sum(dim=1) + eps).sum(dim=1)  # (B,)
        net_vec = rel_traj[:, :, -1] - rel_traj[:, :, 0]                      # (B, 2)
        net_disp = torch.sqrt((net_vec * net_vec).sum(dim=1) + eps)           # (B,)
        inv_tortuosity = net_disp / (path_len + 0.05)

        # Rotation-invariant radial hollowness of trajectory around its centroid
        traj_r = torch.sqrt((rel_traj * rel_traj).sum(dim=1) + eps)           # (B, T)
        traj_r_mean = traj_r.mean(dim=1)
        traj_r_max = traj_r.amax(dim=1).clamp(min=0.04)
        traj_r_std = traj_r.std(dim=1, unbiased=False)
        radial_ring_ratio = traj_r_mean / (traj_r_max + 0.04)                 # high for circle
        radial_cv = traj_r_std / (traj_r_mean + 0.04)                         # low for circle

        # Signed winding angle & winding consistency (rotation-invariant + CW/CCW preserving)
        cross = rel_traj[:, 0, :-1] * rel_traj[:, 1, 1:] - rel_traj[:, 1, :-1] * rel_traj[:, 0, 1:]
        dot = rel_traj[:, 0, :-1] * rel_traj[:, 0, 1:] + rel_traj[:, 1, :-1] * rel_traj[:, 1, 1:]
        # Soft gate tiny radii so static noise at centroid doesn't spin atan2 wildly
        r_gate = torch.sigmoid((torch.minimum(traj_r[:, :-1], traj_r[:, 1:]) - 0.04) * 40.0)
        dtheta = torch.atan2(cross, dot + eps) * r_gate
        net_angle = dtheta.sum(dim=1)
        signed_winding = (net_angle / (2.0 * math.pi)).clamp(-2.0, 2.0)
        winding_consistency = net_angle.abs() / (dtheta.abs().sum(dim=1) + 0.25)

        geom = torch.stack(
            [
                # Static hand contour (8 dims)
                pose_aspect.clamp(-3.0, 3.0),
                pose_elong_mean,
                scale_log_delta.clamp(-2.0, 2.0),
                tip_compact_mean.clamp(0.0, 4.0),
                tip_compact_delta.clamp(-2.0, 2.0),
                tip_disp_mean.clamp(0.0, 3.0),
                tip_spread_mean.clamp(0.0, 4.0),
                solidity_mean.clamp(0.0, 2.0),
                # Dynamic swept region (12 dims)
                traj_aspect.clamp(-3.0, 3.0),
                traj_span_x.clamp(0.0, 3.0),
                traj_span_y.clamp(0.0, 3.0),
                traj_elong,
                inv_tortuosity.clamp(0.0, 1.5),
                net_vec[:, 0].clamp(-2.0, 2.0),
                net_vec[:, 1].clamp(-2.0, 2.0),
                radial_ring_ratio.clamp(0.0, 1.5),
                radial_cv.clamp(0.0, 2.0),
                signed_winding,
                winding_consistency.clamp(0.0, 1.0),
                grid_occ_ratio.clamp(0.0, 1.0),
                grid_hollowness.clamp(-1.0, 1.0),
            ],
            dim=-1,
        )
        return grids, geom

    def forward(self, x: torch.Tensor, gcn_pooled: torch.Tensor) -> torch.Tensor:
        """Compute gated residual fusion of relative shape features onto GCN pooled vector."""
        grids, geom = self.extract_shape_features(x)
        grid_emb = self.grid_encoder(grids)
        geom_emb = self.geom_encoder(geom)
        shape_feat = self.proj(torch.cat([grid_emb, geom_emb], dim=-1))
        alpha = torch.sigmoid(self.gate(torch.cat([gcn_pooled, shape_feat], dim=-1)))
        return gcn_pooled + alpha * shape_feat
