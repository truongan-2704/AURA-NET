"""
AURA-NET v3.0 - Adaptive Evidence Router (AER)
===============================================

Novel Component #3: Routes evidence regions to appropriate processors.

Key innovations:
1. Dynamic K selection based on image complexity
2. Gating mechanism (not softmax routing)
3. Three processors with different capacities (Easy/Medium/Hard)
4. Evidence-aware feature enhancement

Author: AURA-NET Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DifficultyEstimator(nn.Module):
    """
    Difficulty Estimator

    Estimates how difficult each region is to detect.
    Based on evidence score and feature statistics.
    """
    def __init__(self, feature_dim=256):
        super().__init__()

        self.difficulty_net = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),  # +1 for evidence score
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] difficulty score
        )

    def forward(self, region_features, evidence_scores):
        """
        Args:
            region_features: (B, K, feature_dim)
            evidence_scores: (B, K, 1)

        Returns:
            difficulty: (B, K, 1) - difficulty scores
        """
        # Concatenate features and evidence
        combined = torch.cat([region_features, evidence_scores], dim=-1)

        # Estimate difficulty
        difficulty = self.difficulty_net(combined)

        return difficulty


class LightweightProcessor(nn.Module):
    """
    Lightweight Processor for Easy Regions

    Uses fewer layers and smaller capacity.
    """
    def __init__(self, feature_dim=256):
        super().__init__()

        self.processor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        return self.processor(x)


class ModerateProcessor(nn.Module):
    """
    Moderate Processor for Medium Difficulty Regions

    Uses moderate capacity.
    """
    def __init__(self, feature_dim=256):
        super().__init__()

        self.processor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        return self.processor(x)


class HeavyProcessor(nn.Module):
    """
    Heavy Processor for Hard Regions

    Uses more layers and larger capacity.
    """
    def __init__(self, feature_dim=256):
        super().__init__()

        self.processor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.LayerNorm(feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim * 4),
            nn.LayerNorm(feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, x):
        return self.processor(x)


class GatingNetwork(nn.Module):
    """
    Gating Network

    Novel: Uses gating (not softmax) to route features.
    Multiple gates can be active simultaneously.
    """
    def __init__(self, feature_dim=256):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),  # +1 for difficulty
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 3),  # 3 gates: easy, medium, hard
            nn.Sigmoid()  # Independent gates
        )

    def forward(self, region_features, difficulty):
        """
        Args:
            region_features: (B, K, feature_dim)
            difficulty: (B, K, 1)

        Returns:
            gates: (B, K, 3) - gate values for [easy, medium, hard]
        """
        # Concatenate features and difficulty
        combined = torch.cat([region_features, difficulty], dim=-1)

        # Compute gates
        gates = self.gate_net(combined)

        return gates


class AdaptiveEvidenceRouter(nn.Module):
    """
    Adaptive Evidence Router (AER)

    Novel: Dynamically samples top-K evidence regions and routes
    them to appropriate processors based on difficulty.

    Key features:
    1. Dynamic K selection (100-300 based on complexity)
    2. Gating mechanism (not softmax)
    3. Three processors with different capacities
    4. Evidence-aware routing

    Args:
        feature_dim: Feature dimension (default: 256)
        min_K: Minimum number of regions (default: 100)
        max_K: Maximum number of regions (default: 300)
    """
    def __init__(self, feature_dim=256, min_K=100, max_K=300):
        super().__init__()

        self.feature_dim = feature_dim
        self.min_K = min_K
        self.max_K = max_K

        # Complexity estimator for dynamic K
        self.complexity_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Difficulty estimator
        self.difficulty_net = DifficultyEstimator(feature_dim)

        # Three processors
        self.easy_processor = LightweightProcessor(feature_dim)
        self.medium_processor = ModerateProcessor(feature_dim)
        self.hard_processor = HeavyProcessor(feature_dim)

        # Gating network
        self.gate_net = GatingNetwork(feature_dim)

    def select_K(self, evidence_map):
        """
        Dynamically select K based on image complexity.

        Args:
            evidence_map: (B, 1, H, W)

        Returns:
            K: int - number of regions to sample
        """
        B = evidence_map.size(0)

        # Estimate complexity from evidence distribution
        # High variance = complex image = more regions needed
        mean_evidence = evidence_map.mean(dim=[2, 3], keepdim=True)
        var_evidence = ((evidence_map - mean_evidence) ** 2).mean(dim=[2, 3])

        # Normalize variance to [0, 1]
        complexity = torch.clamp(var_evidence.mean(), 0, 1)

        # Map to K range
        K = int(self.min_K + complexity.item() * (self.max_K - self.min_K))

        return K

    def sample_top_k(self, evidence_map, K):
        """
        Sample top-K evidence regions.

        Args:
            evidence_map: (B, 1, H, W)
            K: int

        Returns:
            top_k_coords: (B, K, 2) - normalized coordinates [0, 1]
            top_k_scores: (B, K, 1) - evidence scores
        """
        B, _, H, W = evidence_map.shape

        # Flatten evidence map
        evidence_flat = evidence_map.view(B, -1)  # (B, H*W)

        # Get top-K indices
        top_k_scores, top_k_indices = torch.topk(evidence_flat, K, dim=1)  # (B, K)

        # Convert indices to coordinates
        top_k_y = (top_k_indices // W).float() / H  # Normalized [0, 1]
        top_k_x = (top_k_indices % W).float() / W

        top_k_coords = torch.stack([top_k_x, top_k_y], dim=-1)  # (B, K, 2)
        top_k_scores = top_k_scores.unsqueeze(-1)  # (B, K, 1)

        return top_k_coords, top_k_scores

    def extract_regions(self, features, coords):
        """
        Extract region features at given coordinates.

        Args:
            features: (B, C, H, W)
            coords: (B, K, 2) - normalized coordinates

        Returns:
            region_features: (B, K, C)
        """
        B, C, H, W = features.shape
        K = coords.size(1)

        # Convert normalized coords to grid_sample format [-1, 1]
        grid = coords * 2 - 1  # [0, 1] -> [-1, 1]
        grid = grid.unsqueeze(2)  # (B, K, 1, 2)

        # Sample features
        region_features = F.grid_sample(
            features,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (B, C, K, 1)

        region_features = region_features.squeeze(-1).permute(0, 2, 1)  # (B, K, C)

        return region_features

    def forward(self, features, evidence_map):
        """
        Args:
            features: (B, 256, 80, 80) from EPN
            evidence_map: (B, 1, 80, 80) from EPN

        Returns:
            output: (B, K, 256) - routed and processed features
            top_k_coords: (B, K, 2) - sampled coordinates
            top_k_scores: (B, K, 1) - evidence scores
            routing_info: dict - routing statistics
        """
        B, C, H, W = features.shape

        # Dynamic K selection
        K = self.select_K(evidence_map)

        # Sample top-K evidence regions
        top_k_coords, top_k_scores = self.sample_top_k(evidence_map, K)

        # Extract region features
        region_features = self.extract_regions(features, top_k_coords)

        # Estimate difficulty for each region
        difficulty = self.difficulty_net(region_features, top_k_scores)

        # Compute gates
        gates = self.gate_net(region_features, difficulty)  # (B, K, 3)

        # Process through all paths
        easy_out = self.easy_processor(region_features)
        medium_out = self.medium_processor(region_features)
        hard_out = self.hard_processor(region_features)

        # Gated fusion
        output = (easy_out * gates[..., 0:1] +
                  medium_out * gates[..., 1:2] +
                  hard_out * gates[..., 2:3])

        # Routing statistics
        routing_info = {
            'K': K,
            'difficulty_mean': difficulty.mean().item(),
            'difficulty_std': difficulty.std().item(),
            'easy_gate_mean': gates[..., 0].mean().item(),
            'medium_gate_mean': gates[..., 1].mean().item(),
            'hard_gate_mean': gates[..., 2].mean().item(),
        }

        return output, top_k_coords, top_k_scores, routing_info


if __name__ == '__main__':
    # Test
    model = AdaptiveEvidenceRouter(feature_dim=256, min_K=100, max_K=300)

    features = torch.randn(2, 256, 80, 80)
    evidence_map = torch.rand(2, 1, 80, 80)

    with torch.no_grad():
        output, coords, scores, routing_info = model(features, evidence_map)

    print(f"Features shape: {features.shape}")
    print(f"Evidence map shape: {evidence_map.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Coords shape: {coords.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"\nRouting info:")
    for k, v in routing_info.items():
        print(f"  {k}: {v}")

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {params / 1e6:.2f}M")
