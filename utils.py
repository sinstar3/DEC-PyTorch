import torch
import torch.nn as nn


class SoftClusterAssignment(nn.Module):
    """Soft cluster assignment layer: computes soft assignment probabilities using Student's t-distribution.

    Args:
        num_cluster: Number of clusters
        hidden_dim: Dimension of feature vectors
        alpha: Degrees of freedom parameter of the t-distribution, controls sharpness
        centroid: Optional initial cluster centers; randomly initialized if None
    """
    def __init__(
        self,
        num_cluster: int,
        hidden_dim: int,
        alpha: float = 1.0,
        centroid: torch.Tensor = None,
    ):
        super(SoftClusterAssignment, self).__init__()
        self.num_cluster = num_cluster
        self.hidden_dim = hidden_dim
        self.alpha = alpha

        if centroid is None:
            initial_centroid = torch.zeros(num_cluster, hidden_dim, dtype=torch.float)
            nn.init.normal_(initial_centroid, mean=0.0, std=0.01)
        else:
            initial_centroid = centroid.detach().clone()  # avoid affecting external tensor

        # Must be registered as a Parameter to be updated by the optimizer
        self.centroid = nn.Parameter(initial_centroid)

    def forward(self, z):
        """Forward pass: compute soft assignment matrix Q

        q_ij = (1 + ||z_i - c_j||^2 / alpha)^(-(alpha+1)/2) / sum_j(...)
        """
        z = z.to(self.centroid.device)
        # Compute squared distances (batch, num_cluster)
        diff = torch.sum((z.unsqueeze(1) - self.centroid) ** 2, dim=2)
        # Compute t-distribution similarity
        numerator = (1.0 + diff / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q