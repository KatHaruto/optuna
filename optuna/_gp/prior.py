from __future__ import annotations

from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    import torch

    from optuna._gp import gp
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")
import optuna.logging
_logger = optuna.logging.get_logger(__name__)


DEFAULT_MINIMUM_NOISE_VAR = 1e-6


def dim_scaled_log_normal_prior(gpr: gp.GPRegressor) -> torch.Tensor:
    # Log of prior distribution of kernel parameters.

    def gamma_log_prior(x: torch.Tensor, concentration: float, rate: float) -> torch.Tensor:
        # We omit the constant factor `rate ** concentration / Gamma(concentration)`.
        return concentration * math.log(rate) - math.log(math.gamma(concentration))  + (concentration - 1) * torch.log(x) - rate * x
    
    def log_lognormal_prior_on_inv_sq_lengthscales(z: torch.Tensor) -> torch.Tensor:
        """
        Log-prior on inverse squared lengthscales z = 1 / ℓ²,
        where ℓ ~ LogNormal(μ, σ), and μ = log(√D) + √2, σ = √3.
        Then, z ~ LogNormal(-2μ, 4σ²).
        """
        D = z.shape[0]
        _logger.info(f"dim_scaled_log_normal_prior: D = {D}")
        mu_ell = math.log(math.sqrt(D)) + math.sqrt(2.0)
        sigma_ell = math.sqrt(3.0)

        mu_z = -2 * mu_ell
        sigma_z = 2 * sigma_ell

        logz = torch.log(z)
        # Log PDF of LogNormal
        log_prob = -((logz - mu_z) ** 2) / (2 * sigma_z**2) - logz
        log_prob = log_prob.sum() - D * (math.log(sigma_z) + 0.5 * math.log(2 * math.pi))
        return log_prob
    
    # TODO(contramundum53): Check whether these priors are appropriate.
    return (
        log_lognormal_prior_on_inv_sq_lengthscales(gpr.inverse_squared_lengthscales)
        + gamma_log_prior(gpr.kernel_scale, 2, 1)
        + gamma_log_prior(gpr.noise_var, 1.1, 30)
    )

def default_log_prior(gpr: gp.GPRegressor) -> torch.Tensor:
    # Log of prior distribution of kernel parameters.

    def gamma_log_prior(x: torch.Tensor, concentration: float, rate: float) -> torch.Tensor:
        # We omit the constant factor `rate ** concentration / Gamma(concentration)`.
        return (concentration - 1) * torch.log(x) - rate * x

    # NOTE(contramundum53): The priors below (params and function
    # shape for inverse_squared_lengthscales) were picked by heuristics.
    # TODO(contramundum53): Check whether these priors are appropriate.
    return (
        -(0.1 / gpr.inverse_squared_lengthscales + 0.1 * gpr.inverse_squared_lengthscales).sum()
        + gamma_log_prior(gpr.kernel_scale, 2, 1)
        + gamma_log_prior(gpr.noise_var, 1.1, 30)
    )
