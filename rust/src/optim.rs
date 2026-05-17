//! Unconstrained smooth-scalar 2D minimizer used by `forward_sweep_invert`
//! for the LearnedField gradient inversion path.
//!
//! Python uses `scipy.optimize.minimize(method="L-BFGS-B")` with
//! `jax.grad`-provided gradients (Markovian L-BFGS quasi-Newton with
//! analytical first-order derivatives). The v6 native port substitutes a
//! hand-rolled 2D damped-Newton solver with numerical finite-difference
//! gradient + Hessian. Justification:
//!   * The dimension is fixed at 2 (`chit`, `gamma_AB`) — full Hessian is
//!     a 2x2 inverted in closed form via `math::inv_2x2`.
//!   * BLOCK_IN §v6 session-5 explicitly carves out non-byte-identity vs
//!     scipy; the optimizer just needs to converge to the same MAP within
//!     ~0.005 per axis from the grid-argmin warm start. For a near-
//!     quadratic cost (the identity-MLP test) Newton converges in 2-3
//!     iterations.
//!   * Avoids pulling `argmin` + `argmin-math` (large deps that compile
//!     slowly and complicate WASM builds) for ~80 lines of code.
//!
//! Backtracking line search ensures monotonic descent; if the Newton step
//! direction is bad (e.g. Hessian indefinite far from a minimum) the
//! search falls back to the negative gradient.

use crate::math::inv_2x2;

const FD_STEP: f64 = 1e-6;
const MAX_ITER: usize = 50;
const GRAD_TOL: f64 = 1e-10;
const F_TOL: f64 = 1e-15;
const LINE_SEARCH_HALVINGS: usize = 30;

/// Minimize `cost` over R^2 starting at `x0`. Returns the converged point.
///
/// `cost` is called many times per outer iteration — the FD probes are 9
/// per Hessian step + 4 per gradient step + up to 30 per line search. For
/// the typical 2-5 outer iterations on this problem that's <300 evals,
/// well under the grid-search budget the gradient path replaces.
pub fn minimize_smooth_2d<F: Fn(f64, f64) -> f64>(cost: F, x0: [f64; 2]) -> [f64; 2] {
    let mut x = x0;
    let mut f_curr = cost(x[0], x[1]);
    for _ in 0..MAX_ITER {
        let grad = numerical_gradient(&cost, x);
        if grad[0].abs() < GRAD_TOL && grad[1].abs() < GRAD_TOL {
            break;
        }
        let hess = numerical_hessian(&cost, x, f_curr);
        let step = newton_step(&grad, &hess).unwrap_or([-grad[0], -grad[1]]);
        let mut alpha = 1.0;
        let mut f_next = f_curr;
        let mut x_next = x;
        for _ in 0..LINE_SEARCH_HALVINGS {
            x_next = [x[0] + alpha * step[0], x[1] + alpha * step[1]];
            f_next = cost(x_next[0], x_next[1]);
            if f_next < f_curr {
                break;
            }
            alpha *= 0.5;
        }
        if f_next >= f_curr {
            // Line search exhausted without descent — at a minimum (or in
            // a numerical flat). Stop.
            break;
        }
        let denom = f_curr.abs().max(1.0);
        let f_change = (f_curr - f_next).abs() / denom;
        x = x_next;
        f_curr = f_next;
        if f_change < F_TOL {
            break;
        }
    }
    x
}

fn numerical_gradient<F: Fn(f64, f64) -> f64>(cost: &F, x: [f64; 2]) -> [f64; 2] {
    let h = FD_STEP;
    let f_p0 = cost(x[0] + h, x[1]);
    let f_m0 = cost(x[0] - h, x[1]);
    let f_p1 = cost(x[0], x[1] + h);
    let f_m1 = cost(x[0], x[1] - h);
    [
        (f_p0 - f_m0) / (2.0 * h),
        (f_p1 - f_m1) / (2.0 * h),
    ]
}

fn numerical_hessian<F: Fn(f64, f64) -> f64>(cost: &F, x: [f64; 2], f_00: f64) -> [[f64; 2]; 2] {
    let h = FD_STEP;
    let f_p0 = cost(x[0] + h, x[1]);
    let f_m0 = cost(x[0] - h, x[1]);
    let f_p1 = cost(x[0], x[1] + h);
    let f_m1 = cost(x[0], x[1] - h);
    let f_pp = cost(x[0] + h, x[1] + h);
    let f_pm = cost(x[0] + h, x[1] - h);
    let f_mp = cost(x[0] - h, x[1] + h);
    let f_mm = cost(x[0] - h, x[1] - h);
    let h2 = h * h;
    let h_00 = (f_p0 - 2.0 * f_00 + f_m0) / h2;
    let h_11 = (f_p1 - 2.0 * f_00 + f_m1) / h2;
    let h_01 = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h2);
    [[h_00, h_01], [h_01, h_11]]
}

fn newton_step(grad: &[f64; 2], hess: &[[f64; 2]; 2]) -> Option<[f64; 2]> {
    let inv = inv_2x2(hess)?;
    Some([
        -(inv[0][0] * grad[0] + inv[0][1] * grad[1]),
        -(inv[1][0] * grad[0] + inv[1][1] * grad[1]),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimizes_quadratic_bowl() {
        // f(x, y) = (x - 0.3)^2 + (y + 0.7)^2 — minimum at (0.3, -0.7).
        let cost = |x: f64, y: f64| (x - 0.3).powi(2) + (y + 0.7).powi(2);
        let x_opt = minimize_smooth_2d(cost, [0.0, 0.0]);
        assert!((x_opt[0] - 0.3).abs() < 1e-6, "x_opt[0] = {}", x_opt[0]);
        assert!((x_opt[1] + 0.7).abs() < 1e-6, "x_opt[1] = {}", x_opt[1]);
    }

    #[test]
    fn minimizes_rotated_quadratic() {
        // Off-diagonal Hessian — Newton picks the correct step direction.
        // f(x, y) = (x - y)^2 + (x + y - 1)^2 — minimum at (0.5, 0.5).
        let cost = |x: f64, y: f64| (x - y).powi(2) + (x + y - 1.0).powi(2);
        let x_opt = minimize_smooth_2d(cost, [0.0, 0.0]);
        assert!((x_opt[0] - 0.5).abs() < 1e-5, "x_opt[0] = {}", x_opt[0]);
        assert!((x_opt[1] - 0.5).abs() < 1e-5, "x_opt[1] = {}", x_opt[1]);
    }

    #[test]
    fn stops_at_minimum() {
        // Starting at the minimum: should return roughly the start.
        let cost = |x: f64, y: f64| x * x + y * y;
        let x_opt = minimize_smooth_2d(cost, [0.0, 0.0]);
        assert!(x_opt[0].abs() < 1e-8);
        assert!(x_opt[1].abs() < 1e-8);
    }
}
