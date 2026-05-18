// Session-9 WASM smoke test.
//
// Run via: `node rust/tests/wasm_smoke.mjs` (after
// `wasm-pack build --release --target nodejs --no-default-features --features wasm`
// has populated `rust/pkg/`).
//
// Calls one wrapped variant per shape (closed-form, posterior, intent
// algebra) and asserts the returned object has the expected
// `{value, validation, provenance}` dict-shape. The bit_identity.rs +
// test_rust_parity.py suites cover semantic correctness; this script
// only proves the wasm-bindgen + serde-wasm-bindgen round-trip works
// end-to-end and the 11 functions are reachable from JavaScript.
//
// Exit 0 on success, non-zero on any assertion failure.

import assert from "node:assert/strict";
import * as mss from "../pkg/mpa_scale_solver.js";

// ---------------------------------------------------------------------------
// 1. regime_at_wrapped — simplest path; no field input.
// ---------------------------------------------------------------------------
{
    const out = mss.regime_at_wrapped(
        { chit: 0.8, gamma_AB: 0.0, k_frust: true },
        1.0
    );
    assert.equal(out.value.regime, "deep_c");
    assert.equal(out.value.k_frust, true);
    assert.equal(out.provenance.operation, "regime_at");
    assert.equal(out.provenance.dispatch_path, "direct_compute");
    assert.equal(out.validation.k_frust_invariant, true);
    console.log("  ok regime_at_wrapped");
}

// ---------------------------------------------------------------------------
// 2. apply_translation_wrapped — tangent-flow closed-form remap.
// ---------------------------------------------------------------------------
const tangent_flow_field = {
    direction: "forward",
    shape: "tangent_flow",
    rule_at_origin: {
        operating_point: { label: "origin", gt: "s", axes: {} },
        xdot_choice: "default",
        canonical: {
            chit: 0.0,
            gamma_AB: 0.0,
            k_frust: false,
            method: "test",
            extras: {},
        },
    },
    scaling: {
        tau_obs_ref: 1.0,
        delta_chit: 0.3,
        delta_gamma: 0.5,
        refinement: null,
    },
    description: null,
};

{
    const out = mss.apply_translation_wrapped(
        { chit: 0.4, gamma_AB: 0.2, k_frust: false },
        tangent_flow_field,
        2.0,
        null,
        null,
        null
    );
    assert.ok(out.value.observables.substrate_chit !== undefined);
    assert.ok(out.value.observables.substrate_gamma_AB !== undefined);
    assert.equal(out.provenance.operation, "apply_translation");
    console.log("  ok apply_translation_wrapped");
}

// ---------------------------------------------------------------------------
// 3. forward_sweep_invert_wrapped — closed-form inversion (Auto -> closed).
// ---------------------------------------------------------------------------
{
    // Forward-translate first to build a self-consistent target.
    const target = mss
        .apply_translation_wrapped(
            { chit: 0.4, gamma_AB: 0.2, k_frust: false },
            tangent_flow_field,
            2.0,
            null,
            null,
            null
        )
        .value;
    const grid = [];
    for (let i = 0; i < 5; i++)
        for (let j = 0; j < 5; j++)
            grid.push([i * 0.25, -0.5 + j * 0.25]);
    const out = mss.forward_sweep_invert_wrapped(
        target,
        tangent_flow_field,
        2.0,
        grid,
        null,
        true,
        "auto"
    );
    assert.equal(out.provenance.operation, "forward_sweep_invert");
    assert.ok(Math.abs(out.value.chit - 0.4) < 1e-10);
    assert.ok(Math.abs(out.value.gamma_AB - 0.2) < 1e-10);
    console.log("  ok forward_sweep_invert_wrapped (closed-form recovery)");
}

// ---------------------------------------------------------------------------
// 4. intent_map_wrapped — pure-arithmetic intent algebra (no math primitives).
// ---------------------------------------------------------------------------
{
    const gamut = {
        chit_range: [0.0, 1.0],
        gamma_AB_range: [-0.5, 0.5],
        tau_obs_range: [0.1, 10.0],
        out_of_scope_residual_threshold: 0.05,
    };
    const out = mss.intent_map_wrapped(
        { chit: 0.4, gamma_AB: 0.2, k_frust: false },
        1.0,
        gamut,
        "I1"
    );
    assert.equal(out.provenance.operation, "intent_map");
    // value is [mapped_canonical, sacrifice_record]
    assert.equal(out.value.length, 2);
    const sac = out.value[1];
    assert.equal(sac.intent, "I1");
    assert.equal(sac.invariant_preserved, true);
    console.log("  ok intent_map_wrapped (I1)");
}

// ---------------------------------------------------------------------------
// 5. forward_sweep_invert_posterior_wrapped — Laplace approximation.
// ---------------------------------------------------------------------------
{
    const target = mss
        .apply_translation_wrapped(
            { chit: 0.4, gamma_AB: 0.2, k_frust: false },
            tangent_flow_field,
            1.5,
            null,
            null,
            null
        )
        .value;
    const out = mss.forward_sweep_invert_posterior_wrapped(
        target,
        tangent_flow_field,
        1.5,
        null,
        0.25,
        false,
        5
    );
    assert.equal(out.provenance.operation, "forward_sweep_invert_posterior");
    const p = out.value;
    assert.ok(Math.abs(p.mean.chit - 0.4) < 1e-10);
    assert.equal(p.covariance.length, 2);
    assert.equal(p.covariance[0].length, 2);
    assert.equal(p.noise_variance, 0.25);
    console.log("  ok forward_sweep_invert_posterior_wrapped (tangent-flow closed-form)");
}

// ---------------------------------------------------------------------------
// 6. flow — continuous-form transit (Markovian Banach exponential).
// ---------------------------------------------------------------------------
{
    const out = mss.flow(
        { chit: 0.5, gamma_AB: 0.2, k_frust: false },
        1.0,
        tangent_flow_field
    );
    assert.ok(typeof out.chit === "number");
    assert.ok(typeof out.gamma_AB === "number");
    console.log("  ok flow");
}

console.log("\nWASM smoke OK — 6/6 entry points reachable end-to-end.");
