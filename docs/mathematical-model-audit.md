# Mathematical Model Audit

Date: 2026-08-18

## Sources Checked

Local PhD memory and vault notes point to:

- `/home/charl/Dev/PhD/phd-assistant/PhD-Vault/PhD Knowledge/Concepts/Monge Patch Track Modelling.md`
- `/home/charl/Dev/PhD/phd-assistant/PhD-Vault/PhD Knowledge/Code/MongePatchTrackModelling Codebase.md`
- `/home/charl/Dev/PhD/phd-assistant/PhD-Vault/PhD Knowledge/Code/StackelbergGameMonge Codebase.md`
- `/home/charl/Dev/PhD/WitsMotorsportAndControl/StackelbergGameMonge/StackelbergMongeDAE.m`
- `/home/charl/Dev/PhD/WitsMotorsportAndControl/StackelbergGameMonge/QualifyingPlots.m`
- `/home/charl/Dev/PhD/WitsMotorsportAndControl/MongePatchTrackModelling/getRoadCoords_Monge.m`

The relevant model is a vehicle trajectory on a Monge-patch road surface. The
Python project should therefore be treated as a post-processing visualiser of
MATLAB optimiser output.

## What The Python Code Gets Mostly Right

### State Ordering

The Python HUD assumes:

```text
[n, xi, v, omega_Bz, u, delta, k_f, k_r, t]
```

This matches the MATLAB setup comments in `QualifyingMain.m` and the
post-processing in `QualifyingPlots.m`.

### 2D Road Coordinate Mapping

The Python HUD computes:

```python
x = xc - n * sin(psi)
y = yc + n * cos(psi)
```

This matches `getRoadCoords_Monge.m`, which uses:

```matlab
xi = x_r - sin(psi).*n;
yi = y_r + cos(psi).*n;
zi = z0 + m;
```

It also matches the original MATLAB animation/post-processing scripts.

### Tyre-Force Formula Structure

The Python HUD uses the same broad normal-load interpolation and combined
slip-radius tyre-force form as the MATLAB `Fxy_explicit.m` and
`StackelbergMongeDAE.m`:

```text
kn = k / kmax
alphan = alpha / alphamax
rho = sqrt(kn^2 + alphan^2)
Fx = mux * Fz * kn / rho
Fy = muy * Fz * alphan / rho
```

The Python version protects against division by zero with machine epsilon,
which is sensible for plotting.

## Mathematical Issues Found And Fixed

### 1. Yaw Rate Was Unscaled The Wrong Way In The HUD

MATLAB post-processing unscales yaw rate as:

```matlab
omega_Bz = states(:,4) * timescale;
```

The Python HUD previously did:

```python
self.omega_BzL = self.statesL[:, 3] / self.timescale
self.omega_BzF = self.statesF[:, 3] / self.timescale
```

Given the MATLAB state scale

```matlab
statescale = [lengthscale 1 velscale 1/timescale velscale 1 1 1 timescale]
```

the Python must multiply by `timescale`, not divide by it.

This affects:

- numerical body-frame accelerations;
- direct body-frame accelerations adjusted by yaw terms;
- slip angles;
- tyre forces;
- G-G diagram data if using the computed accelerations;
- friction-circle plots.

Status: fixed in `animator_math.unscale_vehicle_states()` and used by
`Stackelberg_HUD.DataProcessor` and `Stackleberg_3DAnimator.Vehicle3DAnimatorGL`.
Covered by `tests/test_math_consistency.py`.

### 2. Wheelbase Distances Were Used In Scaled Units In Slip Angles

The MATLAB post-processing uses:

```matlab
alp_f = atan2(v + omega_Bz .* (a/lengthscale), u) - delta
alp_r = atan2(v - omega_Bz .* (b/lengthscale), u)
```

The Python HUD previously used:

```python
alp_f = atan2(v + omega_Bz * a, u) - delta
alp_r = atan2(v - omega_Bz * b, u)
```

The `auxdata.a` and `auxdata.b` values are scaled in MATLAB as
`physical_length * lengthscale`. They must be divided by `lengthscale` before
being combined with physical `u`, `v`, and `omega_Bz`.

This compounded the yaw-rate scaling error.

Status: fixed via `animator_math.physical_wheelbase()`. Slip angles and 2D/3D
vehicle footprints now use physical `a_m` and `b_m` values. Covered by
`tests/test_math_consistency.py`.

## Remaining Mathematical Caveats

### 3. The 3D Renderer Uses A Display Frame That Is Not The Model Frame

The HUD and MATLAB use model-space `y`. The 3D renderer still flips `y` and `z`
for OpenGL display, but this is now explicit:

```python
model_to_display_points(...)
display_to_model_points(...)
display_pose_from_model_pose(...)
```

This is a valid display choice as long as all OpenGL inputs go through the same
transform. The new regression tests cover round-tripping and HUD/renderer
coordinate agreement.

### 4. The 3D Car Pose Is A Visual Monge Pose, Not The Full Dynamic Attitude

The full MATLAB dynamics compute Monge derivatives, surface normal, first and
second fundamental forms, the Jacobian, body rates `omega_Bx`, `omega_By`, and
the heading geometry `chi`.

The Python 3D renderer now builds an orthonormal pose matrix from the Monge
surface normal and the vehicle heading projected onto the tangent plane. This
is much closer than the old yaw-plus-bank approximation and includes
longitudinal grade. It still does not reconstruct the full dynamic attitude
from all MATLAB body-rate quantities, because those are not exported as a
complete pose in the current visualiser inputs.

### 5. Interpolation Uses Linear Interpolation Where MATLAB Often Uses Spline

The Python code mostly uses `np.interp`, which is linear. Several MATLAB
post-processing/model routines use `interp1(..., 'spline')`.

For a visualiser, linear interpolation may be acceptable, but it can cause small
heading, surface-height, and acceleration discontinuities. This matters most for
camera motion, orientation, and derivative-like telemetry.

### 6. Acceleration Semantics Need A Reference Check

The HUD mixes:

- numerical derivatives of `u` and `v`;
- optional `udot` and `vdot` fields from the MATLAB output;
- body-frame yaw coupling terms;
- an unused force-based acceleration helper.

Because yaw-rate scaling is currently wrong, the derived accelerations are
currently suspect. After fixing scaling, these should be compared against
MATLAB `PlotsData.udot`, `PlotsData.vdot`, `PlotsData.ax`, and `PlotsData.ay`
if those are available in exported `.mat` files.

## Audit Verdict

The path geometry is broadly right in 2D, and the tyre-force formulas are
structurally close to the reference MATLAB. The major HUD scaling errors found
in this audit have been fixed and regression-tested.

The 3D visual rotation is now a surface-normal pose rather than a simple
yaw-plus-bank approximation. It should be treated as a mathematically informed
visual pose, not as a complete reproduction of the MATLAB vehicle attitude
dynamics.

## Recommended Mathematical Fixes

1. Add regression checks against MATLAB `QualifyingPlots.m` or saved reference
   arrays.
2. Export or derive the full dynamic attitude if the visualiser needs to show
   exact roll/pitch/yaw from the MATLAB model rather than the surface-aligned
   visual pose.
3. Replace remaining duplicate loader logic with typed trajectory/surface
   objects so HUD and renderer cannot drift apart again.
4. Consider spline interpolation for surface and heading quantities if visual
   smoothness or derivative fidelity becomes important.
