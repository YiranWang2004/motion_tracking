# OmniContact asset notice

The following files were derived from the local OmniContact_sim2sim source at
commit `3a61521`:

- `policy.onnx`: byte-identical OmniContact carry-box tracker;
- `OmniContact.yaml`: model/control values preserved, with whitespace cleanup;
- `DefaultPose.yaml`: reference pose configuration;
- `g1_29dof_fk.xml`: `g1_29dof.xml` with visual mesh assets and mesh geoms
  removed; the kinematic joint/body tree is retained for policy observation FK;
- `../assets/omnicontact_carry_box.xml`, `../assets/g1_29dof.xml`, and
  `../assets/g1_ghost.xml`: the carry-box MuJoCo validation scene and its two
  included robot models. The scene reuses the byte-identical G1 mesh files
  already stored under `../assets/meshes/`.

The carry-box CFgen code under `src/omnicontact/reference` is the carry-only
adaptation previously developed in the sibling Deploy repository.

The upstream README identifies OmniContact as CC BY-NC-SA 4.0. Preserve
upstream attribution and license obligations when redistributing this code,
configuration, or model. This notice does not relicense upstream material under
the motion_tracking MIT license.
