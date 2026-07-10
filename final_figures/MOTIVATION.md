# Motivation — what the method is for (poster text)

## One-line verdict
An **affordance-region *proposer* for known object types and functional verbs** —
a *hint / planning* tool, **not** a precise executor.

## What problem it solves (the gap)
An embodied agent (robot, AR assistant) looking at a real object needs to know
**where** on it an action applies — where to pour into, place onto, rest on,
grasp. Prior 3D-affordance methods need **clean CAD/scanned point clouds** and a
**fixed label set**. We instead go **from a single real photo** → 3D mesh →
a **per-vertex, open-vocabulary, verb-conditioned** affordance map. Cheap input,
arbitrary verbs (CLIP-text), no per-object CAD model.

## What you'd actually use it for
1. **Manipulation / task-planning priors** — a coarse "where to act" hint for a
   downstream policy or a human-robot handoff ("pour *here*", "place *into
   here*"). Geometric/functional verbs even transfer to unseen categories, so it
   works on novel objects of familiar *kinds*.
2. **Scalable affordance auto-annotation** — the model **beats the GEAL teacher
   it distilled from (+0.058)**, so it can bootstrap/refine 3D affordance
   datasets semi-automatically.
3. **AR / embodied assistance** — highlight functional regions on everyday
   objects from a phone image.

## Honest scope (small "not for" line on the poster)
- ✅ Region *proposals* for functional verbs (contain, pour, place-on, display) on
  **known categories** — and cross-category for the geometric ones.
- ❌ **Precise robotic grasping** (grasp is the weakest verb, fails cross-category)
  and **open-world zero-shot** novel verbs — though **few-shot** bridges new verbs
  with a handful of labels.

## Poster-ready Motivation block (drop-in)
> **Why.** Robots and AR agents need to know *where* on an object an action can be
> applied — in 3D, from cheap input, for arbitrary verbs. Prior 3D-affordance
> methods assume clean scans and a fixed vocabulary. **We predict per-vertex,
> open-vocabulary, verb-conditioned affordances directly from a single real
> image.** Use it as a **manipulation/planning prior** and a **scalable
> auto-annotator** — a region proposer for functional verbs, strong within and
> (for geometric verbs) across categories.
