
# Dataset and labels

## Candidate dataset roles
| Dataset | Proposed role | Caveat |
|---|---|---|
| SAFER-Activities | Activity diversity and hard negatives | Generate estimated depth from RGB; verify labels/access |
| URFD | Small RGB-D diagnostic/reference set | Kinect depth differs from monocular estimated depth |
| Le2i | External RGB-video evaluation | Generate estimated depth; inspect labels/resolution |
| IKAR | Additional trip/slip/fainting-like events | Carefully map source labels |
| MCFD | Viewpoint robustness | Verify scenarios and split leakage |
| Custom stair recordings | Stair-fall evaluation | Essential for stair-fall claims |

Verify current access terms, licences, consent and redistribution restrictions. Keep source provenance; do not blindly merge datasets.

## Label ontology
Stage A: `normal`, `hard_negative_motion`, `fall`.
Stage B (falls only): `slip_fall`, `stair_fall`, `unknown_fall`.
Scene context: `flat_ground`, `stairs`, `unknown`.

Label event mechanism separately from scene context. Being near stairs does not prove a stair fall.

## Provenance and splits
Preserve dataset/video/subject/session/camera IDs, original and mapped labels, event times, scene context and annotation notes. Split by subject and session. Keep every camera view of the same physical event in one partition. Never randomly split adjacent frames. Reserve external scenes/datasets for domain-shift evaluation.
