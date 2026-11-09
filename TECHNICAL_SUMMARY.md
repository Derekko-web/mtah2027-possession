# Team possession from tactical and broadcast tracking

## Inputs and model

We produce one label per tactical frame for home possession (`H`), away possession (`A`), dead ball (`D`) or an ungraded frame (`X`). The system uses the supplied player and ball tracking, team metadata, homography validity and tactical Smart Tagging. It aligns the two unsynchronised camera feeds by matching team formations and checking that neighbouring matches agree on camera orientation and time progression. A finer search fills some gaps in the original mapping. The alignment uses tracking geometry and automatic metadata.

The ownership model takes 102 ball, player, motion and contact features and 64 features from an anonymous player-set encoder trained on public DFL data. The encoder reads player coordinates and team groups in 31-sample windows at 5 Hz. It averages four pitch reflections and interpolates the hidden features between supported observations within a half. TabPFN v2 estimates home versus away ownership from 2,352 paired public training rows. The pair augmentation exchanges teams at training and inference to reduce dependence on team order. Separate XGBoost heads supply fast control, joint live/dead predictions and a second view of ownership from broadcast and pair-motion features. The dead-ball heads combine the public training labels with weak evidence from ball physics and automatically tagged restarts.

## Possession decisions

The temporal decoder combines frame evidence with a cost for switching states. During loose balls or challenges, existing control can persist when the other side lacks sustained evidence. We accept a brief change of ownership when the fast control model and player-to-ball contacts corroborate it. The final short-control stage proposes changes from unsmoothed ownership estimates and requires contact evidence in both aligned cameras. Stoppage rules use stationary or off-pitch ball evidence, ball visibility and restart context. A short dead interval can be bridged when the same side controls a moving ball on both sides and opposing control is weak. Both halves are decoded separately using offline observations.

The final stage was added after a 51-frame smoother erased a 19-frame home possession. It recovered the full interval and split the two surrounding away possessions. The same rule applies to both teams, using model probabilities and observations as inputs.

## Evaluation and limits

| Official scoring on the public window | Selected CSV |
| --- | ---: |
| SpellF1, mean across IoUs 0.3 to 0.7 | 1.000000 |
| MacroF1 | 0.993200 |
| Frame accuracy | 0.993733 |
| Matched possessions at every scored IoU | 17 of 17 |

The final stage preserved all nine out-of-fold predictions across three seeds and three temporal splits. Those comparisons use the same 7,500 public frames that guided repeated development. They measure consistency on the public match window. The private full-match score and performance on a new match remain unknown. Missing or misaligned camera observations and brief handovers near stoppages remain difficult. A match-disjoint evaluation would need both tracking views and independently labelled live ownership and stoppages. The external DFL data has ownership and ball-status labels with a single integrated tracking stream. [DFL data paper](https://www.nature.com/articles/s41597-025-04505-y)

## Reproduction

The [build instructions](SUBMISSION_BUILD.md) specify the pinned environments and one build command. It regenerates the CSV from the provided tracking bundle, the shipped public labels, the public-source player encoder and the author TabPFN checkpoint. The selected CSV SHA256 is `801cb64b59260daf4586865e9ef13d124fc61d7584e2798283e465a1758e9ed7`.

Built with Prior Labs TabPFN. [TabPFN v2 model card and license](https://huggingface.co/Prior-Labs/TabPFN-v2-clf)
