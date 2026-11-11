# Nested ownership stacking: source review and application

Research date: 2026-10-27. This note records the method rationale, not a score
result. The prospective protocol is [NESTED_OWNER_STACK_PLAN.json](NESTED_OWNER_STACK_PLAN.json).

Wolpert's original stacked-generalization paper constructs a second training
problem from predictions made on examples omitted from each base learner's
training partition. At inference, base learners trained on the available
training set provide the second learner's inputs. The paper also describes
averaging and weighted combinations as restricted alternatives and warns that
an unsuitable second learner can perform worse. Its discussion favors a smooth,
stable second learner when learning from a small prediction space. These ideas
justify testing learned expert combination; they do not guarantee improvement
for possession intervals. [Wolpert, 1992, sections 2.4 and 4](https://cafri-labs.github.io/lab-manual/papers/wolpert1992.pdf).

The official scikit-learn documentation likewise trains its final stacking
classifier from cross-validated base predictions. Its default final classifier
is logistic regression. It warns that using already fitted experts' predictions
on the same data used to train those experts creates a serious overfitting
risk. This supports an additional inner split inside each outer evaluation
fold, rather than treating the repository's global OOF predictions as safe
meta-training inputs for every outer fold.
[StackingClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html).

Logistic regression applies regularization by default; its L-BFGS solver supports
L2 regularization. `C` controls its inverse strength, and `fit_intercept=False`
omits the constant term. The experiment fixes these choices prospectively and
uses the existing installed library without a parameter search.
[LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

The repository application uses two unchanged XGBoost expert interfaces:
ordered102 and ordered102 plus the R37 encoder's fixed64 features. Within every
outer training pool, three fixed100-second chunks produce inner OOF predictions.
Each inner fit excludes the held chunk plus250 frames and all outer validation
labels. A read-only feasibility check finds36 usable inner folds, requiring72
expert fits; all retain H/A/D training classes, with a minimum class count of46.
Empty inner validation sets are recorded rather than fitted.

The second learner receives only two clipped H/A log odds. It trains on live
inner-OOF rows, mirrors both logits and H/A targets, uses uniform weights, and
has two coefficients with no intercept or scaler. Each outer evaluation uses
the corresponding immutable outer-fit expert predictions. The resulting owner
probability is combined with the original slot0 dead-ball mass, passed through
the unchanged selected8 decoder, and accepted only through the existing live
overlay. Original D/X decisions remain protected.

This differs from R41's fixed probability average, R42's new direct owner head,
and the earlier nested dead/live stack. It introduces learned ownership fusion
with both exclusion layers. It still relies on17 repeatedly reused public
possessions, so any resulting score remains adaptive within-match development
evidence. Expert correlation, inner-to-outer training-size differences, and
optimizing framewise logistic loss instead of interval matching can all prevent
a gain. The fixed protocol's score and secondary safeguards decide advancement.
