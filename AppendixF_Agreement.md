## Appendix F — Pipeline Validation: Per-Tag Agreement with Human Labels

Agreement is computed over 541 hand-labeled responses (191 at 4×4, 350 at 5×5), weighted toward higher dimensions where automated agreement is lowest. Dashes indicate the tag was not present at that dimension.

| Tag | 3×3 Agree% | 4×4 Agree% | 5×5 Agree% | Note |
|---|---|---|---|---|
| HALLUCINATION | 100% | 92.8% | 92.7% | |
| SIGN_ERROR | 100% | 91.7% | 82.5% | |
| ARITHMETIC | 82.9% | 68.3% | 70.2% | |
| INPUT_TRANSCRIPTION | 87.5% | 75.0% | 68.1% | |
| METHOD_FAIL | 100% | 88.9% | 68.3% | |
| GENERATION_TRUNCATION | 100% | 83.3% | 63.6% | <70% at 5×5 |
| MEMORY_LOSS | — | 100% | 57.1% | <70% at 5×5 |
| FALSE_VERIFICATION | — | 0% | 83.3% | <70% at 4×4 |
| VARIABLE_ENTANGLEMENT | — | 20% | 0% | <70%; excluded |
| ALGEBRAIC_PRECEDENCE | — | 0% | — | <70%; excluded |
| OTHER_UNMAPPED | 0% | 50% | 25% | <70%; excluded |
| GENERATION_LOOP | — | 100% | 0% | <70% at 5×5 |

> **Note:** Tags marked <70% do not contribute to the distributional analysis in Section 5. Overall pipeline agreement with human labels: 95.5% at 4×4, 87.7% at 5×5, 90.4% overall.
