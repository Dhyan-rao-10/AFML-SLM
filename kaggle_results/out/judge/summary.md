| x | y | Prompts | x score (95% CI) | x wins | y wins | Ties | Order-consistent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_masked | base | 200 | 0.784 ± 0.034 | 127 | 2 | 71 | 100% |
| full_masked | lora_masked | 200 | 0.549 ± 0.045 | 63 | 40 | 97 | 98% |
| full_masked | full_unmasked | 200 | 0.536 ± 0.043 | 55 | 37 | 108 | 100% |
| lora_masked | lora_unmasked | 200 | 0.529 ± 0.043 | 56 | 46 | 98 | 99% |

x score: 1 = x always preferred, 0.5 = no preference, 0 = y always preferred (ties count half).
Order-consistent: share of prompts where the two orderings did not pick different models.

x score by Dolly category:

| Category | full_masked vs base | full_masked vs lora_masked | full_masked vs full_unmasked | lora_masked vs lora_unmasked |
| --- | --- | --- | --- | --- |
| brainstorming | 0.73 | 0.53 | 0.46 | 0.54 |
| classification | 0.78 | 0.49 | 0.53 | 0.62 |
| closed_qa | 0.76 | 0.49 | 0.49 | 0.42 |
| creative_writing | 0.90 | 0.69 | 0.69 | 0.60 |
| general_qa | 0.82 | 0.58 | 0.49 | 0.55 |
| information_extraction | 0.82 | 0.65 | 0.62 | 0.43 |
| open_qa | 0.70 | 0.48 | 0.52 | 0.55 |
| summarization | 1.00 | 0.70 | 0.73 | 0.36 |
