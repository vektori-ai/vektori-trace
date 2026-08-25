# Qwen3-14B tau2 retail run analysis

Date audited: 2026-08-24  
Scope: Qwen3-14B only. The 4B and 8B reports were completed first.

## Evidence and run identity

The evidence was read directly from the live AWS box
`i-0a348ff3d7be9769a` in `ap-south-1` through SSM. Three artifacts exist:

| Artifact | Bytes | Modified (UTC) | SHA-256 | Interpretation |
|---|---:|---|---|---|
| `/data/tau2/data/simulations/qwen3_14b_smoke_20260823_164609.json` | 282,761 | 2026-08-23 16:54:07 | `0ee3d36d345b03c2a2a4be37b6a9b95b305855bd694d1427c74300c0471ff05b` | Invalid capability run: wrong tool parser, three stored tasks, zero executable tool calls |
| `/data/tau2/data/simulations/qwen3_14b_smoke_20260823_181615.json` | 145,392 | 2026-08-23 18:21:10 | `ab3f2960e6d9cf6dd1d20b43c8823c3eb96f5f68c1de5b16affa56ecd8584b1d` | Corrected Hermes parser; tasks 73, 75, 93 |
| `/data/tau2/data/simulations/qwen3_14b_smoke_20260823_183746.json` | 99,565 | 2026-08-23 18:44:33 | `2e08a2511bd510787ed07b30ab2d86174d5eb351b7821dd65fb10cad38fa791e` | Corrected Hermes parser; task 57 retry |

The matching logs are `/data/tau2/smoke14b.log`,
`/data/tau2/smoke14b_hermes.log`, and `/data/tau2/smoke57.log`. All artifacts
record retail, temperature 0, `max_steps=200`, agent
`hosted_vllm/Qwen3-14B`, and DeepSeek-V4-Flash-0731 as user simulator. Unlike
the 4B/8B artifacts, their `git_commit` field is `unknown`; the tau2 revision is
therefore not proven by the JSON itself.

## The first run is not a model-quality score

The initial artifact says 0/3, but vLLM used a tool parser that did not match
Qwen3-14B's chat-template output. The trajectories contain 22 literal
`<tool_call>...</tool_call>` blocks as ordinary assistant text and zero
structured tool calls. The environment never executed them. The model then
invented fake user IDs, order IDs, items, and payment methods because it never
received tool observations.

The serving script's recorded diagnosis says Qwen3-14B emits Hermes-style
`<tool_call>` tags and must use `--tool-call-parser hermes`; the earlier
`qwen3_coder` parser extracted nothing. The corrected runs do execute tools.
Thus 0/3 is proof of a deployment incompatibility and poor failure recovery,
not a fair tau2 capability measurement. It is still operationally important:
with the wrong parser this model fails silently rather than stopping.

## Headline result on corrected-parser trajectories

The corrected artifacts contain one trial per task and score 2/4:

| Task | Reward | Trajectory verdict |
|---|---:|---|
| 73, return all but espresso machine | 1 | Correct authentication, selection, confirmation, and DB mutation; unsupported return logistics afterward |
| 75, exchange earbuds | 0 | No authentication; acts before full attributes/confirmation and picks the wrong water-resistant variant |
| 93, exchange laptop | 0 | Authenticates and obtains all three order IDs, then refuses to inspect them and transfers |
| 57, conditional cancellation | 1 | Preserves DB after gift-card feasibility fails; correctly avoids cancellation and transfers for unavailable shipping estimates |

This 2/4 is a diagnostic outcome, not a stable pass-rate estimate: every task
has only one corrected-parser trial.

## Trajectory findings

### Task 73: the cleanest end-to-end 14B path

The model authenticates Fatima Wilson by email, retrieves her user details and
order, lists the four non-espresso items, identifies the original credit card,
and explicitly asks for “yes.” After the user confirms, it calls
`return_delivered_order_items` with the expected IDs and reaches the correct DB
state.

The final reply nevertheless invents a prepaid label and a 5–7 business-day
refund timeline, neither of which is supported by the supplied return policy.
So the state-changing core is correct, while operational grounding remains
weak.

### Task 75: premature commitment to a partial match

The model calls `get_order_details` without authenticating Liam Moore. It then
retrieves the product variants but immediately selects item `9580569596`
(black, 4-hour, **IPX7**) and calls the exchange tool. The required item is
`4063058357` (black, 4-hour, **not resistant**).

The user had initially disclosed only that the desired replacement was black;
the remaining attributes were meant to emerge progressively. Instead of asking
for them, the model committed to the first black/4-hour partial match. It also
did not list the action details or obtain explicit confirmation before the DB
mutation. When the user later asks whether the new pair has no water
resistance, the model correctly reports that it selected IPX7—but the wrong
exchange is already irreversible.

This is a precise failure of clarification and action gating, not inability to
read the variant table: the correct variant was present in the same tool output.

### Task 93: order IDs found, then discarded

The model authenticates Lei Wilson and calls `get_user_details`, which returns
all three order IDs: `#W3826449`, `#W2905754`, and `#W4073673`. It then asks the
user to supply the order ID, says it cannot look up orders by product details,
and transfers to a human.

This is a planning/tool-use failure. The candidate IDs were already available,
and three `get_order_details` calls would have identified `#W4073673`. Unlike
8B's common failure, 14B does not even perform a partial search in this trial.
It recognizes uncertainty but resolves it through unnecessary escalation.

### Task 57: the strongest evidence in favor of 14B

The model authenticates Ivan Hernandez, recovers the `#`-prefixed order ID from
the user's account, and identifies that the pending order cannot cancel only
the air purifier. It understands the user's condition that the whole order
must remain intact unless the refund can reach the gift card.

After explicit user approval it tries
`modify_pending_order_payment(... gift_card_9368765)`. The tool rejects the
change because the `$85` gift-card balance cannot cover the `$2,020.61` order.
The model then asks whether the user instead wants cancellation to the original
credit card. The user says no; the model performs no cancellation, leaving the
DB unchanged as required. It later transfers for shipping estimates it cannot
obtain with available tools, using the exact mandated handoff message.

This trajectory demonstrates the capability both 4B and 8B failed: checking a
nested financial precondition before an irreversible cancellation. It is only
one trial, so repeatability is unknown.

One caveat is visible in the trajectory: the DeepSeek user simulator's opening
message says “I'd be happy to help you with your order,” briefly reversing the
customer/agent roles. The conversation recovers and still exposes the full task,
but the simulator is not perfectly clean.

## Failure taxonomy

1. **Parser sensitivity:** the wrong vLLM parser turns valid-looking model text
   into non-actions with no hard failure.
2. **Clarification failure:** the model selects a partial attribute match before
   the user reveals all required attributes.
3. **Search refusal:** it possesses a three-order candidate set but declines to
   inspect it.
4. **Authentication/action-gate violations:** task 75 skips authentication and
   confirmation before mutation.
5. **Unsupported operational claims:** prepaid labels and shipping/refund
   timings are invented even on the successful path.
6. **Extremely small corrected sample:** one trial per task makes variance and
   repeatability unmeasured.

## What is and is not proven

Proven: Qwen3-14B requires the Hermes tool parser in this deployment; the
corrected model can execute structured retail tools and, in one task-57
trajectory, preserves a nested user constraint that both smaller models violate.
It also fails two ordinary workflows through premature commitment and refusal
to search.

Not proven: that 14B's 2/4 is its stable tau2 rate, that its task-57 advantage
repeats, or that the corrected runs used the exact same tau2 git revision as the
4B/8B runs (the field is `unknown`). A final model recommendation must weight
the qualitatively important task-57 success but cannot treat one sample as a
size-scaling law.
