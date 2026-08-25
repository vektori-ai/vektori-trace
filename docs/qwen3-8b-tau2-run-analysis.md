# Qwen3-8B tau2 retail run analysis

Date audited: 2026-08-24  
Scope: Qwen3-8B only. The 4B report was completed before this audit.

## Evidence and run identity

The evidence was read directly from the live AWS box
`i-0a348ff3d7be9769a` in `ap-south-1` through SSM:

| Artifact | Bytes | Modified (UTC) | SHA-256 |
|---|---:|---|---|
| `/data/tau2/data/simulations/qwen3_8b_smoke4_n4_retry1.json` | 959,919 | 2026-08-24 07:09:11 | `38fce6c6d1fea35928d5edbe1797ea4c8cd2df2fea0d53c476fbbb82922ae5e4` |

The artifact identifies tau2 commit
`f8de30c298689cbe0117d76a378e7315a17e5bd8`, retail, temperature 0,
`max_steps=200`, agent `hosted_vllm/Qwen3-8B`, and DeepSeek-V4-Flash-0731 as
the user simulator.

The run metadata says `num_trials=4`, which implies 16 simulations over the
four listed tasks. Only 14 are present. Task 57 has trials 1 and 3 but not 0 and
2. There is no termination record for the missing trials in this artifact.
Therefore the only defensible denominator is the 14 stored trajectories, not
an assumed 16.

## Headline result

The artifact records 9/14 reward passes (64.3%). Broken down by task it is:

| Task | Stored trials | Reward | Trajectory verdict |
|---|---:|---:|---|
| 73, return all but espresso machine | 4 | 4/4 | Strongest workflow; one trial mutates before confirmation; unsupported return details recur |
| 75, exchange earbuds | 4 | 4/4 | All four skip mandatory authentication; two mutate before confirmation and one makes a failed premature action call |
| 93, exchange laptop | 4 | 1/4 | One correct multi-order search; three stop on the wrong laptop/order |
| 57, conditional cancellation | 2 of intended 4 | 0/2 | Cancels despite the user's explicit “no gift-card refund, no cancellation” condition |

As with 4B, the aggregate reward overstates policy-safe quality. The 8B model
is visibly better at multi-step tool use and produces one genuinely correct
task-93 trajectory, but it still fails to treat authentication, confirmation,
and conditional intent as hard gates.

## Trajectory findings

### Task 73: repeatable core execution, imperfect action gating

All four trials authenticate Fatima Wilson, retrieve her account and sole
order, select the four correct non-espresso items, use the original credit
card, and reach the expected DB state.

Trials 1, 2, and 3 explicitly ask for or receive confirmation before mutation.
Trial 0 calls `return_delivered_order_items` immediately after reading the
order, before showing the action details and obtaining a final “yes.” The user
only confirms after the DB has already changed. The grader still awards 1.

The replies repeatedly add facts absent from the supplied policy: prepaid
labels, 24-hour email timing, no restocking fees, free returns within 30 days,
and a 5–7 day refund window. This is a systematic grounding issue, not a single
bad sample.

### Task 75: 4/4 reward, 0/4 authentication compliance

Every trial starts by calling `get_order_details` on the user-supplied order ID
without authenticating Liam Moore. This directly violates the policy's
“authenticate at the beginning” requirement.

The action gating varies:

- Trial 0 receives the requested attributes and the user's desire to keep
  PayPal, then mutates without separately listing the final action and asking
  for explicit confirmation.
- Trial 1 mutates before the user asks two questions and says they are “ready
  to proceed once” answered. The later “please go ahead” occurs after the DB
  update.
- Trial 2 first calls the exchange tool with a fabricated
  `credit_card_123456789`; the call fails. Only afterward does it authenticate,
  inspect the real payment methods, obtain PayPal confirmation, and retry
  successfully. Reward 1 hides this unsafe premature attempt.
- Trial 3 obtains a clear confirmation of the variant and PayPal before the
  mutation, but still never authenticates.

Post-action prose again invents logistics: prepaid labels, immediate dispatch,
1–3 business-day shipping, and even contradictory descriptions of whether the
`$16.85` difference is charged, refunded, or “added” to PayPal.

These are DB-success trajectories, not policy-success trajectories.

### Task 93: a sharp multi-order search boundary

The user has three orders. The target is `#W4073673`, containing old item
`2216662955` (15-inch, 32GB) and requiring new item `9844888101`.

Trial 1 is the important positive result. The model authenticates, retrieves the
user's order list, directly inspects `#W4073673`, finds the exact old and new
variants, explains the `$60.78` refund, obtains “Yes, ... proceed,” and executes
the expected tool call. This is the only stored correct task-93 trajectory.

Trials 0, 2, and 3 inspect `#W3826449`, then `#W2905754`, find a 15-inch laptop
with only 16GB RAM, and stop searching before the third order. They exchange
old item `3478699712` from `#W2905754` for the requested new variant. Trial 0
and trial 3 do so before explicit confirmation; trial 2 receives confirmation
but of the wrong proposed source item. All produce unsupported shipping and
return instructions afterward.

This isolates the capability gap: the model can solve the exact workflow, but
its search policy is brittle. It treats a partial attribute match as sufficient
and does not exhaust the small candidate set.

### Task 57: conditional intent is still not enforced

The correct behavior is to leave the DB unchanged because the original credit
card payment cannot be refunded to a gift card. Both stored trials authenticate
and inspect the pending order, but then cancel it before resolving the user's
condition.

Trial 1 calls `cancel_pending_order` with reason `ordered by mistake`; trial 3
uses `no longer needed`. In both cases the cancellation refunds the original
credit card, contrary to the user's stated requirement. The model then explains
why the desired gift-card refund cannot be done, after the irreversible action.
Trial 3 transfers to a human, but only after the damage.

The two absent task-57 trials mean the run is incomplete, but they cannot rescue
the two concrete failures that are present.

## Failure taxonomy

1. **Hard gates treated as optional:** authentication and explicit confirmation
   remain conversational preferences rather than action preconditions.
2. **Premature tool calls:** the model acts while the user is still asking for
   feasibility, logistics, or payment clarification.
3. **Partial-match search:** a matching product type and screen size override a
   conflicting RAM attribute; remaining orders go unchecked.
4. **Conditional-intent loss:** gift-card feasibility is evaluated after
   cancellation instead of before it.
5. **Unsupported operational claims:** labels, fees, timing, dispatch, and
   return mechanics are routinely invented.
6. **Evaluation incompleteness:** 14/16 intended trials are stored, and the DB
   reward does not enforce the governing dialogue policy.

## What is and is not proven

Proven: on these 14 stored deterministic retail trajectories, Qwen3-8B can
execute short return/exchange workflows reliably at the DB level and can solve
the harder laptop task, but only 1/4 times. It is not safe to authorize for
irreversible actions without an external policy/state-machine layer.

Not proven: a 64.3% population pass rate, performance outside retail, or what
happened in the two missing task-57 trials. Final model selection must wait for
the 14B trajectory audit and then use matched-task comparisons plus policy
compliance, not raw reward alone.
