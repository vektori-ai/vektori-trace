# Qwen3 4B vs 8B vs 14B on the audited tau2 retail runs

Date audited: 2026-08-24

This comparison was written only after the separate 4B, 8B, and 14B reports.
It uses the trajectories on the live AWS driver box, not remembered benchmark
claims. DeepSeek-V4-Flash-0731 is the **user simulator** in every audited run;
its separate tau2 score is not a Qwen reward and is not evidence for choosing
among these three agents.

## Bottom line

**Best-supported experimental pick today: Qwen3-8B.** It has the largest usable
sample (14 stored trajectories), perfect DB outcomes on the two simple tasks,
and the only repeated evidence that a smaller Qwen can sometimes solve the hard
multi-order laptop search. It is the most defensible cost/capability baseline
for further training or evaluation.

**Most promising but not yet proven: Qwen3-14B.** It is the only model that
preserved the nested “gift-card refund or do not cancel” constraint, which is
the most safety-relevant task in this set. But that claim rests on one corrected
trial. Its corrected run is only 2/4, its tau2 git revision is recorded as
`unknown`, and an earlier parser-mismatched run failed operationally.

**Not acceptable for autonomous deployment: any of the three.** Every size has
at least one trajectory that mutates without required authentication or final
confirmation, and all invent unsupported return/shipping details. An external
policy/state-machine gate is required regardless of model size.

## What the numbers actually say

| Model | Usable stored trajectories | Reward passes | Raw rate | Evidence defect |
|---|---:|---:|---:|---|
| Qwen3-4B | 8 | 4 | 50.0% | Only two trials per task; task 57 was a separate diagnostic artifact |
| Qwen3-8B | 14 | 9 | 64.3% | Metadata requests 16; task-57 trials 0 and 2 are missing |
| Qwen3-14B | 4 corrected-parser | 2 | 50.0% | One trial per task; git revision `unknown` |

The raw ranking is therefore 8B > 4B = 14B, but the denominators are unequal
and tiny. It would be invalid to infer a size scaling curve from these rates.

The first 14B artifact's 0/3 is excluded from capability scoring because the
wrong vLLM tool parser left every `<tool_call>` block in assistant text. It is
retained as deployment evidence: 14B requires the Hermes parser and fails
silently when misconfigured.

## Matched task behavior

| Capability | 4B | 8B | 14B | Best evidence |
|---|---|---|---|---|
| Return a known set after authentication (73) | 2/2 DB pass | 4/4 DB pass | 1/1 DB pass | All can do it; 8B has most replication |
| Exchange a fully specified item (75) | 2/2 DB pass | 4/4 DB pass | 0/1 | Raw: 8B; policy-safe: none, because all sizes skip authentication in this task |
| Search several orders for an attribute match (93) | 0/2 | 1/4 | 0/1 | 8B, narrowly; its failures and 4B stop at partial matches, while 14B refuses the search |
| Preserve nested condition before cancellation (57) | 0/2 | 0/2 stored | 1/1 | 14B, but replication is required |

This reveals two different capability axes rather than one monotonic “bigger is
better” axis:

- 8B has the best demonstrated search/execution coverage.
- 14B has the only demonstrated success on conditional financial intent.
- 4B has no unique capability win and repeats the severe task-57 failure.

## Where each model goes wrong

### Qwen3-4B

- Drops nested conditions at the exact point of irreversible tool use.
- Claims completed cancellations can be undone without any supporting tool.
- Gives up on order enumeration or stops at a superficial product match.
- Produces DB-success false positives that skip authentication/confirmation.

The 4B model knows much of the policy in prose, but does not reliably convert it
into execution preconditions. It is the weakest pick for an action-taking agent.

### Qwen3-8B

- Executes short workflows more consistently than 4B.
- Can solve the full laptop task, but only 1/4 times.
- Usually stops after finding the first 15-inch laptop even when RAM conflicts
  and a third order remains unchecked.
- Still cancels against the gift-card condition in both stored task-57 trials.
- Makes premature action attempts, including one fabricated payment-method ID.

The improvement is chiefly tool-workflow breadth and occasional search success,
not reliable policy safety.

### Qwen3-14B

- With Hermes parsing, performs the clean simple return and the difficult
  conditional no-cancel task.
- Commits to a partial product match before collecting all requested attributes.
- Retrieves the user's order list but refuses to inspect its three entries.
- Is operationally sensitive to tool-parser configuration.

The 14B trajectory suggests better conditional reasoning, but not consistently
better agency. Its ordinary search failure is worse than 8B's best behavior.

## Safety-adjusted interpretation

The tau2 DB reward does not enforce all supplied policy requirements. Examples
awarded reward 1 despite missing authentication, missing explicit confirmation,
or premature failed action calls. Consequently “pass” means final DB match, not
fully compliant behavior.

A conservative manual core-policy screen changes the picture:

- 4B has two reasonably compliant task-73 paths out of eight stored paths.
- 8B has three task-73 paths plus one task-93 path out of fourteen; the other
  DB passes contain material gate violations.
- 14B has two reasonably compliant paths out of four: task 73 and task 57.

These are audit counts, not a replacement benchmark metric. They do show why
the 14B result deserves a rerun even though its raw rate is unimpressive.

## Deployment and cost facts

The AWS instance is the CPU driver (`t3.xlarge`); the run logs show the 14B
endpoint was served on a Modal A100-80GB and torn down after each run. Therefore
the JSON's `agent_cost: 0` does **not** mean free inference—it only means tau2
did not meter the self-hosted endpoint.

Recorded conversation duration is not a clean model-speed benchmark because it
includes the DeepSeek user simulator, tool calls, different turn counts, and
concurrent scheduling. The observed means are roughly 137 seconds (4B), 172
seconds (8B), and 129 seconds (14B corrected), which clearly do not scale with
parameter count and should not drive the choice.

The defensible cost fact is structural: 4B < 8B < 14B in model compute/memory,
while this run provides no normalized tokens/second or dollars/trajectory
measurement. Do not invent a cost ratio from parameter count.

## Recommendation

### If choosing now for the next experiment

Choose **Qwen3-8B** as the baseline. It has the strongest replicated evidence,
the highest raw stored reward, and is the smallest model with a demonstrated
success on the multi-order task. Do not present it as production-safe; put all
DB mutations behind deterministic authentication, confirmation, allowed-payment,
and user-condition checks.

### Before replacing it with 14B

Run a matched corrected-parser evaluation with the same pinned tau2 revision,
the same four task IDs, and the same four seeds. The minimum decision gate should
be:

1. all 16 trajectories are present;
2. Hermes parsing produces structured calls, with zero raw `<tool_call>` blocks;
3. task 57 succeeds at least 3/4 times without cancellation;
4. task 93 beats 8B's 1/4 rather than merely tying it;
5. no mutation occurs before authentication and explicit confirmation; and
6. unsupported logistics are counted separately from DB reward.

If 14B clears those gates, choose **14B**: reliable conditional-intent handling
is worth more than the compute savings for a financial/transactional agent. If
it does not, stay with **8B plus deterministic guards**; the current evidence
does not justify paying for 14B merely because it is larger.

### Production verdict

Choose **none without guards**. Model scaling did not eliminate the common
failure mode: natural-language policy is not reliably enforced at tool-call
time. The correct architecture is model for dialogue/planning, deterministic
code for authorization and irreversible-action preconditions.
