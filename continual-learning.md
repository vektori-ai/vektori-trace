
2. Experience replay - keep a set of earlier experineces + mix it into later updates. while if u do Dark Experience replay(der) thats just preserving the odl models behavrio on those experiences.

with ER basically u do researvoir sampling - every observed example has equal prob of reamining in the buffer(fixed)
starttified replay - reserve based on capability/failure mode.

for a coding agent - requires task state, prefix, action, verifier result and so on

So DER stores the trajectories(logits of the model upahead) and penalizes between the driff of current vs stored output. 


basically simple replay is baseline.

Reasoning-Augmented Continual Learning - add in the reasoning and rationale + "meta-decision"

it exploits the observation that examples containing reasoning paths caused less forgetting


O-LORA
Allocate successive LoRA updates to mutually orthogonal low-rank subspaces so new tasks interfere less with previous adapters. At inference, learned updates can be accumulated rather than requiring replay of old examples.

OPLora- Project LoRA updates away from the frozen model’s dominant singular directions to preserve high-energy pretrained structure. Ordinary LoRA can overwrite important pretrained directions even when its rank is small. Apply projected LoRA to both OPD and RL routes, then evaluate base coding skills, instruction following, tool syntax, and all learned deficits.
