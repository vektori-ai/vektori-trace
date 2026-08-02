"""Teacher and student model backends for distillation.

`teacher/` holds the scoring pools distillation reads logprobs from; `student/`
holds the trainable side. Import the concrete backend you need — this package
deliberately re-exports nothing, so importing a provider never drags in the
optional SDK of a provider you are not using.
"""
