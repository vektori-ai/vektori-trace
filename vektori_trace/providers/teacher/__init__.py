"""Teacher scoring pools.

`base` carries the shared errors and helpers plus the vLLM pool; `bedrock`,
`fireworks`, and `cross` are the hosted and cross-tokenizer backends built on
it. Kept as submodules rather than re-exports so that `import ...bedrock` is
the only thing that requires boto3, and `...fireworks` the only thing that
requires the Fireworks SDK.
"""
