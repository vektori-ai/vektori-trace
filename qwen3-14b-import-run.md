# Qwen3-14B Bedrock import run

Date: 2026-08-19
Region: us-east-1
S3 source: s3://mh-models-478499050241/qwen3-14b/

## Cost guardrail

- Import and upload first; no inference calls before validation.
- Run the minimum generation/logprob tests back-to-back in one billing window.
- Stop requests immediately, allow scale-to-zero, then delete the imported model.
- Retain the reusable S3 snapshot.
- Terminate temporary EC2 instances and delete their auto-delete volumes.

## Resources and commands

- Bedrock import role: `arn:aws:iam::478499050241:role/BedrockQwen3TeacherImportRole`
- First staging instance: `i-0e0ec4dcb098ce225` (t3.small; became CPU-bound during Xet reconstruction; stopping/cleanup pending)
- First download command: `0f20d57b-71fa-4171-a0f5-4e8fc629213b` (cancelled safely)
- Replacement staging instance: `i-0b4e0a74cf4cb6d47` (t3.xlarge, direct HTTP download)
- Active download command: `1332f116-c460-4e68-8142-0070aa6ad289`
- Temporary EC2 role policy: `qwen3-14b-staging-temp` (remove after verified upload)

## Observations

- Target S3 prefix was empty before the run.
- Existing Bedrock import role trust and bucket read permissions were verified.
- Direct HTTP download reached approximately 30 GB with 19 GB disk space remaining.
- Bedrock inference calls made so far: 0.

## Test payloads

### Generation plus logprobs

```json
{
  "messages": [{"role": "user", "content": "Reply with exactly QWEN3_OK and nothing else."}],
  "max_tokens": 16,
  "temperature": 0.0,
  "logprobs": true,
  "top_logprobs": 3,
  "prompt_logprobs": 1
}
```

### OPD-shaped assistant-prefix scoring

```json
{
  "messages": [
    {"role": "user", "content": "What is 17 multiplied by 19? Give only the number."},
    {"role": "assistant", "content": "The answer is"}
  ],
  "max_tokens": 8,
  "temperature": 0.0,
  "logprobs": true,
  "top_logprobs": 3,
  "prompt_logprobs": 3
}
```

Final validation, responses, CMU count, timestamps, and cleanup evidence will be appended after completion.
