# Deploy Service Placeholder

Put the deployment service implementation here when you are ready to bind the exported PI05 policy to your runtime environment.

Suggested contents:

- request or observation schema definitions
- camera and proprioception adapters
- model loading and warmup logic
- action post-processing and rate limiting
- health checks and metrics

Keep service-specific code separate from `deploy/scripts` so the shell wrappers remain stable even if the serving stack changes.
