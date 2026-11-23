
# API and integration

`GET /health` returns service status and model-loaded state.

`POST /events` accepts event ID, camera ID, track ID, timezone-aware start/end timestamps, Stage A class/confidence, optional Stage B class/confidence, scene context and model version.

Example payload values in API tests are schema examples only, not real predictions. Keep inference separate from transport. Add authentication, authorization, audit logs and retention controls before deployment. Do not transmit continuous RGB by default.
