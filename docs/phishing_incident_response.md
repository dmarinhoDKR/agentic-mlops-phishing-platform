# Phishing Incident Response Runbook

## Scope

This runbook defines the recommended response when a suspicious message is
classified as phishing. The copilot provides decision support and does not
perform destructive or privileged actions automatically.

## Immediate Containment

- Do not click links, open attachments, reply, or forward the message.
- Report the message through the approved security channel.
- If an attachment was opened or code was executed, disconnect the affected
  device from the network and contact the security team.
- Preserve the original message for investigation.

## Credential Protection

If credentials were entered into a suspicious page:

- Reset the affected password from a trusted device.
- Revoke active sessions and authentication tokens.
- Confirm that multifactor authentication is enabled.
- Review account recovery settings and email forwarding rules.
- Check recent sign-ins for unknown devices or locations.

## Evidence Preservation

Collect evidence without interacting with suspicious content:

- Sender address and display name.
- Message headers and timestamps.
- URLs shown in the message.
- Attachment names and hashes, when safely available.
- Screenshots and the list of recipients.
- Related authentication, email, proxy, and endpoint logs.

## Investigation

- Determine whether other users received the same message.
- Search logs for access to identified domains, URLs, or IP addresses.
- Check whether credentials, tokens, or sensitive information were exposed.
- Inspect affected endpoints for downloaded files or suspicious processes.
- Record indicators of compromise for containment and monitoring.

## Recovery And Monitoring

- Remove confirmed phishing messages from affected mailboxes.
- Block malicious senders, domains, URLs, hashes, and IP addresses.
- Restore affected accounts and devices through approved procedures.
- Monitor authentication and endpoint activity for recurrence.
- Document the incident timeline, decisions, evidence, and remediation.

## Escalation Criteria

Escalate immediately when:

- Credentials were submitted.
- An attachment or executable was opened.
- A privileged or administrative account is involved.
- Sensitive data may have been disclosed.
- Multiple users or systems are affected.
- Malicious activity continues after initial containment.

## Automation Boundaries

The copilot may classify messages, inspect model quality, retrieve trusted
guidance, and produce cited recommendations. Account changes, message removal,
network isolation, credential revocation, and blocking actions require explicit
human approval and authorized operational tooling.
