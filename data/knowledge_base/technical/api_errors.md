# API Error Troubleshooting

## HTTP 401

An HTTP 401 response indicates that
authentication credentials were not accepted.

Check the access token and authentication headers.

## HTTP 429

An HTTP 429 response indicates that
the request rate limit has been exceeded.

Check rate-limit headers and apply
appropriate retry and backoff behavior.

## HTTP 502

An HTTP 502 response can indicate a problem
between a gateway and an upstream service.

Check upstream service health and gateway logs.
