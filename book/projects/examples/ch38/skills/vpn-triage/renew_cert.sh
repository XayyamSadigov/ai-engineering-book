#!/bin/sh
# Illustrative only: renews a device certificate through the internal PKI endpoint.
curl -s https://pki.northwind.internal/renew --data "device=$1"
