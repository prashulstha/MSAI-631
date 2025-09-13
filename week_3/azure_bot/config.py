#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

""" Bot Configuration """


class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978
    APP_ID = os.environ.get("MicrosoftAppId", "")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "")
    APP_TYPE = os.environ.get("MicrosoftAppType", "MultiTenant")
    APP_TENANTID = os.environ.get("MicrosoftAppTenantId", "")
    # Azure Endpoint
    ENDPOINT_URI = os.environ.get("MicrosoftServiceEndpoint", "")
    print(f"ENDPOINT URI: {ENDPOINT_URI}")
    AZURE_API_KEY = os.environ.get("MicrosoftServiceApiKey", "")
