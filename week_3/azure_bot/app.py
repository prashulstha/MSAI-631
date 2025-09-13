# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import traceback
from datetime import datetime, timezone
from http import HTTPStatus

from aiohttp import web
from aiohttp.web import Request, Response, json_response
from botbuilder.core import (
    TurnContext,
)
from botbuilder.core.integration import aiohttp_error_middleware
from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication
from botbuilder.schema import Activity, ActivityTypes
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from bots import EchoBot
from config import DefaultConfig

CONFIG = DefaultConfig()

# 2025-09-13: START Extended for MSAI 631 - Week 3 Project to Add Azure Language Service
credential = AzureKeyCredential(CONFIG.AZURE_API_KEY)
endpointURI = CONFIG.ENDPOINT_URI
text_analytics_client = TextAnalyticsClient(endpoint=endpointURI, credential=credential)
#2025-09-13: END Extended for MSAI 631 Week 3 Project

# Create adapter.
# See https://aka.ms/about-bot-adapter to learn more about how bots work.
ADAPTER = CloudAdapter(ConfigurationBotFrameworkAuthentication(CONFIG))


# Catch-all for errors.
async def on_error(context: TurnContext, error: Exception):
    # This check writes out errors to console log .vs. app insights.
    # NOTE: In production environment, you should consider logging this to Azure
    #       application insights.
    print(f"\n [on_turn_error] unhandled error: {error}", file=sys.stderr)
    traceback.print_exc()

    # Send a message to the user
    await context.send_activity("The bot encountered an error or bug.")
    await context.send_activity(
        "To continue to run this bot, please fix the bot source code."
    )
    # Send a trace activity if we're talking to the Bot Framework Emulator
    if context.activity.channel_id == "emulator":
        # Create a trace activity that contains the error object
        trace_activity = Activity(
            label="TurnError",
            name="on_turn_error Trace",
            timestamp=datetime.now(timezone.utc),
            value=f"{error}",
            value_type="https://www.botframework.com/schemas/error",
        )
        # Send a trace activity, which will be displayed in Bot Framework Emulator
        await context.send_activity(trace_activity)


ADAPTER.on_turn_error = on_error

# Create the Bot
BOT = EchoBot()


# Listen for incoming requests on /api/messages
async def messages(req: Request) -> Response:
    try:
        # If the request is not of json type - return unsupported response
        if "application/json" not in req.headers["Content-Type"]:
            return Response(status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        # 2025-09-13: START MSAI 631 Azure Sentiment Analysis
        # Process the data
        data = await req.json()
        text = data.get("text", "")
        if not text:
            return await ADAPTER.process(req, BOT)
        documents = [{"id": "1", "language": "en", "text": text}]
        azure_response = text_analytics_client.analyze_sentiment(documents)
        if not azure_response:
            return Response(status=HTTPStatus.BAD_GATEWAY, text="No response from Azure Sentiment Analysis")
        
        successful_responses = [doc for doc in azure_response if not doc.is_error]
        data['text'] = successful_responses
        activity = Activity().deserialize(data)
        auth_header = req.headers['Authorization'] if "Authorization" in req.headers else ""
        bot_response = await ADAPTER.process_activity(auth_header, activity, BOT.on_message_activity)
        if bot_response:
            return json_response(status=HTTPStatus.OK, data=bot_response.body)
        return Response(status=HTTPStatus.OK)
        # 2025-09-13: END MSAI 631 Azure Sentiment Analysis
    except Exception as e:
        print(f"Error {str(e)}")
        return Response(status=HTTPStatus.BAD_REQUEST)

APP = web.Application(middlewares=[aiohttp_error_middleware])
APP.router.add_post("/api/messages", messages)

if __name__ == "__main__":
    try:
        web.run_app(APP, host="localhost", port=CONFIG.PORT)
    except Exception as error:
        raise error
