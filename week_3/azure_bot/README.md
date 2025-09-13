# EchoBot

Bot Framework v4 echo bot sample.

This bot has been created using [Bot Framework](https://dev.botframework.com), it shows how to create a simple bot that accepts input from the user and echoes it back.

## To try this sample

- Clone the repository

```bash
git clone https://github.com/Microsoft/botbuilder-samples.git
```

- In a terminal, navigate to `botbuilder-samples\samples\python\02.echo-bot` folder
- Activate your desired virtual environment
- In the terminal, type `pip install -r requirements.txt`
- Run your bot with `python app.py`

## Testing the bot using Bot Framework Emulator

[Bot Framework Emulator](https://github.com/microsoft/botframework-emulator) is a desktop application that allows bot developers to test and debug their bots on localhost or running remotely through a tunnel.

- Install the latest Bot Framework Emulator from [here](https://github.com/Microsoft/BotFramework-Emulator/releases)

### Connect to the bot using Bot Framework Emulator

- Launch Bot Framework Emulator
- File -> Open Bot
- Enter a Bot URL of `http://localhost:3978/api/messages`

## Interacting with the bot

Enter text in the emulator. The text will be echoed back by the bot.

## Deploy the bot to Azure

To learn more about deploying a bot to Azure, see [Deploy your bot to Azure](https://aka.ms/azuredeployment) for a complete list of deployment instructions.

## New Changes

- Config.py
  - Add new key for Azure Language Service Endpoint + Key
- App.py
  - Initialize the TextAnalyticsClient using Config Endpoint + Key
  - Update the function messages to:
    - Extract the request body
    - Run the sentiment analysis using the TextAnalyticsClient
    - Parse the response
    - Return the parsed data as Bot Response
  - Add error handling per action
