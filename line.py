from flask import Flask, request
from linebot.models import *
from linebot import *
from vertex_ai import prompt_ai
import datetime
import requests  # Import the requests library

app = Flask(__name__)



@app.route("/healthz", methods=['GET'])
def health_check():
    return "health check"

@app.route("/callback", methods=['POST'])
def callback():
    body = request.get_data(as_text=True)
    print(body)
    req = request.get_json(silent=True, force=True)
    events = req['events']
    destination = req['destination']
    if events:
        event = events[0]
        reply_token = event['replyToken']
        text = event['message']['text']
        user_id = event['source']['userId']  # Extract the user ID from the event data

        print(f"Reply Token: {reply_token}")
        print(f"Text: {text}")
        print(f"req: {req}")

        # Call LINE API to get user profile
        user_profile_url = f"https://api.line.me/v2/bot/profile/{user_id}"
        bearer = f'Bearer {line_bot_api_test}'
        headers = {
            'Authorization': bearer
        }

        # Make the GET request to fetch user profile
        try:
            profile_response = requests.get(user_profile_url, headers=headers)
            if profile_response.status_code == 200:
                user_profile = profile_response.json()
                display_name = user_profile.get('displayName', 'User')
                print(f"User Display Name: {display_name}")
            else:
                print(f"Failed to fetch user profile: {profile_response.status_code}")
        except Exception as e:
            print(f"Error fetching user profile: {e}")

        # Get response from prompt_ai
        response = prompt_ai(text, session_id=destination, user_name=display_name)

        # Reply with the AI response
        line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{response}"))

    return 'OK'

def reply(intent, text, reply_token, id, disname, lotto_number, session):
    print(text)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)
