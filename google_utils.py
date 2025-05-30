import argparse
import os
import pandas as pd
import gspread
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

def create_google_sheet_with_dataframe(client_secret_file, spreadsheet_name, dataframe, table_title):
    """
    Args:
        client_secret_file (str): Path to the client_secret.json file for OAuth2 credentials.
        spreadsheet_name (str): The name of the new Google Spreadsheet.
        dataframe (pd.DataFrame): The DataFrame to insert as a table.
        table_title (str): The title for the table.
    """
    # Define the scope (spreadsheets and google drive)
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/spreadsheets.readonly'
    ]
    
    # Authenticate using OAuth2
    creds = None
    token_file = 'token.json'
    
    # Check if token.json file exists and load the credentials
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    
    # If there are no (valid) credentials available, prompt the user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    
    # Build the service
    service = build('sheets', 'v4', credentials=creds)
    
    # Create a new spreadsheet
    spreadsheet = {
        'properties': {
            'title': spreadsheet_name
        }
    }
    spreadsheet = service.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
    spreadsheet_id = spreadsheet.get('spreadsheetId')
    
    # Open the spreadsheet
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(spreadsheet_id).sheet1

    # Insert the table title
    sheet.update('A1', [[table_title]])
    sheet.format('A1', {'textFormat': {'bold': True, 'fontSize': 14}})

    # Insert the DataFrame as a table starting from cell A2
    sheet.update('A2', [dataframe.columns.values.tolist()] + dataframe.values.tolist())

    print(f"Spreadsheet '{spreadsheet_name}' created with table.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize Google OAuth token.")
    parser.add_argument('--client-secret', required=True, help='Path to client_secret.json')
    args = parser.parse_args()

    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/spreadsheets.readonly'
    ]
    token_file = 'token.json'

    if not os.path.exists(token_file):
        flow = InstalledAppFlow.from_client_secrets_file(args.client_secret, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
        print("Token created and saved to token.json.")
    else:
        print("Token already exists.")








