import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Get environment variables
api_key = os.getenv('API_KEY')
print(api_key)  # prints: your-api-key

# Check if an environment variable is set
if 'API_KEY' in os.environ:
    print('API_KEY is set')
else:
    print('API_KEY is not set')

# Remove an environment variable (if it exists)
if 'API_KEY' in os.environ:
    del os.environ['API_KEY']