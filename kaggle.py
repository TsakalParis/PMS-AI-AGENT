# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # PMS AI AGENT (Beds24)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:08:24.699479Z","iopub.execute_input":"2025-08-25T19:08:24.699733Z","iopub.status.idle":"2025-08-25T19:08:43.424664Z","shell.execute_reply.started":"2025-08-25T19:08:24.699716Z","shell.execute_reply":"2025-08-25T19:08:43.423644Z"},"jupyter":{"outputs_hidden":false}}
!pip install huggingface_hub
!pip install langchain==0.1.16 langchain_core==0.1.45 langgraph==0.0.33
!pip install -U langchain-community

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Initial Setup (Libraries, Paths, Data)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:08:43.426521Z","iopub.execute_input":"2025-08-25T19:08:43.426770Z","iopub.status.idle":"2025-08-25T19:08:43.597019Z","shell.execute_reply.started":"2025-08-25T19:08:43.426748Z","shell.execute_reply":"2025-08-25T19:08:43.596443Z"},"jupyter":{"outputs_hidden":false}}
import kagglehub
from pathlib import Path
import os
import json
import requests
from datetime import datetime, timedelta, timezone

# Configuration
SETUP_URL = "https://beds24.com/api/v2/authentication/setup"
TOKEN_URL = "https://beds24.com/api/v2/authentication/token"
DETAILS_URL = "https://beds24.com/api/v2/authentication/details"

# Kaggle paths
DATASET_PATH = Path(kagglehub.dataset_download("jacall/beds24-agent-data"))
INPUT_DIR = Path(DATASET_PATH)           # read-only dataset
WORKING_DIR = Path("/kaggle/working")    # writable space

# Define constants for the auth/token files
INVITE_CODE_FILE   = Path(DATASET_PATH) / "beds24_invite_code.json"
REFRESH_TOKEN_FILE = Path(DATASET_PATH) / "beds24_refresh_token.json"
AUTH_TOKEN_FILE    = Path(DATASET_PATH) / "beds24_auth_token.json"

# Define the mock messages directory once (outside the function)
MOCK_MESSAGES_DIR = Path(DATASET_PATH) / "mock_messages"

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Set up LLM

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:08:43.597702Z","iopub.execute_input":"2025-08-25T19:08:43.597877Z","iopub.status.idle":"2025-08-25T19:09:09.862985Z","shell.execute_reply.started":"2025-08-25T19:08:43.597862Z","shell.execute_reply":"2025-08-25T19:09:09.862354Z"},"jupyter":{"outputs_hidden":false}}
# Install pre-built wheel for llama-cpp-python with CUDA support
!pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Now continue with your code
import yaml
from pathlib import Path

# Load prompts
with open(Path(DATASET_PATH) / "prompt.yaml", "r") as f:
    prompts = yaml.safe_load(f)

system_prompt = prompts["system_prompt"]
user_prompt_template = prompts["user_prompt_template"]
init_message = prompts["initialization_message"]

# Load hf_token
with open(Path(DATASET_PATH) / "hf_token.txt", "r") as f:
    hf_token = f.read().strip()

# Download model
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

model_path = hf_hub_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    token=hf_token
)

# model_path = hf_hub_download(
#     repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
#     filename="Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf",
#     token=hf_token  
# )

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:09:09.864504Z","iopub.execute_input":"2025-08-25T19:09:09.864696Z","iopub.status.idle":"2025-08-25T19:09:11.870066Z","shell.execute_reply.started":"2025-08-25T19:09:09.864680Z","shell.execute_reply":"2025-08-25T19:09:11.868766Z"},"jupyter":{"outputs_hidden":false}}
llm = Llama(
    model_path=model_path,
    n_ctx=4000,
    n_threads=8,
    n_gpu_layers=35,  # Can use more layers with smaller model
    verbose=False
)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:09:11.870976Z","iopub.execute_input":"2025-08-25T19:09:11.871279Z","iopub.status.idle":"2025-08-25T19:09:16.991867Z","shell.execute_reply.started":"2025-08-25T19:09:11.871260Z","shell.execute_reply":"2025-08-25T19:09:16.991158Z"},"jupyter":{"outputs_hidden":false}}
# Test the model with a simple prompt
output = llm(
    "Q: What is the capital of France? A:",
    max_tokens=32,
    stop=["Q:", "\n"],
    echo=False
)

print(output['choices'][0]['text'])

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:09:16.992567Z","iopub.execute_input":"2025-08-25T19:09:16.992791Z","iopub.status.idle":"2025-08-25T19:09:16.997090Z","shell.execute_reply.started":"2025-08-25T19:09:16.992774Z","shell.execute_reply":"2025-08-25T19:09:16.996238Z"},"jupyter":{"outputs_hidden":false}}
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import tool

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Authentication Manager

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:09:16.997905Z","iopub.execute_input":"2025-08-25T19:09:16.998079Z","iopub.status.idle":"2025-08-25T19:09:17.022385Z","shell.execute_reply.started":"2025-08-25T19:09:16.998065Z","shell.execute_reply":"2025-08-25T19:09:17.021857Z"},"jupyter":{"source_hidden":true}}
class Beds24AuthManager:
    def __init__(self, logger=None):
        self.logger = logger or print
        self.invite_code_data = self._load_file(INVITE_CODE_FILE)
        self.refresh_token_data = self._load_file(REFRESH_TOKEN_FILE)
        self.auth_token_data = self._load_file(AUTH_TOKEN_FILE)
    
    def _resolve_path(self, filename):
        """Look for file in /kaggle/working first, then dataset input"""
        working_path = WORKING_DIR / filename
        if working_path.exists():
            return working_path
        return INPUT_DIR / filename
    
    def _load_file(self, filename):
        """Load JSON data from file, return empty dict if missing or invalid"""
        filepath = self._resolve_path(filename)
        if filepath.exists() and filepath.stat().st_size > 0:
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_file(self, filename, data):
        """Always save to /kaggle/working (writable)"""
        filepath = WORKING_DIR / filename
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except IOError as e:
            print(f"❌ Failed to save {filename}: {e}")
            return False

    def _has_not_expired(self, data):
        """Check if token/credentials have not expired using UTC time"""
        if not data or "expiration" not in data:
            return False
        try:
            expiration_date = datetime.fromisoformat(data["expiration"])
            if expiration_date.tzinfo is None:
                expiration_date = expiration_date.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) < expiration_date
        except (ValueError, TypeError) as e:
            print(f"Error parsing expiration date: {str(e)}")
            return False

    def _get_token_details(self, token):
        """Get information about a token using the details endpoint"""
        headers = {"accept": "application/json", "token": token}
        try:
            response = requests.get(DETAILS_URL, headers=headers)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error getting token details: {str(e)}")
        return None

    def setup_with_invite_code(self, invite_code, expiration_days=30):
        """Generate initial tokens using invite code"""
        if not invite_code:
            return False
        headers = {"accept": "application/json", "code": invite_code}
        try:
            response = requests.get(SETUP_URL, headers=headers)
            response.raise_for_status()
            token_data = response.json()

            # Calculate expiration dates
            expires_in = token_data.get("expiresIn", 86400)
            current_utc = datetime.now(timezone.utc)
            auth_expiration = current_utc + timedelta(seconds=expires_in)
            refresh_expiration = current_utc + timedelta(days=expiration_days)

            # Save auth token
            auth_data = {
                "access_token": token_data["token"],
                "created": current_utc.isoformat(),
                "expiration": auth_expiration.isoformat(),
            }
            self._save_file(AUTH_TOKEN_FILE, auth_data)

            # Save refresh token
            refresh_data = {
                "refresh_token": token_data["refreshToken"],
                "created": current_utc.isoformat(),
                "expiration": refresh_expiration.isoformat(),
            }
            self._save_file(REFRESH_TOKEN_FILE, refresh_data)

            # Remove invite code after successful setup
            invite_path = WORKING_DIR / INVITE_CODE_FILE
            if invite_path.exists():
                os.remove(invite_path)

            print("✅ Setup completed successfully")
            return True
        except Exception as e:
            print(f"❌ Setup failed: {str(e)}")
            return False

    def refresh_auth_token(self):
        """Refresh access token using refresh token (GET with header)"""
        if not self.refresh_token_data or not self._has_not_expired(self.refresh_token_data):
            print("❌ No valid refresh token available")
            return False

        refresh_token = self.refresh_token_data.get("refresh_token")
        if not refresh_token:
            return False

        headers = {"accept": "application/json", "refreshToken": refresh_token}

        try:
            response = requests.get(TOKEN_URL, headers=headers)
            response.raise_for_status()
            token_data = response.json()

            if "token" not in token_data:
                print("❌ No access token in refresh response")
                return False

            expires_in = token_data.get("expiresIn", 86400)
            current_utc = datetime.now(timezone.utc)
            expiration = current_utc + timedelta(seconds=expires_in)

            # Save new auth token
            auth_data = {
                "access_token": token_data["token"],
                "created": current_utc.isoformat(),
                "expiration": expiration.isoformat(),
            }
            self._save_file(AUTH_TOKEN_FILE, auth_data)
            self.auth_token_data = auth_data

            # Verify token
            if self.validate_token(auth_data["access_token"]):
                print("✅ Token refresh completed successfully")
                return True
            else:
                print("❌ New token failed validation")
                return False

        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                print("❌ Refresh token rejected - may be invalid/expired")
            else:
                print(f"❌ Token refresh failed: {str(e)}")
            return False
        except Exception as e:
            print(f"❌ Token refresh failed: {str(e)}")
            return False

    def get_valid_token(self):
        """Get a valid access token, refreshing if necessary"""
        if self.auth_token_data and self._has_not_expired(self.auth_token_data):
            print("✅ Using existing valid auth token")
            return self.auth_token_data.get("access_token")

        if self.refresh_token_data and self._has_not_expired(self.refresh_token_data):
            print("🔄 Auth token expired, attempting refresh using refresh token")
            if self.refresh_auth_token():
                return self.auth_token_data.get("access_token")

        print("❌ No valid authentication method available")
        return None

    def check_token_status(self):
        """Check status of all tokens"""
        status = {
            "auth_token": {
                "exists": bool(self.auth_token_data),
                "valid": self.auth_token_data and self._has_not_expired(self.auth_token_data),
                "expired": self.auth_token_data and not self._has_not_expired(self.auth_token_data),
            },
            "refresh_token": {
                "exists": bool(self.refresh_token_data),
                "valid": self.refresh_token_data and self._has_not_expired(self.refresh_token_data),
                "expired": self.refresh_token_data and not self._has_not_expired(self.refresh_token_data),
            },
            "invite_code": {
                "exists": bool(self.invite_code_data),
                "valid": self.invite_code_data and self._has_not_expired(self.invite_code_data),
                "expired": self.invite_code_data and not self._has_not_expired(self.invite_code_data),
            },
        }
        return status

    def validate_token(self, token):
        """Validate a token using details endpoint"""
        details = self._get_token_details(token)
        return bool(details and details.get("validToken", False))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Agent Tools

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:09:17.023089Z","iopub.execute_input":"2025-08-25T19:09:17.023272Z","iopub.status.idle":"2025-08-25T19:09:17.038189Z","shell.execute_reply.started":"2025-08-25T19:09:17.023248Z","shell.execute_reply":"2025-08-25T19:09:17.037651Z"}}
class Beds24Bookings:
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self.base_url = "https://beds24.com/api/v2/bookings"
        self.messages_url = f"{self.base_url}/messages"
        
    def _get_headers(self):
        """Get headers with authentication token"""
        token = self.auth_manager.get_valid_token()
        if not token:
            return None
            
        return {
            "accept": "application/json",
            "content-type": "application/json",
            "token": token
        }
        
    def get_messages(self, **params):
        """
        Fetches booking messages from the Beds24 API with optional filters.
        """
        headers = self._get_headers()
        if not headers:
            return None

        try:
            response = requests.get(self.messages_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to get messages: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

    def post_message(self, messages_data):
        """
        Sends new message(s) to guest(s) through the Beds24 platform.
        """
        headers = self._get_headers()
        if not headers:
            return None

        try:
            response = requests.post(self.messages_url, headers=headers, json=messages_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to post message: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

    def get_bookings(self, **params):
        """
        Retrieves booking records from Beds24 with advanced filtering capabilities.

        """
        headers = self._get_headers()
        if not headers:
            return None

        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Failed to get bookings: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            return None

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:09:17.039018Z","iopub.execute_input":"2025-08-25T19:09:17.039382Z","iopub.status.idle":"2025-08-25T19:09:17.060651Z","shell.execute_reply.started":"2025-08-25T19:09:17.039364Z","shell.execute_reply":"2025-08-25T19:09:17.060112Z"}}
class Beds24Inventory:
    def __init__(self, auth_manager, property_id=289195):
        """
        Initializes the Beds24 inventory management client.
        """
        self.auth_manager = auth_manager
        self.base_url = "https://beds24.com/api/v2"
        self.property_id = property_id
        

    def _make_request(self, endpoint, params=None):
        """
        Internal method to make authenticated HTTP requests to the Beds24 API.
        """
        # Get a valid access token
        token = self.auth_manager.get_valid_token()
        if not token:
            print("❌ No valid authentication token available")
            return None
        
        # Set up headers
        headers = {
            "accept": "application/json",
            "token": token
        }
        
        # Make the request
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ API request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error making API request: {str(e)}")
            return None
            
    def get_availability(self, room_id=None, from_date=None, to_date=None, detailed=False):
        """
        Retrieves detailed availability information for rooms in a date range.
        """
        # Build query parameters
        params = {"propertyId": self.property_id}
        
        if room_id:
            params["roomId"] = room_id
        
        # Set default date range if not provided (1 year from now)
        if not from_date:
            from_date = datetime.now().strftime("%Y-%m-%d")
            
        if not to_date:
            next_year = datetime.now() + timedelta(days=365)
            to_date = next_year.strftime("%Y-%m-%d")
        
        params["fromDate"] = from_date
        params["toDate"] = to_date
        
        # Add detailed parameter if requested
        if detailed:
            params["includeRoomInfo"] = "true"
        
        return self._make_request("inventory/availability", params)

    def get_availability_summary(self, from_date=None, to_date=None):
        """
        Generates a summarized report of room availability across all rooms.
        """
        availability_data = self.get_availability(from_date=from_date, to_date=to_date, detailed=True)
        
        if not availability_data or 'data' not in availability_data:
            return {"error": "Failed to fetch availability data"}
        
        summary = {
            "total_rooms": len(availability_data['data']),
            "rooms": [],
            "date_range": {
                "from": from_date,
                "to": to_date
            }
        }
        
        for room_data in availability_data['data']:
            room_info = room_data.get('roomInfo', {})
            availability = room_data.get('availability', {})
            
            # Count available days - handle both boolean and dictionary formats
            available_days = 0
            for day_status in availability.values():
                if isinstance(day_status, bool):
                    if day_status:  # True means available
                        available_days += 1
                elif isinstance(day_status, dict):
                    if day_status.get('available', False):
                        available_days += 1
            
            total_days = len(availability)
            
            room_summary = {
                "room_id": room_data.get('roomId'),
                "room_name": room_info.get('name', 'Unknown'),
                "available_days": available_days,
                "total_days": total_days,
                "occupancy_rate": round((available_days / total_days * 100), 2) if total_days > 0 else 0
            }
            
            summary['rooms'].append(room_summary)
        
        return summary
    

    def check_date_availability(self, check_date, room_id=None):
        """
        Checks availability for a specific date across one or all rooms.
        """
        availability_data = self.get_availability(
            room_id=room_id, 
            from_date=check_date, 
            to_date=check_date,
            detailed=True
        )
        
        if not availability_data or 'data' not in availability_data:
            return {"error": "Failed to fetch availability data"}
        
        result = {
            "date": check_date,
            "rooms_available": 0,
            "rooms": []
        }
        
        for room_data in availability_data['data']:
            room_info = room_data.get('roomInfo', {})
            availability = room_data.get('availability', {})
            
            # Check if the specific date is available - handle both boolean and dictionary formats
            day_status = availability.get(check_date, False)
            if isinstance(day_status, bool):
                is_available = day_status
            elif isinstance(day_status, dict):
                is_available = day_status.get('available', False)
            else:
                is_available = False
            
            room_status = {
                "room_id": room_data.get('roomId'),
                "room_name": room_info.get('name', 'Unknown'),
                "available": is_available
            }
            
            if is_available:
                result['rooms_available'] += 1
                
            result['rooms'].append(room_status)
        
        return result

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:09:17.062996Z","iopub.execute_input":"2025-08-25T19:09:17.063184Z","iopub.status.idle":"2025-08-25T19:09:17.081308Z","shell.execute_reply.started":"2025-08-25T19:09:17.063169Z","shell.execute_reply":"2025-08-25T19:09:17.080739Z"}}
def extract_message_info(directory="mock_messages"):
    """
    Extracts message information from mock JSON files in a specified directory.
    """
    extracted_data = []
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return extracted_data
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in '{directory}' directory.")
        return extracted_data
    
    # Process each JSON file
    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract the required fields
                booking_id = data.get('bookingId', 'N/A')
                subject = data.get('subject', 'No Subject')
                message = data.get('message', 'No Message')
                
                # Add to our results
                extracted_data.append({
                    'file': json_file,
                    'bookingId': booking_id,
                    'subject': subject,
                    'message': message
                })
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error processing file {json_file}: {str(e)}")
            continue
    
    return extracted_data

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:09:17.081987Z","iopub.execute_input":"2025-08-25T19:09:17.082170Z","iopub.status.idle":"2025-08-25T19:09:17.123766Z","shell.execute_reply.started":"2025-08-25T19:09:17.082156Z","shell.execute_reply":"2025-08-25T19:09:17.123245Z"}}
# Assign tools
from functools import partial
from typing import Optional

auth_manager = Beds24AuthManager()
bookings = Beds24Bookings(auth_manager)
inventory = Beds24Inventory(auth_manager)

# Create tool functions that wrap your class methods
@tool
def get_messages_tool(bookingId: Optional[str] = None, dateFrom: Optional[str] = None, dateTo: Optional[str] = None):
    """Fetches booking messages. Use bookingId, dateFrom, and dateTo as filters."""
    params = {}
    if bookingId:
        params['bookingId'] = bookingId
    if dateFrom:
        params['dateFrom'] = dateFrom
    if dateTo:
        params['dateTo'] = dateTo
    return bookings.get_messages(**params)

@tool
def post_message_tool(bookingId: str, subject: str, message: str):
    """Post a message to a guest. Requires booking ID, subject, and message content."""
    messages_data = {
        "bookingId": bookingId,
        "subject": subject,
        "message": message
    }
    return bookings.post_message(messages_data)

@tool
def get_bookings_tool(dateFrom: Optional[str] = None, dateTo: Optional[str] = None, status: Optional[str] = None):
    """Retrieve bookings filtered by date range (YYYY-MM-DD) and/or status."""
    params = {}
    if dateFrom:
        params['dateFrom'] = dateFrom
    if dateTo:
        params['dateTo'] = dateTo
    if status:
        params['status'] = status
    return bookings.get_bookings(**params)

@tool
def get_availability_summary_tool(from_date: Optional[str] = None, to_date: Optional[str] = None):
    """Get room availability summary between dates (YYYY-MM-DD format)."""
    return inventory.get_availability_summary(from_date=from_date, to_date=to_date)

@tool
def check_date_availability_tool(check_date: str, room_id: Optional[str] = None):
    """Check availability for a specific date (YYYY-MM-DD) and optional room ID."""
    return inventory.check_date_availability(check_date=check_date, room_id=room_id)

@tool
def extract_message_info_tool():
    """Extract message information from mock data files for testing."""
    return extract_message_info()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-08-25T19:09:17.124370Z","iopub.execute_input":"2025-08-25T19:09:17.124636Z","iopub.status.idle":"2025-08-25T19:09:17.128305Z","shell.execute_reply.started":"2025-08-25T19:09:17.124612Z","shell.execute_reply":"2025-08-25T19:09:17.127724Z"}}
# Create your tools list
tools = [
    get_messages_tool,
    post_message_tool,
    get_bookings_tool,
    get_availability_summary_tool,
    check_date_availability_tool,
    extract_message_info_tool,
]

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Langchain & Langgraph Setup

# %% [code] {"execution":{"iopub.status.busy":"2025-08-25T19:09:17.129040Z","iopub.execute_input":"2025-08-25T19:09:17.129549Z","iopub.status.idle":"2025-08-25T19:09:17.148412Z","shell.execute_reply.started":"2025-08-25T19:09:17.129528Z","shell.execute_reply":"2025-08-25T19:09:17.147756Z"},"jupyter":{"outputs_hidden":false}}
# First, let's create a proper LangChain LLM wrapper
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from typing import List, Optional, Dict, Any

class DolphinLangChainLLM(BaseLLM):
    def _generate(
        self, 
        prompts: List[str], 
        stop: Optional[List[str]] = None, 
        **kwargs: Any
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            # Call llama-cpp
            response = llm(prompt, max_tokens=512, stop=stop)
            # Extract text
            text = response["choices"][0]["text"].strip()
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        return "dolphin-llm"

# Create the LangChain compatible LLM
dolphin_llm_langchain = DolphinLangChainLLM()

# Create memory for the agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the agent using initialize_agent
agent = initialize_agent(
    tools,
    dolphin_llm_langchain,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Gradui UI

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-08-25T19:09:17.149190Z","iopub.execute_input":"2025-08-25T19:09:17.149408Z","iopub.status.idle":"2025-08-25T19:09:17.787007Z","shell.execute_reply.started":"2025-08-25T19:09:17.149392Z","shell.execute_reply":"2025-08-25T19:09:17.786379Z"}}
import gradio as gr
import json

# Initialize chat with the initialization message
def initialize_chat():
    return [{"role": "assistant", "content": init_message}]

# Update the chat_with_agent function
def chat_with_agent(user_input, history):
    try:
        # Clear memory and rebuild from history
        memory.clear()
        for h in history:
            if h["role"] == "user":
                memory.chat_memory.add_user_message(h["content"])
            else:
                memory.chat_memory.add_ai_message(h["content"])
        
        # Run the agent
        response = agent.invoke(input=user_input)
        
        # Extract just the output text from the response
        if isinstance(response, dict) and 'output' in response:
            output_text = response['output']
        else:
            output_text = str(response)
        
        # Add to history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": output_text})
        
        return history, history
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": error_msg})
        return history, history

auth_manager = Beds24AuthManager()

def run_authentication():
    logs = []
    auth_manager.logger = lambda msg: logs.append(msg)

    # Token status
    status = auth_manager.check_token_status()
    logs.append("### Token Status")
    logs.append("```json\n" + json.dumps(status, indent=2) + "\n```")

    # Setup with invite code if needed
    if (status['auth_token']['expired'] and 
        status['refresh_token']['expired'] and 
        status['invite_code']['exists'] and 
        status['invite_code']['valid']):
        logs.append("🔑 Starting setup with invite code...")
        invite_code = auth_manager.invite_code_data.get("invite_code")
        auth_manager.setup_with_invite_code(invite_code)

    token = auth_manager.get_valid_token()

    if token:
        logs.append(f"✅ Valid token obtained: {token[:15]}...")
        if auth_manager.validate_token(token):
            logs.append("✅ Token validation successful")
            return "\n\n".join(logs), gr.update(visible=True), gr.update(visible=False), initialize_chat()
        else:
            logs.append("❌ Token validation failed")
    else:
        logs.append("❌ Failed to obtain valid token")
        status = auth_manager.check_token_status()
        if (status['invite_code']['exists'] and status['invite_code']['valid'] and
            not status['refresh_token']['exists']):
            logs.append("🔑 Attempting initial setup with invite code...")
            invite_code = auth_manager.invite_code_data.get("invite_code")
            if auth_manager.setup_with_invite_code(invite_code):
                token = auth_manager.get_valid_token()
                if token:
                    logs.append(f"✅ Valid token obtained after setup: {token[:15]}...")
                    return "\n\n".join(logs), gr.update(visible=True), gr.update(visible=False), initialize_chat()

    return "\n\n".join(logs), gr.update(visible=False), gr.update(visible=True), gr.update()

with gr.Blocks() as demo:
    gr.Markdown("## Beds24 Agent")
    gr.Markdown(
        "⚠️ **Please update `beds24_invite_code.json` with your Invite Code, "
        "Created date, and Expiration date before starting the authentication process.**"
    )

    auth_btn = gr.Button("Start authentication process")
    status_msg = gr.Markdown("")
    
    # Store chat history in session state
    chat_history = gr.State(initialize_chat())

    # Chat UI (hidden until auth success)
    with gr.Column(visible=False) as chat_col:
        chatbot = gr.Chatbot(type="messages", value=initialize_chat())
        with gr.Row():
            msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
            send = gr.Button("Send")

    fail_msg = gr.Markdown("", visible=False)

    # Update authentication to also return chat history
    auth_btn.click(
        run_authentication, 
        [], 
        [status_msg, chat_col, fail_msg, chat_history]
    )
    
    # Update chat functions to use the stored history
    send.click(
        chat_with_agent, 
        [msg, chat_history], 
        [chatbot, chat_history]
    )
    msg.submit(
        chat_with_agent, 
        [msg, chat_history], 
        [chatbot, chat_history]
    )

demo.launch()

# %% [code] {"jupyter":{"outputs_hidden":false}}
