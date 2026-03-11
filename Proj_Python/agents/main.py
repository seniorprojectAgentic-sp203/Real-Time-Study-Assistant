

import asyncio
import os
import base64
from typing import Optional
from dotenv import load_dotenv

# Google ADK imports
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai import types

# Import agents from existing file
# agent definitions in 'agents.py'
from agents import root_agent

# Load environment variables
load_dotenv()

# Check for API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# Configure Gemini
genai_config = types.GenerativeModelConfig(api_key=API_KEY)

# Set up services
session_service = InMemorySessionService()
artifact_service = InMemoryArtifactService()

# Create a persistent session
session = session_service.create_session(
    app_name="study-assistant",
    user_id="user_001",
    session_id="session_001"
)

# Create runner
runner = Runner(
    agent=root_agent,
    session_service=session_service,
    artifact_service=artifact_service,
)

async def send_message(message: str, image_bytes: Optional[bytes] = None) -> str:
    """
    Send a message to the agent and get the response
    
    Args:
        message: The user's text input
        image_bytes: Optional image data as bytes
    
    Returns:
        The agent's response as a string
    """
    # Build message parts
    parts = [types.Part(text=message)]
    
    # Add image if provided
    if image_bytes:
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    data=image_bytes,
                    mime_type="image/jpeg"  # Adjust mime type as needed
                )
            )
        )
    
    # Create the message content
    content = types.Content(parts=parts, role="user")
    
    # Run the agent and collect response
    final_response = ""
    async for event in runner.run_async(
        session_id="session_001",
        user_id="user_001",
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text
            break
    
    return final_response

async def interactive_cli():
    """Run an interactive command-line interface"""
    print("\n" + "="*60)
    print("📚 REAL-TIME MULTIMODAL STUDY ASSISTANT".center(60))
    print("="*60)
    print("\nCommands:")
    print("  /image <path>  - Attach an image to your next message")
    print("  /quit          - Exit the assistant")
    print("  /help          - Show this help message")
    print("\n" + "-"*60)
    
    # Initialize with a greeting
    response = await send_message("Start conversation")
    print(f"\n📌 Assistant: {response}\n")
    
    pending_image = None
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            # Handle commands
            if user_input.lower() == '/quit':
                print("\n👋 Goodbye! Happy studying!\n")
                break
                
            elif user_input.lower() == '/help':
                print("\nCommands:")
                print("  /image <path>  - Attach an image to your next message")
                print("  /quit          - Exit the assistant")
                print("  /help          - Show this help message\n")
                continue
                
            elif user_input.lower().startswith('/image '):
                # Extract image path
                image_path = user_input[7:].strip()
                try:
                    with open(image_path, 'rb') as f:
                        pending_image = f.read()
                    print(f"Image loaded: {os.path.basename(image_path)}")
                    print("Now type your question about this image:")
                except FileNotFoundError:
                    print(f"Error: Image file '{image_path}' not found")
                    pending_image = None
                except Exception as e:
                    print(f"Error loading image: {e}")
                    pending_image = None
                continue
            
            elif not user_input and not pending_image:
                print("Please enter a message or use /image to attach an image")
                continue
            
            # Process the message
            print("Thinking...", end="", flush=True)
            
            # Use pending image if available, then clear it
            response = await send_message(user_input, pending_image)
            pending_image = None  # Clear image after use
            
            print("\r", end="")  # Clear the "Thinking..." line
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\n")

def main():
    """Main entry point"""
    asyncio.run(interactive_cli())

if __name__ == "__main__":
    main()