from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
import uuid

from app.services.rag import get_chatbot_service
from app.services.rag.chatbot import session_manager

router = APIRouter(prefix="/ws", tags=["Chat"])


@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for RAG-based chat."""
    await websocket.accept()

    # Create unique session ID for this connection
    session_id = f"ws_{uuid.uuid4().hex[:8]}_{datetime.now().timestamp()}"

    # Get chatbot service for this session
    chatbot = get_chatbot_service(session_id)

    try:
        # Send welcome message
        await websocket.send_json(
            {
                "type": "connected",
                "session_id": session_id,
                "message": "Connected to chat service",
            }
        )

        while True:
            # Receive JSON data from client
            data = await websocket.receive_json()

            # Extract the question
            question = data.get("question")

            if question is None:
                await websocket.send_json(
                    {"type": "error", "error": "Missing 'question' key in request"}
                )
                continue

            # Validate question
            if not question.strip():
                await websocket.send_json(
                    {"type": "error", "error": "Question cannot be empty"}
                )
                continue

            try:
                # Get response from chatbot (async)
                answer = await chatbot.chat(question)

                # Send response
                await websocket.send_json(
                    {
                        "type": "answer",
                        "answer": answer,
                        "session_id": session_id,
                    }
                )

            except Exception as e:
                print(f"Chat error: {e}")
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": "Failed to process your question. Please try again.",
                    }
                )

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        # Clean up session
        session_manager.remove(session_id)

    except Exception as e:
        print(f"WebSocket error: {e}")
        session_manager.remove(session_id)
        await websocket.close()
