# app/routers/chat.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
import uuid

from app.services.rag import get_chatbot_service, session_manager

router = APIRouter(prefix="/ws", tags=["Chat"])


@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for RAG-based chat."""
    await websocket.accept()

    # Create unique session ID
    session_id = f"ws_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"

    # Get chatbot service
    chatbot = get_chatbot_service(session_id)

    try:
        # Welcome message
        await websocket.send_json(
            {
                "type": "connected",
                "session_id": session_id,
                "message": "Connected to chat service",
            }
        )

        while True:
            # Receive message
            data = await websocket.receive_json()
            question = data.get("question")

            if question is None:
                await websocket.send_json(
                    {"type": "error", "error": "Missing 'question' key in request"}
                )
                continue

            if not question.strip():
                await websocket.send_json(
                    {"type": "error", "error": "Question cannot be empty"}
                )
                continue

            try:
                # Get response
                answer = await chatbot.chat(question)

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
        session_manager.remove(session_id)

    except Exception as e:
        print(f"WebSocket error: {e}")
        session_manager.remove(session_id)
        await websocket.close()
