from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/ws", tags=["KnowledgeBase"])


@router.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # Receive JSON data from client
            data = await websocket.receive_json()

            # Extract the question
            question = data.get("question")

            if question is None:
                await websocket.send_json(
                    {"error": "Missing 'question' key in request"}
                )
                continue

            # Send constant response (for now)
            response = {"answer": "This is a constant response"}

            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
