from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from route_tracer import Tracer, VehicleNotFound, VEHICLE_LABEL, INITIAL_LOCATION

app = FastAPI()


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    async def intialise(self, websocket: WebSocket):
        resp = await websocket.receive_json()
        print(resp)
        data = resp["DATA"]
        self.route_tracer = Tracer(
            v_type=data["v_type"], 
            c_id=int(data["c_id"]), 
            start_time=float(data["start_time"]), 
            end_time=float(data["end_time"])
        )

    async def loop(self, websocket: WebSocket):
        if self.route_tracer.vehicle is None:
            await websocket.send_json({
                "TYPE": "SELECT_VEHICLES",
                "DATA": self.route_tracer.vehicle_list
            })
            resp = await websocket.receive_json()
            print(resp)
            vehicle_index = resp["DATA"].get("vid")
            self.route_tracer.vehicle = self.route_tracer.vehicle_list[vehicle_index]
        self.route_tracer.trace_loop()
        await websocket.send_json({
            "TYPE": "DISPLAY",
            "DATA": self.route_tracer.get_trace_status()
        })

    async def trace_end(self, websocket: WebSocket):
        await websocket.send_json({
            "TYPE": "END",
            "DATA": self.route_tracer.get_trace_status()
        })
    

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        self.route_tracer = None

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.get("/")
async def get():
    return JSONResponse({ "vehicles": VEHICLE_LABEL, "locations": INITIAL_LOCATION})



@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        try:
            await manager.intialise(websocket)
            while True:
                await manager.loop(websocket)
        except VehicleNotFound:
            await manager.trace_end(websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        