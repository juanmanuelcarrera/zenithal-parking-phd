"""Build and serialise occupancy events. Requires parking_event_pb2 (make proto)."""
import json
import time

from occupancy import OccupancyState

from . import parking_event_pb2 as pb

STATUS = {
    OccupancyState.FREE: pb.ParkingEvent.FREE,
    OccupancyState.LOW: pb.ParkingEvent.PARTIAL,
    OccupancyState.HIGH: pb.ParkingEvent.PARTIAL,
    OccupancyState.OCCUPIED: pb.ParkingEvent.OCCUPIED,
}


def build_event(node_id: str, spot_id: int, state: OccupancyState, ratio: float,
                timestamp: int = None) -> pb.ParkingEvent:
    return pb.ParkingEvent(
        node_id=node_id,
        spot_id=spot_id,
        timestamp=timestamp or int(time.time()),
        status=STATUS[state],
        occupied_area=ratio,
        vehicle_type=pb.ParkingEvent.CAR,
    )


def to_json(event: pb.ParkingEvent) -> bytes:
    return json.dumps({
        "node_id": event.node_id,
        "spot_id": event.spot_id,
        "timestamp": event.timestamp,
        "status": pb.ParkingEvent.Status.Name(event.status),
        "occupied_area": round(event.occupied_area, 2),
        "vehicle_type": pb.ParkingEvent.VehicleType.Name(event.vehicle_type),
    }).encode()
