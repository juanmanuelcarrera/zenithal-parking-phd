"""Payload size of one event, Protobuf vs JSON."""
from occupancy import OccupancyState

from .events import build_event, to_json

event = build_event("raspi_sector_A", 1024, OccupancyState.OCCUPIED, 0.92, timestamp=1678900000)
binary = event.SerializeToString()
text = to_json(event)

print(f"protobuf: {len(binary)} bytes")
print(" ".join(f"{b:02X}" for b in binary))
print(f"json:     {len(text)} bytes")
print(text.decode())
print(f"saving:   {100 * (1 - len(binary) / len(text)):.1f} %")
