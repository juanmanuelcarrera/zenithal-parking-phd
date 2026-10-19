"""MQTT publisher for occupancy events (QoS 1, at least once)."""
import paho.mqtt.client as mqtt


class EventPublisher:
    def __init__(self, node_id: str, host: str = "localhost", port: int = 1883):
        self.node_id = node_id
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=node_id)
        self.client.connect(host, port)
        self.client.loop_start()

    def publish(self, event) -> None:
        topic = f"parking/{self.node_id}/spots/{event.spot_id}"
        self.client.publish(topic, event.SerializeToString(), qos=1)

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()
