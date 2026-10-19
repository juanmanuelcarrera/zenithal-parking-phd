# Zenithal Parking

Research code from the PhD thesis *Inferencia de ocupación y asignación dinámica de aparcamientos mediante visión artificial y aprendizaje profundo* (Universidad Complutense de Madrid, 2026).

The system replaces binary free/occupied detection with an **area-based occupancy estimate** computed from a zenithal camera, and assigns each incoming vehicle the spot that best fits its measured size.

Each folder is a self-contained proof of concept for one part of the system.

## Structure

| Folder | What it does |
|---|---|
| `mapping/` | Classic OpenCV pipeline that extracts spot polygons from an empty-lot image (filter → Otsu → Canny → contours → Douglas-Peucker → row grouping). Runs once at installation. |
| `detection/` | Vehicle instance segmentation with Mask R-CNN (Detectron2, R101-FPN) and YOLOv11-seg (Ultralytics): inference, fine-tuning from COCO, area-precision evaluation and latency benchmark. |
| `motion/` | Five motion detectors (MOG, MOG2, GMG, weighted average, frame difference) and a benchmark (time, FPS, CPU, memory). |
| `measurement/` | Pixel-to-metre calibration (γ = d_real / d_img), shoelace area, vehicle length and width in the entrance measurement region. |
| `occupancy/` | O_i = A(spot ∩ ∪ vehicles) / A(spot), four states (free < 10 %, low < 50 %, high < 90 %, occupied), 3-frame hysteresis, and the area-vs-binary scenario comparison. |
| `assignment/` | Best-fit assignment: size filter plus cost J = α·dist/D_max + β·(A(p) − A(v))/A(p). |
| `core-rs/` | Rust crate (`rust_parking_core`, Python module `parking_logic`) with the polygon geometry, exposed through PyO3. |
| `messaging/` | Protobuf event schema, MQTT publisher (QoS 1) and a JSON vs Protobuf size comparison. |
| `pipeline/` | Processing node loop: motion → YOLO sentinel → wait for a still scene → Mask R-CNN expert → occupancy → events. |
| `tests/` | Unit tests for geometry, occupancy and assignment. |
| `data/` | ZenithalParking dataset (scale model, VIA polygon annotations). |

## Data

`data/ZenithalParking.zip`: 66 zenithal images of the scale model (48 train / 18 val) with VIA polygon annotations, single class `car`. Trained weights are not included.

## License

Code: [MIT](LICENSE). Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
