from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

from rich.progress import track

from pedsense.config import ANNOTATIONS_DIR


@dataclass
class BoundingBox:
    frame: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    occluded: bool
    outside: bool
    track_id: str = ""
    action: str = ""
    cross: str = ""
    look: str = ""
    occlusion: str = "none"


@dataclass
class Track:
    label: str
    boxes: list[BoundingBox] = field(default_factory=list)


@dataclass
class VideoAnnotation:
    video_id: str
    num_frames: int
    width: int
    height: int
    time_of_day: str = ""
    weather: str = ""
    location: str = ""
    tracks: list[Track] = field(default_factory=list)


def parse_annotation(xml_path: Path) -> VideoAnnotation:
    """Parse a single CVAT XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    meta = root.find("meta/task")
    video_id = xml_path.stem

    # Extract video metadata
    num_frames = int(meta.findtext("size", "0"))
    orig_size = meta.find("original_size")
    width = int(orig_size.findtext("width", "1920"))
    height = int(orig_size.findtext("height", "1080"))

    video_attrs = meta.find("video_attributes")
    time_of_day = video_attrs.findtext("time_of_day", "") if video_attrs is not None else ""
    weather = video_attrs.findtext("weather", "") if video_attrs is not None else ""
    location = video_attrs.findtext("location", "") if video_attrs is not None else ""

    annotation = VideoAnnotation(
        video_id=video_id,
        num_frames=num_frames,
        width=width,
        height=height,
        time_of_day=time_of_day,
        weather=weather,
        location=location,
    )

    # Parse tracks
    for track_elem in root.findall("track"):
        label = track_elem.get("label", "")
        trk = Track(label=label)

        for box_elem in track_elem.findall("box"):
            outside = box_elem.get("outside", "0") == "1"
            if outside:
                continue

            # Extract attributes
            attrs = {}
            for attr_elem in box_elem.findall("attribute"):
                attrs[attr_elem.get("name", "")] = attr_elem.text or ""

            box = BoundingBox(
                frame=int(box_elem.get("frame", "0")),
                xtl=float(box_elem.get("xtl", "0")),
                ytl=float(box_elem.get("ytl", "0")),
                xbr=float(box_elem.get("xbr", "0")),
                ybr=float(box_elem.get("ybr", "0")),
                occluded=box_elem.get("occluded", "0") == "1",
                outside=False,
                track_id=attrs.get("id", ""),
                action=attrs.get("action", ""),
                cross=attrs.get("cross", ""),
                look=attrs.get("look", ""),
                occlusion=attrs.get("occlusion", "none"),
            )
            trk.boxes.append(box)

        if trk.boxes:
            annotation.tracks.append(trk)

    return annotation


def load_all_annotations() -> dict[str, VideoAnnotation]:
    """Parse all XML annotation files. Returns {video_id: VideoAnnotation}."""
    annotations = {}
    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))

    for xml_path in track(xml_files, description="Parsing annotations..."):
        ann = parse_annotation(xml_path)
        annotations[ann.video_id] = ann

    return annotations
