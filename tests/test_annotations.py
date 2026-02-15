from pedsense.processing.annotations import (
    BoundingBox,
    Track,
    VideoAnnotation,
    parse_annotation,
)

VALID_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <task>
      <size>300</size>
      <original_size>
        <width>1920</width>
        <height>1080</height>
      </original_size>
      <video_attributes>
        <time_of_day>day</time_of_day>
        <weather>clear</weather>
        <location>urban</location>
      </video_attributes>
    </task>
  </meta>
  <track label="pedestrian">
    <box frame="10" xtl="100.0" ytl="200.0" xbr="150.0" ybr="400.0"
         occluded="0" outside="0">
      <attribute name="id">ped_01</attribute>
      <attribute name="action">walking</attribute>
      <attribute name="cross">crossing</attribute>
    </box>
    <box frame="11" xtl="102.0" ytl="198.0" xbr="152.0" ybr="398.0"
         occluded="1" outside="0">
      <attribute name="id">ped_01</attribute>
      <attribute name="action">standing</attribute>
      <attribute name="cross">not-crossing</attribute>
    </box>
  </track>
</annotations>
"""

MINIMAL_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <task>
      <size>100</size>
      <original_size>
        <width>1280</width>
        <height>720</height>
      </original_size>
    </task>
  </meta>
  <track label="pedestrian">
    <box frame="0" xtl="50.0" ytl="60.0" xbr="100.0" ybr="200.0"
         occluded="0" outside="0">
    </box>
  </track>
</annotations>
"""

OUTSIDE_ONLY_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <task>
      <size>50</size>
      <original_size>
        <width>1920</width>
        <height>1080</height>
      </original_size>
    </task>
  </meta>
  <track label="pedestrian">
    <box frame="0" xtl="10.0" ytl="20.0" xbr="30.0" ybr="40.0"
         occluded="0" outside="1">
    </box>
  </track>
</annotations>
"""


# --- Dataclasses ---


class TestDataclasses:
    def test_bounding_box_defaults(self):
        box = BoundingBox(frame=0, xtl=0, ytl=0, xbr=10, ybr=10, occluded=False, outside=False)
        assert box.track_id == ""
        assert box.action == ""
        assert box.cross == ""
        assert box.look == ""
        assert box.occlusion == "none"

    def test_track_default_boxes(self):
        track = Track(label="pedestrian")
        assert track.boxes == []

    def test_video_annotation_defaults(self):
        ann = VideoAnnotation(video_id="v1", num_frames=100, width=1920, height=1080)
        assert ann.time_of_day == ""
        assert ann.weather == ""
        assert ann.location == ""
        assert ann.tracks == []


# --- parse_annotation ---


class TestParseAnnotation:
    def test_parses_valid_xml(self, tmp_path):
        xml_path = tmp_path / "video_0001.xml"
        xml_path.write_text(VALID_XML)

        ann = parse_annotation(xml_path)

        assert ann.video_id == "video_0001"
        assert ann.num_frames == 300
        assert ann.width == 1920
        assert ann.height == 1080
        assert ann.time_of_day == "day"
        assert ann.weather == "clear"
        assert ann.location == "urban"
        assert len(ann.tracks) == 1
        assert ann.tracks[0].label == "pedestrian"
        assert len(ann.tracks[0].boxes) == 2

    def test_parses_box_attributes(self, tmp_path):
        xml_path = tmp_path / "video_0001.xml"
        xml_path.write_text(VALID_XML)

        ann = parse_annotation(xml_path)
        box = ann.tracks[0].boxes[0]

        assert box.frame == 10
        assert box.xtl == 100.0
        assert box.ytl == 200.0
        assert box.xbr == 150.0
        assert box.ybr == 400.0
        assert box.occluded is False
        assert box.outside is False
        assert box.track_id == "ped_01"
        assert box.action == "walking"
        assert box.cross == "crossing"

    def test_parses_occluded_flag(self, tmp_path):
        xml_path = tmp_path / "video_0001.xml"
        xml_path.write_text(VALID_XML)

        ann = parse_annotation(xml_path)
        second_box = ann.tracks[0].boxes[1]

        assert second_box.occluded is True

    def test_handles_missing_optional_attributes(self, tmp_path):
        xml_path = tmp_path / "video_0002.xml"
        xml_path.write_text(MINIMAL_XML)

        ann = parse_annotation(xml_path)

        assert ann.video_id == "video_0002"
        assert ann.num_frames == 100
        assert ann.width == 1280
        assert ann.height == 720
        assert ann.time_of_day == ""
        assert ann.weather == ""
        assert ann.location == ""

    def test_filters_outside_boxes(self, tmp_path):
        xml_path = tmp_path / "video_0003.xml"
        xml_path.write_text(OUTSIDE_ONLY_XML)

        ann = parse_annotation(xml_path)

        # Track had only outside boxes, so it should be skipped entirely
        assert len(ann.tracks) == 0

    def test_box_defaults_when_no_attributes(self, tmp_path):
        xml_path = tmp_path / "video_0002.xml"
        xml_path.write_text(MINIMAL_XML)

        ann = parse_annotation(xml_path)
        box = ann.tracks[0].boxes[0]

        assert box.track_id == ""
        assert box.action == ""
        assert box.cross == ""
        assert box.look == ""
        assert box.occlusion == "none"
