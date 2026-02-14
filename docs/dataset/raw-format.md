# Raw JAAD Format

## Directory Structure

```
data/raw/
    clips/
        video_0001.mp4 ... video_0346.mp4
    annotations/
        video_0001.xml ... video_0346.xml
```

## Video Clips

- **Format:** MP4
- **Resolution:** 1920 x 1080
- **Frame rate:** ~30 FPS
- **Duration:** ~5-15 seconds per clip (~600 frames)
- **Total size:** ~2.9 GB

## XML Annotations (CVAT Format)

Each XML file corresponds to one video and contains frame-by-frame bounding box annotations.

### Structure

```xml
<annotations>
  <meta>
    <task>
      <name>video_0001</name>
      <size>600</size>                    <!-- Total frames -->
      <original_size>
        <width>1920</width>
        <height>1080</height>
      </original_size>
      <video_attributes>
        <time_of_day>daytime</time_of_day>
        <weather>cloudy</weather>
        <location>plaza</location>
      </video_attributes>
    </task>
  </meta>

  <track label="pedestrian">
    <box frame="0" xtl="465.0" ytl="730.0" xbr="533.0" ybr="848.0"
         occluded="0" outside="0">
      <attribute name="id">0_1_3b</attribute>
      <attribute name="action">standing</attribute>
      <attribute name="cross">not-crossing</attribute>
      <attribute name="look">not-looking</attribute>
      <attribute name="occlusion">none</attribute>
    </box>
    <!-- More boxes for subsequent frames... -->
  </track>
</annotations>
```

### Track Labels

| Label | Description | Has crossing intent? |
|-------|-------------|---------------------|
| `pedestrian` | Fully annotated pedestrians with behavior attributes | Yes |
| `ped` | Bystanders far from the vehicle | No |
| `people` | Groups of pedestrians | No |

Only `pedestrian` tracks have crossing intent annotations and are used for training.

### Bounding Box Attributes

| Attribute | Format | Values |
|-----------|--------|--------|
| `frame` | Integer | Frame number (0-indexed) |
| `xtl`, `ytl` | Float | Top-left corner (pixels) |
| `xbr`, `ybr` | Float | Bottom-right corner (pixels) |
| `occluded` | 0/1 | Whether the pedestrian is occluded |
| `outside` | 0/1 | Whether the pedestrian has left the frame |

### Pedestrian Attributes

| Attribute | Values | Description |
|-----------|--------|-------------|
| `id` | String | Unique pedestrian identifier |
| `action` | `standing`, `walking` | Current movement state |
| `cross` | `crossing`, `not-crossing` | **Crossing intent label** |
| `look` | `looking`, `not-looking` | Whether looking at the vehicle |
| `occlusion` | `none`, `part`, `full` | Occlusion level |
| `hand_gesture` | `greet`, `yield`, `rightofway`, `other` | Hand signal (if any) |
| `reaction` | `clear_path`, `speed_up`, `slow_down` | Reaction to vehicle |
| `nod` | `nodding` | Head nod detected |

### Video Attributes

Each video also has metadata:

| Attribute | Example Values |
|-----------|---------------|
| `time_of_day` | `daytime`, `nighttime` |
| `weather` | `cloudy`, `clear`, `rain` |
| `location` | `plaza`, `intersection`, `midblock` |
