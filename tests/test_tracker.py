from fall_detection.schemas import Region
from fall_detection.tracking.iou_tracker import IoUTracker, iou


def test_iou_identical():
    r = Region(0, 0, 10, 10, 100)
    assert iou(r, r) == 1.0


def test_tracker_keeps_id():
    t = IoUTracker(threshold=0.1)
    a = t.update([Region(0, 0, 20, 40, 800)])
    b = t.update([Region(1, 1, 20, 40, 800)])
    assert a[0][0] == b[0][0]


def test_id_survives_upright_to_horizontal_fall_box():
    stand, lying = Region(100, 50, 40, 120, 4000), Region(60, 135, 120, 40, 4000)
    assert iou(stand, lying) < 0.2            # IoU alone would lose the track
    t = IoUTracker()
    a = t.update([stand]); b = t.update([lying])
    assert a[0][0] == b[0][0]


def test_plain_iou_mode_loses_id_on_same_fall():
    t = IoUTracker(gate_factor=0)
    a = t.update([Region(100, 50, 40, 120, 4000)]); b = t.update([Region(60, 135, 120, 40, 4000)])
    assert a[0][0] != b[0][0]


def test_distant_people_get_separate_ids():
    t = IoUTracker()
    out = t.update([Region(0, 0, 40, 120, 4000), Region(400, 0, 40, 120, 4000)])
    assert len({tid for tid, _ in out}) == 2
    out2 = t.update([Region(2, 0, 40, 120, 4000), Region(402, 0, 40, 120, 4000)])
    assert sorted(tid for tid, _ in out2) == [1, 2]
