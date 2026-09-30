#!/usr/bin/env python3
"""
PerioBoneAI - offline inference pipeline for one periapical radiograph.

Contract
  input   : /input/image.jpg                 (read-only)
  weights : /weights/                        (read-only)
              tooth_seg.pt            YOLO11-seg, class "tooth"
              keypoints.pt            YOLO11-pose, CEJ / alveolar crest / apex per tooth
              bone_level.torchscript  UNet++ (EfficientNet-B4), crestal bone-level line, TorchScript
              pipeline_config.json    thresholds and sizes written by the training notebook
  output  : /output/result.json  ->  {"predictions": [{"class": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]}
            bbox in pixels of the original image. Nothing found -> {"predictions": []}.
  errors  : exit code != 0 and no result.json.

Only torch (CPU), ultralytics, opencv-python-headless, Pillow and numpy are used. No network access is needed.
The paths can be overridden with PIPELINE_INPUT / PIPELINE_WEIGHTS / PIPELINE_OUTPUT (for local testing only).

Output classes
  tooth                                  tooth instance (YOLO11-seg)
  cej, alveolar_crest, apex              anatomical landmarks (small box around the point)
  bone_level                             each connected segment of the predicted crestal bone line
  bone_loss_horizontal                   crest contact with horizontal bone loss
  defect_angular_narrow / _intermediate / _wide
                                         angular (intrabony) bone loss by radiographic defect angle
  prognosis_good / _fair / _poor / _questionable / _hopeless_candidate
                                         radiographic tooth prognosis (tooth box)
  crown_lengthening_3mm_acceptable / _borderline / _unfavourable
                                         would the tooth tolerate crown lengthening (tooth box)
  stage_I / stage_II / stage_III_IV      radiographic periodontitis stage of this image (whole-image box)
Research use only - not a medical device.
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())   # ultralytics settings in a writable place
os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

INPUT = os.environ.get("PIPELINE_INPUT", "/input/image.jpg")
WEIGHTS = os.environ.get("PIPELINE_WEIGHTS", "/weights")
OUTPUT = os.environ.get("PIPELINE_OUTPUT", "/output/result.json")

DEFAULT_CFG = {
    "seg_imgsz": 1024, "pose_imgsz": 1024, "bone_imgsz": 768,
    "seg_conf": 0.25, "pose_conf": 0.25, "kpt_conf": 0.5, "bone_threshold": 0.5,
    "keypoint_names": ["cej", "cej", "alveolar_crest", "alveolar_crest", "apex", "apex"],
    "landmark_box_half": 8, "min_bone_component_px": 30,
    "defect_scales": [0.08, 0.15, 0.25], "horizontal_deg": 54.1, "narrow_deg": 25, "wide_deg": 37,
    "min_depth_ratio": 0.07, "cr_min": 1.0, "cr_border": 1.25,
    "prog_fair": 25, "prog_poor": 50, "prog_hopeless": 80,
    "assumed_root_mm": 13.0, "px_per_mm": None, "cl_mm": 3,
    "outputs": ["tooth", "landmarks", "bone_level", "defects", "prognosis", "crown_lengthening", "stage"],
}


# ----------------------------------------------------------------------------- loading
def load_config():
    cfg = dict(DEFAULT_CFG)
    path = os.path.join(WEIGHTS, "pipeline_config.json")
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def require(name):
    path = os.path.join(WEIGHTS, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing weight file: {path}")
    return path


# ----------------------------------------------------------------------------- models
def run_yolo(weights, img, imgsz, conf):
    model = YOLO(weights)
    return model.predict(img, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]


@torch.inference_mode()
def bone_probability(ts_path, gray, size):
    model = torch.jit.load(ts_path, map_location="cpu").eval()
    h, w = gray.shape
    x = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    logits = model(torch.from_numpy(x)[None, None])
    prob = torch.sigmoid(logits)[0, 0].numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


# ----------------------------------------------------------------------------- geometry
def skeleton(mask):
    """Morphological skeleton with OpenCV only (no scikit-image)."""
    img = (mask > 0).astype(np.uint8)
    skel = np.zeros_like(img)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while img.any():
        eroded = cv2.erode(img, k)
        skel |= img - cv2.dilate(eroded, k)
        img = eroded
    return skel


def densify(poly, step=1.0):
    p = np.asarray(poly, float)
    q = np.roll(p, -1, 0)
    out = []
    for a, b in zip(p, q):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    return np.vstack(out)


def pca_dir(pts):
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return None
    _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    return vt[0] / (np.linalg.norm(vt[0]) + 1e-9)


def tooth_frame(poly, kp):
    """Axis from the CEJ midpoint to the apex midpoint (first pair = CEJ, last pair = apex)."""
    if kp is None or len(kp) < 4:
        return None
    kp = np.asarray(kp, float)
    cej = [p[:2] for p in kp[:2] if p[2] > 0]
    apx = [p[:2] for p in kp[-2:] if p[2] > 0]
    if len(cej) < 2 or not apx:
        return None
    cej = np.array(cej)
    mid, apex = cej.mean(0), np.mean(apx, 0)
    u = apex - mid
    length = np.linalg.norm(u)
    if length < 10:
        return None
    u /= length
    n = np.array([-u[1], u[0]])
    pts = densify(poly)
    d = (pts - mid) @ n
    return dict(mid=mid, u=u, n=n, L=length, cej=cej, poly=pts, width=max(d.max() - d.min(), 1.0))


def side_measure(fr, side, bone_pts, cfg):
    mid, u, n, L, P = fr["mid"], fr["u"], fr["n"], fr["L"], fr["poly"]
    ce = fr["cej"][np.argmax(((fr["cej"] - mid) @ n) * side)]
    tP, dP = (P - mid) @ u, (P - mid) @ n
    root = P[(dP * side > 0) & (tP > -0.05 * L) & (tP < L)]
    if len(root) < 5 or len(bone_pts) < 5:
        return None
    tb, db = (bone_pts - mid) @ u, (bone_pts - mid) @ n
    cand = bone_pts[(db * side > 0) & (tb > -0.1 * L) & (tb < L)]
    if len(cand) < 5:
        return None
    # nearest bone pixel to the root surface (chunked to bound memory)
    best_d, best_j = np.inf, -1
    for s in range(0, len(cand), 2048):
        c = cand[s:s + 2048]
        dist = np.min(np.linalg.norm(c[:, None, :] - root[None, :, :], axis=2), axis=1)
        j = int(np.argmin(dist))
        if dist[j] < best_d:
            best_d, best_j = dist[j], s + j
    if best_d > 0.35 * fr["width"]:
        return None
    C = cand[best_j]
    tC = float((C - mid) @ u)
    angles = []
    for sc in cfg["defect_scales"]:
        r = sc * L
        rs = root[np.abs(((root - mid) @ u) - tC) < r]
        bw = cand[(np.linalg.norm(cand - C, axis=1) < r) & (((cand - C) @ n) * side > 0)]
        dr = pca_dir(rs)
        dw = pca_dir(np.vstack([bw, C[None]])) if len(bw) >= 3 else None
        if dr is None or dw is None:
            continue
        if dr @ u > 0:
            dr = -dr
        if dw @ (side * n) < 0:
            dw = -dw
        angles.append(float(np.degrees(np.arccos(np.clip(dr @ dw, -1, 1)))))
    if not angles:
        return None
    near = cand[(np.linalg.norm(cand - C, axis=1) < max(cfg["defect_scales"]) * L) & (((cand - C) @ n) * side > 0)]
    t_top = float(((near - mid) @ u).min()) if len(near) else tC
    return dict(contact=C, t_contact=tC, theta=float(np.median(angles)),
                depth_ratio=max(tC - t_top, 0.0) / L, rbl_pct=100.0 * max(float((C - ce) @ u), 0.0) / L)


def defect_class(theta, depth_ratio, cfg):
    if theta > cfg["horizontal_deg"] or depth_ratio < cfg["min_depth_ratio"]:
        return "bone_loss_horizontal"
    if theta < cfg["narrow_deg"]:
        return "defect_angular_narrow"
    if theta > cfg["wide_deg"]:
        return "defect_angular_wide"
    return "defect_angular_intermediate"


def prognosis(rbl, cr, cfg):
    if rbl >= cfg["prog_hopeless"]:
        return "hopeless_candidate"
    if rbl >= cfg["prog_poor"]:
        return "questionable" if cr > cfg["cr_min"] else "poor"
    if rbl >= cfg["prog_fair"] or cr > cfg["cr_min"]:
        return "fair"
    return "good"


def crown_lengthening(fr, t_crest, rbl, cfg):
    mid, u, L = fr["mid"], fr["u"], fr["L"]
    t_tip = float(((fr["poly"] - mid) @ u).min())
    ppm = cfg["px_per_mm"] or L / cfg["assumed_root_mm"]
    d = cfg["cl_mm"] * ppm
    cr_after = (t_crest + d - t_tip) / max(L - t_crest - d, 1e-6)
    rbl_after = rbl + 100.0 * d / L
    if cr_after <= cfg["cr_min"] and rbl_after < cfg["prog_poor"]:
        return "acceptable"
    if cr_after <= cfg["cr_border"] and rbl_after < 60:
        return "borderline"
    return "unfavourable"


# ----------------------------------------------------------------------------- helpers
def clip_box(b, w, h):
    x1, y1, x2, y2 = [float(v) for v in b]
    return [max(0.0, min(x1, w - 1.0)), max(0.0, min(y1, h - 1.0)),
            max(0.0, min(x2, w - 1.0)), max(0.0, min(y2, h - 1.0))]


def pred(cls, conf, box, w, h):
    return {"class": cls, "confidence": round(float(min(max(conf, 0.0), 1.0)), 4),
            "bbox": [round(v, 1) for v in clip_box(box, w, h)]}


def write_result(predictions):
    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    tmp = OUTPUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"predictions": predictions}, f)
    os.replace(tmp, OUTPUT)            # the file appears only when complete


# ----------------------------------------------------------------------------- main
def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    cfg = load_config()
    want = set(cfg["outputs"])
    img = load_image(INPUT)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = []

    seg = run_yolo(require("tooth_seg.pt"), img, cfg["seg_imgsz"], cfg["seg_conf"])
    pose = run_yolo(require("keypoints.pt"), img, cfg["pose_imgsz"], cfg["pose_conf"])
    prob = bone_probability(require("bone_level.torchscript"), gray, cfg["bone_imgsz"])
    bone = (prob > cfg["bone_threshold"]).astype(np.uint8)

    # teeth
    teeth = []
    if seg.boxes is not None and len(seg.boxes):
        boxes = seg.boxes.xyxy.cpu().numpy()
        confs = seg.boxes.conf.cpu().numpy()
        polys = seg.masks.xy if seg.masks is not None else [None] * len(boxes)
        for b, c, p in zip(boxes, confs, polys):
            teeth.append(dict(box=b, conf=float(c), poly=p))
            if "tooth" in want:
                out.append(pred("tooth", c, b, w, h))

    # landmarks
    names = cfg["keypoint_names"]
    kp_sets = []
    if pose.keypoints is not None and pose.boxes is not None and len(pose.boxes):
        xy = pose.keypoints.xy.cpu().numpy()
        kc = pose.keypoints.conf.cpu().numpy() if pose.keypoints.conf is not None else np.ones(xy.shape[:2])
        centers = pose.boxes.xywh.cpu().numpy()[:, :2]
        r = cfg["landmark_box_half"]
        for pts, cs, ctr in zip(xy, kc, centers):
            kp_sets.append((ctr, [[x, y, 2 if c > cfg["kpt_conf"] else 0] for (x, y), c in zip(pts, cs)]))
            if "landmarks" in want:
                for k, ((x, y), c) in enumerate(zip(pts, cs)):
                    if c > cfg["kpt_conf"] and k < len(names):
                        out.append(pred(names[k], c, [x - r, y - r, x + r, y + r], w, h))

    # bone-level segments
    if "bone_level" in want and bone.any():
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(bone, connectivity=8)
        for i in range(1, n_lab):
            x, y, bw, bh, area = stats[i]
            if area >= cfg["min_bone_component_px"]:
                out.append(pred("bone_level", prob[labels == i].mean(), [x, y, x + bw, y + bh], w, h))

    # geometry per tooth: defects, prognosis, crown lengthening, stage
    ys, xs = np.nonzero(skeleton(bone))
    bone_pts = np.c_[xs, ys].astype(float)
    worst = None
    for t in teeth:
        if t["poly"] is None or len(t["poly"]) < 3 or not kp_sets:
            continue
        c = np.asarray(t["poly"], float).mean(0)
        kp = min(kp_sets, key=lambda s: np.linalg.norm(s[0] - c))[1]
        fr = tooth_frame(t["poly"], kp)
        if fr is None:
            continue
        sides = [m for m in (side_measure(fr, s, bone_pts, cfg) for s in (-1, 1)) if m is not None]
        if not sides:
            continue
        for m in sides:
            if "defects" in want:
                half = max(10.0, 0.08 * fr["L"])
                cx, cy = m["contact"]
                x1, y1, x2, y2 = int(max(cx - 3, 0)), int(max(cy - 3, 0)), int(cx + 4), int(cy + 4)
                local = prob[y1:y2, x1:x2]
                conf = float(local.max()) if local.size else t["conf"]
                out.append(pred(defect_class(m["theta"], m["depth_ratio"], cfg), conf,
                                [cx - half, cy - half, cx + half, cy + half], w, h))
        rbl = max(m["rbl_pct"] for m in sides)
        t_crest = float(np.mean([m["t_contact"] for m in sides]))
        t_tip = float(((fr["poly"] - fr["mid"]) @ fr["u"]).min())
        cr = (t_crest - t_tip) / max(fr["L"] - t_crest, 1e-6)
        if "prognosis" in want:
            out.append(pred("prognosis_" + prognosis(rbl, cr, cfg), t["conf"], t["box"], w, h))
        if "crown_lengthening" in want:
            out.append(pred(f"crown_lengthening_{cfg['cl_mm']}mm_" + crown_lengthening(fr, t_crest, rbl, cfg),
                            t["conf"], t["box"], w, h))
        vertical3 = False
        ppm = cfg["px_per_mm"] or fr["L"] / cfg["assumed_root_mm"]
        for m in sides:
            if defect_class(m["theta"], m["depth_ratio"], cfg).startswith("defect_angular") \
                    and m["depth_ratio"] * fr["L"] / ppm >= 3:
                vertical3 = True
        if worst is None or rbl > worst[0]:
            worst = (rbl, t["conf"], vertical3)
        elif vertical3:
            worst = (worst[0], worst[1], True)

    if "stage" in want and worst is not None:
        rbl, conf, v3 = worst
        stage = "I" if rbl < 15 else ("II" if rbl <= 33 else "III_IV")
        if v3:
            stage = "III_IV"
        out.append(pred("stage_" + stage, conf, [0, 0, w - 1, h - 1], w, h))

    write_result(out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                      # report and fail; result.json is not written
        print(f"pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.path.exists(OUTPUT + ".tmp"):
            os.remove(OUTPUT + ".tmp")
        sys.exit(1)
