"""Build a blinded side-by-side rating study from LOD result files.

Runs inside Blender; arguments follow ``--``::

    "$B" --background --factory-startup --python-exit-code 1 \
        --python scripts/make_user_study.py -- --lods results/<run>/lods --out study/<name> \
        [--levels 2,3] [--assets car,bunny] [--views front,three_quarter] [--resolution 900] [--seed N]

Input contract: ``<lods>/<asset_id>.blend`` holds ``REF__<asset_id>`` and
``<METHOD>__L<level>__<asset_id>`` meshes (METHOD in DECIMATE, QEM, PERCEPTUAL),
static and in world space.  Every stimulus pairs the reference (left,
"Referensi") with one result (right, "Hasil").  Both halves are rendered with
the same camera, lens and Workbench studio light in one neutral grey, without
textures or cavity/shadow effects, so participants judge geometry together with
the smooth/flat face shading stored in each mesh.  Per asset and view one
perspective camera is fitted to the vertices of every mesh of that asset, so
nothing is cropped and the framing does not change between halves or methods.

``--views`` lists view sets separated by commas; ``+`` joins views into one
set shown as stacked rows of one stimulus (``front+three_quarter``).
Presets (azimuth from the front -Y towards +X, elevation): see VIEW_PRESETS.

Outputs in ``--out``:
    stimuli/<id>.png          side-by-side images (in the participant package)
    stimuli.csv               researcher's stimulus list: blind method codes, LOD, ratio,
                              triangles and build_id; not in the participant package
    rating_form.html          offline form from study/rating_form_template.html
    participant_package.zip   the only thing to give participants
    private/method_key.csv    PRIVATE code -> method key; never give to participants
    private/study_config.json PRIVATE seed, inputs, cameras, budget check and full stimulus key
    ratings/                  empty; returned participant CSVs are collected here
    superseded/<time>/        PRIVATE key and stimulus list of a build replaced by --force

Every input file is checked before --out is touched: its objects, the labels
and ratio stored on them by scripts/run_experiment.py (``smart_lod_level``,
``smart_lod_method``, ``smart_lod_ratio``) and the triangle budgets.  At every
LOD the methods' triangle counts may differ by at most max(2, --budget-tolerance
x smallest) and no count may exceed its ``smart_lod_target`` by more than
max(2, tolerance x target); otherwise the build is refused unless
``--allow-unequal-budgets`` is given (docs/RESEARCH.md section 5 asks for
comparable budgets).  The check is recorded in study_config.json.  A failed
build removes its partial outputs.  A rebuild (--force) is refused once
ratings/ holds files; otherwise the previous key and stimulus list are moved to
superseded/, because forms of that build may already be with participants.

Build id: 16 hex digits of SHA-256 over the study id, the seed, the method key
and every stimulus field.  It is embedded in the form, exported in every rating
row, written to stimuli.csv and method_key.csv, and analyze_user_study.py
rejects rows whose build id differs, so ratings made with a superseded form are
never unblinded with a new key.  The seed in the hash keeps the id from being
inverted with the few possible keys.

Blinding: the method codes M1..Mk and the stimulus ids are drawn from a PRNG.
Without ``--seed`` the seed is 128 random bits and is written only to private/.
Anyone who knows the seed and the inputs can rebuild the key, so a fixed seed
must be kept private as well.  The form holds only stimulus ids and image paths;
method codes, LOD levels, target ratios and triangle counts are neither embedded
nor exported, and the analysis joins them from stimuli.csv by stimulus id.
Stimulus PNGs are written in a seeded random order within each asset and get
one common modification time, as do the ZIP entries, so file dates do not
reveal the method order.  The reference is always on the left; left/right
position is therefore not counterbalanced.

Limits: Workbench is a viewport renderer, not a perceptual model of production
shading; one or two fixed viewpoints do not cover every way an asset is seen;
normals, UV seams and texture distortion are deliberately excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import secrets
import shutil
import struct
import sys
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_HEADER = (  # docs/USER_STUDY.csv: nothing that identifies method, LOD or triangle count
    'study_id', 'build_id', 'anonymous_participant_id', 'stimulus_id', 'presentation_order',
    'shape_fidelity_1_5', 'silhouette_fidelity_1_5', 'important_detail_fidelity_1_5', 'overall_quality_1_5',
    'notes', 'export_sequence', 'exported_at')
STIMULUS_FIELDS = ('stimulus_id', 'asset_id', 'lod_level', 'target_ratio', 'actual_triangles',
                   'view_set_id', 'method_code', 'image')
PUBLIC_COLUMNS = STIMULUS_FIELDS + ('build_id',)  # stimuli.csv
METHODS = ('DECIMATE', 'QEM', 'PERCEPTUAL')
CONTROL = 'REFERENCE'  # reference-vs-reference control stimuli (optional)
DEFAULT_RATIOS = {1: 0.5, 2: 0.2, 3: 0.05}  # docs/RESEARCH.md section 4, step 3
# smart_lod_ratio is what scripts/run_experiment.py writes; the others are accepted aliases.
RATIO_PROPERTIES = ('smart_lod_ratio', 'target_ratio', 'smart_lod_target_ratio', 'requested_ratio')
BUDGET_TOLERANCE, BUDGET_SLACK = 0.02, 2  # relative; triangles (one edge collapse removes two)
VIEW_PRESETS = {  # name: (azimuth degrees from -Y towards +X, elevation degrees)
    'front': (0.0, 0.0), 'three_quarter': (45.0, 20.0), 'three_quarter_left': (-45.0, 20.0),
    'side': (90.0, 0.0), 'side_left': (-90.0, 0.0), 'back': (180.0, 0.0), 'top': (0.0, 90.0)}
OBJECT_PATTERN = re.compile(r'^([A-Z][A-Z0-9_]*?)__L(\d+)__(.+)$')
ID_ALPHABET = '23456789ABCDEFGHJKMNPQRSTUVWXYZ'  # no 0/O, 1/I/L
TEMPLATE_TOKEN = '/*@@STUDY_DATA@@*/null'
STUDY_ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
LENS_MM, SENSOR_MM, FRAME_MARGIN = 50.0, 36.0, 0.06
MODEL_GREY = (0.8, 0.8, 0.8)          # Workbench single colour (scene linear)
PANEL_BACKGROUND = (0.93, 0.93, 0.93)  # sRGB-encoded, behind the model
CANVAS = (1.0, 1.0, 1.0)               # sRGB-encoded, label strip and gaps
TEXT = (0.12, 0.12, 0.12)
LABELS = ('Referensi', 'Hasil')


# ----------------------------------------------------------------- pure helpers (no bpy)

def parse_view_sets(text):
    """'front,three_quarter' -> [('front',), ('three_quarter',)]; '+' joins views into one set."""
    sets = []
    for item in text.split(','):
        views = tuple(v.strip() for v in item.split('+') if v.strip())
        if not views:
            continue
        unknown = [v for v in views if v not in VIEW_PRESETS]
        if unknown:
            raise ValueError(f'unknown view(s) {unknown}; choose from {sorted(VIEW_PRESETS)}')
        if len(set(views)) != len(views):
            raise ValueError(f'view set {item!r} repeats a view')
        sets.append(views)
    if not sets:
        raise ValueError('no views given')
    if len({'+'.join(s) for s in sets}) != len(sets):
        raise ValueError('duplicate view set')
    return sets


def parse_int_list(text, name):
    try:
        values = [int(v) for v in text.split(',') if v.strip()]
    except ValueError as exc:
        raise ValueError(f'{name} must be comma-separated integers') from exc
    if not values or len(set(values)) != len(values) or min(values) < 1:
        raise ValueError(f'{name} must be distinct positive integers')
    return values


def parse_ratios(text):
    """'0.5,0.2,0.05' -> {1: 0.5, 2: 0.2, 3: 0.05} (LOD1, LOD2, ...)."""
    try:
        values = [float(v) for v in text.split(',') if v.strip()]
    except ValueError as exc:
        raise ValueError('--ratios must be comma-separated numbers') from exc
    if not values or not all(0.0 < v <= 1.0 for v in values):
        raise ValueError('--ratios must lie in (0, 1]')
    return {level: value for level, value in enumerate(values, 1)}


def format_ratio(value):
    return format(float(value), '.6g')


def assign_method_codes(methods, rng):
    """Random bijection method -> 'M1'..'Mk' drawn from ``rng`` (random.Random)."""
    methods = sorted(set(methods))
    codes = [f'M{i}' for i in range(1, len(methods) + 1)]
    rng.shuffle(codes)
    return dict(zip(methods, codes))


def make_stimulus_ids(count, rng, length=5):
    """``count`` distinct random ids 'S' + ``length`` characters of ID_ALPHABET."""
    if count > len(ID_ALPHABET) ** length // 4:
        raise ValueError('too many stimuli for the id length')
    ids, seen = [], set()
    while len(ids) < count:
        candidate = 'S' + ''.join(rng.choice(ID_ALPHABET) for _ in range(length))
        if candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)
    return ids


def view_direction(azimuth, elevation):
    """Unit vector from the target towards the camera (front is -Y, up is +Z)."""
    a, e = math.radians(azimuth), math.radians(elevation)
    return np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])


def camera_basis(direction):
    """Right, up and back unit vectors; the camera looks along -back."""
    back = np.asarray(direction, dtype=float)
    back = back / np.linalg.norm(back)
    hint = np.array([0.0, 1.0, 0.0]) if abs(back[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
    right = np.cross(hint, back)
    right /= np.linalg.norm(right)
    return right, np.cross(back, right), back


def fit_camera(points, direction, fov=2 * math.atan(SENSOR_MM / 2 / LENS_MM), margin=FRAME_MARGIN):
    """Perspective camera looking along ``-direction`` that frames ``points`` in a
    square image: the projected extent is centred with lens shift and the distance
    is the smallest (to bisection precision) that keeps every point at least
    ``margin`` (fraction of the half-width) inside the frame.

    Returns a dict; ``shift`` is in normalized device units ([-1, 1] across the
    frame), Blender's ``shift_x/shift_y`` are half of it."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('camera fitting needs finite points')
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    right, up, back = camera_basis(direction)
    rel = points - center
    x, y, z = rel @ right, rel @ up, rel @ back
    radius = float(np.sqrt(np.max(np.einsum('ij,ij->i', rel, rel)))) or 1.0
    tangent, limit = math.tan(fov / 2), 1.0 - margin

    def extents(distance):
        depth = tangent * (distance - z)
        u, v = x / depth, y / depth
        return np.array([u.min(), u.max(), v.min(), v.max()])

    def fits(distance):
        e = extents(distance)
        return max(e[1] - e[0], e[3] - e[2]) * 0.5 <= limit

    # Without shift every point needs |x|, |y| <= limit * tangent * depth; that
    # distance always fits once the shift centres the extent, so bisect below it.
    low = float(z.max()) + 1e-6 * radius
    high = max(float(np.max(np.maximum(np.abs(x), np.abs(y)) / (tangent * limit) + z)), low + 1e-3 * radius)
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (low, middle) if fits(middle) else (middle, high)
    distance = high
    e = extents(distance)
    return dict(location=center + back * distance, center=center, right=right, up=up, back=back,
                distance=distance, fov=fov, shift=np.array([(e[0] + e[1]) * 0.5, (e[2] + e[3]) * 0.5]),
                clip_start=max((distance - float(z.max())) * 0.5, 1e-4),
                clip_end=distance - float(z.min()) + radius)


def project(points, camera):
    """Normalized image coordinates in [-1, 1] (x right, y up) for ``fit_camera`` output."""
    rel = np.asarray(points, dtype=float) - camera['location']
    depth = -(rel @ camera['back']) * math.tan(camera['fov'] / 2)
    return np.column_stack((rel @ camera['right'] / depth, rel @ camera['up'] / depth)) - camera['shift']


def content_margins(alpha, threshold=0.5 / 255):
    """Empty pixels (left, right, top, bottom) around coverage in ``alpha`` (rows top first); None if empty."""
    rows, cols = np.flatnonzero(alpha.max(axis=1) > threshold), np.flatnonzero(alpha.max(axis=0) > threshold)
    if not len(rows):
        return None
    height, width = alpha.shape
    return (int(cols[0]), int(width - 1 - cols[-1]), int(rows[0]), int(height - 1 - rows[-1]))


def over(rgba, background):
    """Straight-alpha float RGBA (rows top first) over an sRGB colour -> uint8 RGB."""
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + np.asarray(background, dtype=np.float32) * (1.0 - alpha)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def compose(rows, labels, gap, canvas=CANVAS):
    """Stack ``rows`` of (left, right) uint8 panels under one label strip per column."""
    size = rows[0][0].shape[0]
    strip = labels[0].shape[0]
    height = strip + len(rows) * size + (len(rows) - 1) * gap
    out = np.empty((height, 2 * size + gap, 3), dtype=np.uint8)
    out[:] = np.rint(np.asarray(canvas) * 255).astype(np.uint8)
    out[:strip, :size], out[:strip, size + gap:] = labels
    top = strip
    for left, right in rows:
        out[top:top + size, :size], out[top:top + size, size + gap:] = left, right
        top += size + gap
    return out


def encode_png(rgb):
    """Deterministic 8-bit RGB PNG (Sub filter, sRGB chunk) from a (h, w, 3) uint8 array."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError('expected RGB')
    rows = rgb.reshape(height, width * 3).astype(np.int16)
    filtered = rows.copy()
    filtered[:, 3:] = (rows[:, 3:] - rows[:, :-3]) & 0xFF
    raw = np.empty((height, width * 3 + 1), dtype=np.uint8)
    raw[:, 0] = 1  # Sub
    raw[:, 1:] = filtered

    def chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'sRGB', b'\x00') + chunk(b'IDAT', zlib.compress(raw.tobytes(), 6)) + chunk(b'IEND', b''))


def script_json(data):
    """JSON that is safe inside an HTML <script> element."""
    text = json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True)
    for raw, escaped in (('<', '\\u003c'), ('>', '\\u003e'), ('&', '\\u0026'),
                         (' ', '\\u2028'), (' ', '\\u2029')):
        text = text.replace(raw, escaped)
    return text


def fill_template(template, data):
    """Embed ``data`` into the rating-form template; checks token and CSV header."""
    if template.count(TEMPLATE_TOKEN) != 1:
        raise ValueError(f'template must contain {TEMPLATE_TOKEN!r} exactly once')
    header = ','.join(CSV_HEADER)
    if template.count(header) != 1:
        raise ValueError('template does not contain the exact USER_STUDY.csv header')
    return template.replace(TEMPLATE_TOKEN, script_json(data))


def form_data(study_id, build_id, stimuli, generated_at):
    """Public payload of the form: stimulus ids and image paths only (no method code, LOD,
    ratio or triangle count; the analysis joins those from stimuli.csv)."""
    return dict(format='smart-lod-rating-form', version=2, study_id=study_id, build_id=build_id,
                generated_at=generated_at, stimuli=[dict(id=s['stimulus_id'], image=s['image'])
                                                    for s in sorted(stimuli, key=lambda s: s['stimulus_id'])])


def build_fingerprint(study_id, seed, codes, stimuli):
    """build_id: 16 hex digits of SHA-256 over study id, seed, method key and every stimulus field."""
    payload = dict(study_id=study_id, seed=str(seed), key=sorted(codes.items()),
                   stimuli=sorted([str(s[c]) for c in STIMULUS_FIELDS] for s in stimuli))
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode('utf-8')).hexdigest()[:16]


def check_labels(props, name, method, level):
    """ValueError when harness properties (smart_lod_method/level) contradict the object name."""
    for key, expected in (('smart_lod_method', method), ('smart_lod_level', level)):
        if key in props.keys() and str(props[key]) != str(expected):
            raise ValueError(f'{name}: property {key}={props[key]!r} contradicts the name ({expected})')


def target_ratio(props, name, level, ratios, ratios_given=False):
    """(ratio, provenance, warning) of one result object.

    The first of RATIO_PROPERTIES on ``props`` (an object or dict) wins; an invalid value is an
    error.  Without one the --ratios value of ``level`` is used.  ``warning`` is set when a
    property disagrees with a --ratios value the user gave explicitly."""
    for key in RATIO_PROPERTIES:
        if key in props.keys():
            try:
                value = float(props[key])
            except (TypeError, ValueError):
                value = math.nan
            if not 0.0 < value <= 1.0:
                raise ValueError(f'{name}: property {key}={props[key]!r} is not a ratio in (0, 1]')
            warning = None
            if ratios_given and level in ratios and not math.isclose(value, ratios[level], rel_tol=1e-9):
                warning = f'{name}: property {key}={format_ratio(value)} differs from --ratios LOD{level}=' \
                          f'{format_ratio(ratios[level])}; the property is used'
            return value, f'object property {key!r}', warning
    if level not in ratios:
        raise ValueError(f'{name}: no target ratio property and no --ratios value for LOD{level}')
    return ratios[level], '--ratios', None


def budget_problems(counts, targets, tolerance=BUDGET_TOLERANCE, slack=BUDGET_SLACK):
    """Triangle-budget comparability of one asset's results, per LOD level.

    ``counts`` {(method, level): triangles}; ``targets`` {(method, level): target or None}.  A
    level fails when its counts differ by more than max(slack, tolerance x smallest) or a count
    exceeds its target by more than max(slack, tolerance x target).
    Returns [dict(level, counts, targets, spread, problems)]; ``problems`` is empty when comparable."""
    report = []
    for level in sorted({lv for _, lv in counts}):
        found = {m: int(n) for (m, lv), n in sorted(counts.items()) if lv == level}
        goals = {m: int(t) for (m, lv), t in sorted(targets.items()) if lv == level and t is not None}
        low, high = min(found.values()), max(found.values())
        problems = []
        if high - low > max(slack, tolerance * low):
            problems.append(f'triangle counts differ by {(high - low) / max(low, 1):.1%}: '
                            + ', '.join(f'{m} {n}' for m, n in found.items()))
        over = [f'{m} {found[m]} > target {t}' for m, t in goals.items()
                if m in found and found[m] - t > max(slack, tolerance * t)]
        if over:
            problems.append('target exceeded: ' + ', '.join(over))
        report.append(dict(level=level, counts=found, targets=goals,
                           spread=round((high - low) / max(low, 1), 6), problems=problems))
    return report


def csv_text(columns, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def docs_header(path=ROOT / 'docs' / 'USER_STUDY.csv'):
    if not path.is_file():
        return None
    with open(path, newline='', encoding='utf-8-sig') as handle:
        return tuple(next(csv.reader(handle), ()))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


PARTICIPANT_NOTE = """PETUNJUK UNTUK PESERTA

1. Ekstrak (unzip) seluruh isi file ZIP ini ke satu folder. Jangan membuka
   rating_form.html langsung dari dalam ZIP, karena gambar tidak akan tampil.
2. Buka rating_form.html dengan Chrome, Edge, atau Firefox versi terbaru.
   Tidak perlu koneksi internet; tidak ada data yang dikirim otomatis.
3. Masukkan ID peserta anonim dari peneliti (misalnya P01). Jangan menulis
   nama, email, atau nomor telepon di mana pun, termasuk di kolom catatan.
4. Setelah selesai, klik "Unduh CSV" dan kirim file CSV tersebut ke peneliti.
   Nama file diakhiri nomor unduhan (misalnya rating_studi_P01_2.csv). Jika
   Anda mengunduh lebih dari sekali, kirim file dengan nomor terbesar.

Jawaban tersimpan sementara di browser ini. Jika halaman tertutup, buka lagi
file yang sama dan masukkan ID yang sama untuk melanjutkan.
"""


# ----------------------------------------------------------------- Blender side

def _bpy():
    import bpy  # noqa: PLC0415 - lazy so the helpers import outside Blender
    return bpy


def configure_scene(scene, width, height, light='STUDIO'):
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.film_transparent = True
    scene.render.dither_intensity = 0.0  # deterministic 8-bit output
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    settings = scene.render.image_settings
    settings.file_format, settings.color_mode, settings.color_depth = 'PNG', 'RGBA', '8'
    scene.display.render_aa = '16'
    shading = scene.display.shading
    shading.light = light
    shading.color_type = 'SINGLE'
    shading.single_color = MODEL_GREY if light == 'STUDIO' else TEXT
    shading.show_cavity = False
    shading.show_shadows = False
    shading.show_object_outline = False
    shading.show_backface_culling = False
    shading.show_xray = False
    shading.show_specular_highlight = True
    shading.use_world_space_lighting = False  # light follows the camera
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.world = scene.world or _bpy().data.worlds.new('Study world')


def render_rgba(scene, path):
    """Render ``scene`` to ``path``; returns straight-alpha float RGBA, top row first."""
    bpy = _bpy()
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True, scene=scene.name)
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        pixels = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
    finally:
        bpy.data.images.remove(image)
    return pixels.reshape(height, width, 4)[::-1]


def render_labels(width, height, tmp):
    """Anti-aliased label strips rendered with Blender's built-in font."""
    bpy = _bpy()
    scene = bpy.data.scenes.new('SmartLOD study labels')
    configure_scene(scene, width, height, light='FLAT')
    camera = bpy.data.objects.new('Label camera', bpy.data.cameras.new('Label camera'))
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = width / height  # frame is 1 unit tall
    camera.location = (0.0, 0.0, 5.0)
    scene.collection.objects.link(camera)
    scene.camera = camera
    strips = []
    for text in LABELS:
        curve = bpy.data.curves.new('Label', 'FONT')
        curve.body, curve.size, curve.align_x, curve.align_y = text, 0.5, 'CENTER', 'CENTER'
        holder = bpy.data.objects.new('Label', curve)
        scene.collection.objects.link(holder)
        alpha = render_rgba(scene, tmp / 'label.png')[..., 3:4]
        bpy.data.objects.remove(holder)
        bpy.data.curves.remove(curve)
        strip = np.asarray(TEXT, np.float32) * alpha + np.asarray(CANVAS, np.float32) * (1 - alpha)
        strips.append(np.clip(np.rint(strip * 255), 0, 255).astype(np.uint8))
    return strips


def find_objects(asset_id, methods, levels):
    """Reference and {(method, level): object} of the open file; errors list what is missing."""
    bpy = _bpy()
    reference = bpy.data.objects.get('REF__' + asset_id)
    if reference is None or reference.type != 'MESH':
        raise ValueError(f'REF__{asset_id} (mesh) not found')
    found = {}
    for obj in bpy.data.objects:
        match = OBJECT_PATTERN.match(obj.name)
        if match and obj.type == 'MESH' and match.group(3) == asset_id:
            found[(match.group(1), int(match.group(2)))] = obj
    missing = [f'{m}__L{lv}__{asset_id}' for m in methods for lv in levels if (m, lv) not in found]
    if missing:
        raise ValueError('missing objects: ' + ', '.join(missing))
    for (m, lv), obj in found.items():
        if m in methods and lv in levels:
            check_labels(obj, obj.name, m, lv)
    return reference, {(m, lv): found[(m, lv)] for m in methods for lv in levels}


def mesh_stats(obj, depsgraph):
    """World vertices and loop-triangle count of the evaluated mesh."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get('co', co)
        matrix = np.asarray(evaluated.matrix_world, dtype=float)
        return co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3], len(mesh.loop_triangles)
    finally:
        evaluated.to_mesh_clear()


def place_camera(camera_object, fit):
    from mathutils import Matrix  # noqa: PLC0415
    matrix = Matrix.Identity(4)
    for row in range(3):
        matrix[row][0], matrix[row][1], matrix[row][2] = fit['right'][row], fit['up'][row], fit['back'][row]
        matrix[row][3] = fit['location'][row]
    camera_object.matrix_world = matrix
    data = camera_object.data
    data.type, data.lens, data.sensor_width, data.sensor_fit = 'PERSP', LENS_MM, SENSOR_MM, 'AUTO'
    data.shift_x, data.shift_y = float(fit['shift'][0]) * 0.5, float(fit['shift'][1]) * 0.5
    data.clip_start, data.clip_end = fit['clip_start'], fit['clip_end']


def open_asset(path, asset_id, args):
    """Open ``path`` and link a copy of the reference and of every result into a fresh study
    scene (the saved file is never written back).  Returns (scene, copies, stats):
    copies {(method, level): (copy, original)}, stats {(method, level): (world vertices, triangles)}."""
    bpy = _bpy()
    bpy.ops.wm.open_mainfile(filepath=str(path))
    reference, candidates = find_objects(asset_id, [m for m in args.methods if m != CONTROL], args.levels)
    scene = bpy.data.scenes.new('SmartLOD study render')
    configure_scene(scene, args.resolution, args.resolution)
    sources = [(CONTROL, 0, reference)] + [(m, lv, obj) for (m, lv), obj in candidates.items()]
    copies = {}
    for method, level, obj in sources:
        copy = obj.copy()  # shares the mesh; the saved file is never written back
        copy.parent = None
        copy.matrix_world = obj.matrix_world.copy()
        copy.hide_render = copy.hide_viewport = False
        scene.collection.objects.link(copy)
        copies[(method, level)] = (copy, obj)
    with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
        depsgraph = bpy.context.evaluated_depsgraph_get()  # the study scene, as rendered
        stats = {key: mesh_stats(copy, depsgraph) for key, (copy, _) in copies.items()}
    return scene, copies, stats


def render_asset(path, asset_id, args, codes, ids, labels, tmp, out, rng):
    """Render every stimulus of one asset file; returns (stimulus rows, provenance)."""
    bpy = _bpy()
    scene, copies, stats = open_asset(path, asset_id, args)
    methods = [m for m in args.methods if m != CONTROL]
    reference = copies[(CONTROL, 0)][1]
    camera = bpy.data.objects.new('Study camera', bpy.data.cameras.new('Study camera'))
    scene.collection.objects.link(camera)
    scene.camera = camera
    points = np.concatenate([vertices for vertices, _ in stats.values()])
    reference_triangles = stats[(CONTROL, 0)][1]
    views = sorted({v for view_set in args.view_sets for v in view_set})
    cameras = {view: fit_camera(points, view_direction(*VIEW_PRESETS[view])) for view in views}
    panels, margins = {}, {}
    for view in views:
        place_camera(camera, cameras[view])
        for key, (copy, original) in copies.items():
            for other, _ in copies.values():
                other.hide_render = other is not copy
            rgba = render_rgba(scene, tmp / 'panel.png')
            margin = content_margins(rgba[..., 3])
            if margin is None or min(margin) < 1:
                print(f'[make_user_study] WARNING: {original.name} {view}: '
                      + ('nothing rendered' if margin is None else f'touches the frame edge {margin}'))
            margins[f'{original.name}/{view}'] = margin
            panels[(key, view)] = over(rgba, PANEL_BACKGROUND)
    gap = max(8, round(0.02 * args.resolution))
    rows, provenance = [], dict(file=str(path), sha256=sha256_file(path), reference=reference.name,
                                reference_triangles=reference_triangles, objects={}, cameras={},
                                content_margins_px_left_right_top_bottom=margins)
    for view, fit in cameras.items():
        provenance['cameras'][view] = dict(preset=dict(zip(('azimuth', 'elevation'), VIEW_PRESETS[view])),
                                           location=fit['location'].round(6).tolist(),
                                           target=fit['center'].round(6).tolist(), lens_mm=LENS_MM,
                                           sensor_mm=SENSOR_MM, shift_ndc=fit['shift'].round(6).tolist(),
                                           distance=round(fit['distance'], 6))
    plan = [(view_set, m, lv) for view_set in args.view_sets for lv in args.levels for m in methods]
    plan += [(view_set, CONTROL, 0) for view_set in args.view_sets] if args.reference_control else []
    rng.shuffle(plan)  # file creation order (and so file dates) must not follow the method order
    for view_set, method, level in plan:
        copy, original = copies[(method, level)]
        if method == CONTROL:
            ratio, ratio_source = 1.0, 'reference control'
        else:
            ratio, ratio_source, _ = target_ratio(original, original.name, level, args.ratios)
        stimulus_id = ids.pop()
        image = compose([(panels[((CONTROL, 0), v)], panels[((method, level), v)]) for v in view_set],
                        labels, gap)
        (out / 'stimuli' / (stimulus_id + '.png')).write_bytes(encode_png(image))
        rows.append(dict(stimulus_id=stimulus_id, asset_id=asset_id, lod_level=level,
                         target_ratio=format_ratio(ratio), actual_triangles=stats[(method, level)][1],
                         view_set_id='+'.join(view_set), method_code=codes[method],
                         image=f'stimuli/{stimulus_id}.png', method=method, object=original.name,
                         target_ratio_source=ratio_source, reference_triangles=reference_triangles))
        provenance['objects'][original.name] = stats[(method, level)][1]
    return rows, provenance


def build_parser():
    parser = argparse.ArgumentParser(prog='make_user_study.py', description=__doc__.split('\n')[0])
    parser.add_argument('--lods', type=Path, required=True, help='directory with <asset_id>.blend files')
    parser.add_argument('--out', type=Path, required=True, help='study output directory, e.g. study/pilot1')
    parser.add_argument('--levels', default='2,3', help='LOD levels to rate (default 2,3)')
    parser.add_argument('--assets', default='', help='comma-separated asset ids (default: all files)')
    parser.add_argument('--views', default='front,three_quarter',
                        help='view sets, comma-separated; "+" stacks views in one stimulus')
    parser.add_argument('--resolution', type=int, default=900, help='square panel size in pixels')
    parser.add_argument('--seed', type=int, help='PRNG seed for codes and ids (default: random, kept private)')
    parser.add_argument('--methods', default=','.join(METHODS), help='methods to include')
    parser.add_argument('--ratios', help='target ratio of LOD1,LOD2,... for objects without a smart_lod_ratio '
                        'property (default ' + ','.join(format_ratio(DEFAULT_RATIOS[k]) for k in sorted(DEFAULT_RATIOS))
                        + ')')
    parser.add_argument('--budget-tolerance', type=float, default=BUDGET_TOLERANCE,
                        help='largest relative triangle-count difference between methods at one LOD (default 0.02)')
    parser.add_argument('--allow-unequal-budgets', action='store_true',
                        help='build even when triangle budgets are not comparable (recorded in study_config.json)')
    parser.add_argument('--study-id', help='study id written to every rating row (default: --out folder name)')
    parser.add_argument('--reference-control', action='store_true',
                        help='add one reference-vs-reference control stimulus per asset and view set')
    parser.add_argument('--template', type=Path, default=ROOT / 'study' / 'rating_form_template.html')
    parser.add_argument('--force', action='store_true',
                        help='replace a previous build in --out (refused when --out/ratings holds files); '
                             'its key and stimulus list are moved to --out/superseded/')
    return parser


def parse(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.ratios_given = args.ratios is not None
    try:
        args.levels = parse_int_list(args.levels, '--levels')
        args.view_sets = parse_view_sets(args.views)
        args.ratios = parse_ratios(args.ratios) if args.ratios_given else dict(DEFAULT_RATIOS)
        args.methods = [m.strip().upper() for m in args.methods.split(',') if m.strip()]
    except ValueError as exc:
        parser.error(str(exc))
    if not args.methods or len(set(args.methods)) != len(args.methods) or CONTROL in args.methods:
        parser.error('--methods must be distinct and must not include REFERENCE')
    if not 64 <= args.resolution <= 4096:
        parser.error('--resolution must be between 64 and 4096')
    if not 0.0 <= args.budget_tolerance < 1.0:
        parser.error('--budget-tolerance must lie in [0, 1)')
    args.out = args.out.resolve()
    args.study_id = args.study_id or args.out.name
    if not STUDY_ID_PATTERN.match(args.study_id):
        parser.error('study id must be 1-64 characters of letters, digits, "_", "-" or "."')
    return args


def generated_paths(out):
    return [out / 'stimuli', out / 'private', out / 'stimuli.csv', out / 'rating_form.html',
            out / 'participant_package.zip', out / '.render_tmp']


def preflight(files, args):
    """Check objects, labels, ratios and triangle budgets of every file before --out is touched.
    Returns {asset_id: budget_problems report}; unequal budgets end the build unless allowed."""
    budgets, failures = {}, []
    for path in files:
        try:
            _, copies, stats = open_asset(path, path.stem, args)
            for (method, level), (_, original) in copies.items():
                if method != CONTROL:
                    warning = target_ratio(original, original.name, level, args.ratios, args.ratios_given)[2]
                    if warning:
                        print(f'[make_user_study] WARNING: {warning}')
        except ValueError as exc:
            raise SystemExit(f'{path.name}: {exc}') from exc
        results = {key: original for key, (_, original) in copies.items() if key[0] != CONTROL}
        report = budget_problems({key: stats[key][1] for key in results},
                                 {key: obj.get('smart_lod_target') for key, obj in results.items()},
                                 args.budget_tolerance)
        budgets[path.stem] = report
        failures += [f'{path.stem} LOD{entry["level"]}: {problem}' for entry in report for problem in entry['problems']]
    if failures:
        listing = '\n  '.join(failures)
        if not args.allow_unequal_budgets:
            raise SystemExit('triangle budgets are not comparable (docs/RESEARCH.md section 5):\n  ' + listing
                             + '\nRegenerate the LODs, or pass --allow-unequal-budgets (recorded in study_config.json).')
        print('[make_user_study] WARNING: unequal triangle budgets kept (--allow-unequal-budgets):\n  ' + listing)
    return budgets


def prepare_output(out, force):
    generated = generated_paths(out)
    ratings = out / 'ratings'
    if ratings.is_dir() and any(ratings.iterdir()):
        raise SystemExit(f'{ratings} already holds rating files; a rebuild would change the method key. '
                         'Use a new --out folder.')
    existing = [p for p in generated if p.exists()]
    if existing and not force:
        raise SystemExit(f'{out} already contains a study build; pass --force to replace it')
    if (out / 'private').is_dir() and any((out / 'private').iterdir()):
        # Forms of the old build may already be out; keep its key so their ratings stay analysable.
        archive = out / 'superseded' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive.mkdir(parents=True)
        for name in ('private', 'stimuli.csv'):
            if (out / name).exists():
                shutil.move(str(out / name), str(archive / name))
        print(f'[make_user_study] previous key and stimulus list moved to {archive} (PRIVATE)')
    for path in generated:
        if path.exists():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
    (out / 'stimuli').mkdir(parents=True)
    (out / 'private').mkdir()


def main(argv=None):
    _bpy()  # fail early and clearly outside Blender
    if argv is None:
        argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    args = parse(argv)
    header = docs_header()
    if header is not None and header != CSV_HEADER:
        raise SystemExit('docs/USER_STUDY.csv header differs from CSV_HEADER; update both study scripts')
    template = args.template.read_text(encoding='utf-8')
    fill_template(template, dict(check=True))  # fail before rendering
    lods = args.lods.resolve()
    files = sorted(lods.glob('*.blend'))
    wanted = [a.strip() for a in args.assets.split(',') if a.strip()]
    if wanted:
        unknown = sorted(set(wanted) - {f.stem for f in files})
        if unknown:
            raise SystemExit(f'no <asset>.blend in {lods} for: {", ".join(unknown)}')
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise SystemExit(f'no .blend files in {lods}')
    budgets = preflight(files, args)
    prepare_output(args.out, args.force)
    try:
        build(args, files, lods, template, budgets)
    except BaseException:
        for path in generated_paths(args.out):  # never leave a partial study behind
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
        if args.out.is_dir() and not any(args.out.iterdir()):
            args.out.rmdir()
        raise


def build(args, files, lods, template, budgets):
    bpy = _bpy()
    seed = args.seed if args.seed is not None else secrets.randbits(128)
    rng = random.Random(seed)
    code_methods = list(args.methods) + ([CONTROL] if args.reference_control else [])
    codes = assign_method_codes(code_methods, rng)
    per_asset = len(args.view_sets) * (len(args.levels) * len(args.methods) + int(args.reference_control))
    ids = make_stimulus_ids(per_asset * len(files), rng)[::-1]  # popped in creation order
    tmp = args.out / '.render_tmp'
    tmp.mkdir(exist_ok=True)
    stimuli, provenance = [], {}
    try:
        strip_height = max(32, round(0.07 * args.resolution))
        labels = render_labels(args.resolution, strip_height, tmp)
        for path in files:
            print(f'[make_user_study] rendering {path.name}', flush=True)
            try:
                rows, info = render_asset(path, path.stem, args, codes, ids, labels, tmp, args.out, rng)
            except ValueError as exc:
                raise SystemExit(f'{path.name}: {exc}') from exc
            stimuli += rows
            provenance[path.stem] = info
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    keys = [(s['asset_id'], s['method_code'], s['lod_level'], s['view_set_id']) for s in stimuli]
    assert len(set(keys)) == len(keys), 'stimulus key (asset, code, level, view set) must be unique'
    stimuli.sort(key=lambda s: s['stimulus_id'])
    build_id = build_fingerprint(args.study_id, seed, codes, stimuli)
    for stimulus in stimuli:
        stimulus['build_id'] = build_id
    moment = datetime.now(timezone.utc).replace(microsecond=0)
    generated_at = moment.isoformat()
    out = args.out
    (out / 'stimuli.csv').write_text(csv_text(PUBLIC_COLUMNS, stimuli), encoding='utf-8')
    (out / 'rating_form.html').write_text(
        fill_template(template, form_data(args.study_id, build_id, stimuli, generated_at)), encoding='utf-8')
    key_rows = [dict(study_id=args.study_id, build_id=build_id, method_code=code, method=method)
                for method, code in sorted(codes.items(), key=lambda item: item[1])]
    (out / 'private' / 'method_key.csv').write_text(
        '# PRIVATE - JANGAN dibagikan kepada peserta / never give to participants. '
        'Unblinding key of study ' + args.study_id + '.\n'
        + csv_text(('study_id', 'build_id', 'method_code', 'method'), key_rows), encoding='utf-8')
    config = dict(
        warning='PRIVATE: reveals the method of every stimulus; never give to participants.',
        study_id=args.study_id, build_id=build_id,
        build_id_recipe='sha256(json [study_id, seed, sorted key, sorted stimulus fields ' + ','.join(STIMULUS_FIELDS)
                        + '])[:16], see make_user_study.build_fingerprint',
        generated_at=generated_at, generated_by='scripts/make_user_study.py',
        blender_version=bpy.app.version_string, seed=seed, seed_source='--seed' if args.seed is not None else 'random',
        arguments=dict(lods=str(lods), levels=args.levels, views=[list(v) for v in args.view_sets],
                       methods=args.methods, ratios=args.ratios, ratios_given=args.ratios_given,
                       resolution=args.resolution, reference_control=args.reference_control,
                       assets=[f.stem for f in files]),
        budget_check=dict(tolerance=args.budget_tolerance, slack_triangles=BUDGET_SLACK,
                          allow_unequal_budgets=args.allow_unequal_budgets,
                          comparable=not any(e['problems'] for report in budgets.values() for e in report),
                          assets=budgets),
        file_order='stimulus PNGs written in a seeded random order per asset; PNG and ZIP entry times = generated_at',
        template=dict(path=str(args.template), sha256=hashlib.sha256(template.encode('utf-8')).hexdigest()),
        render=dict(engine='BLENDER_WORKBENCH', light='STUDIO (camera-relative)', color='SINGLE',
                    model_grey_linear=MODEL_GREY, panel_background_srgb=PANEL_BACKGROUND,
                    cavity=False, shadows=False, specular=True, aa_samples=16, view_transform='Standard',
                    lens_mm=LENS_MM, sensor_mm=SENSOR_MM, frame_margin=FRAME_MARGIN,
                    camera_fit='per asset and view: all vertices of the reference and every rendered result'),
        method_codes=codes, assets=provenance, stimuli=stimuli)
    (out / 'private' / 'study_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    for stimulus in stimuli:  # one common file time: the write order is not recoverable from dates
        os.utime(out / stimulus['image'], (moment.timestamp(), moment.timestamp()))
    with zipfile.ZipFile(out / 'participant_package.zip', 'w') as package:
        members = [('PETUNJUK.txt', PARTICIPANT_NOTE.encode('utf-8'), zipfile.ZIP_DEFLATED),
                   ('rating_form.html', (out / 'rating_form.html').read_bytes(), zipfile.ZIP_DEFLATED)]
        members += [(s['image'], (out / s['image']).read_bytes(), zipfile.ZIP_STORED) for s in stimuli]
        for name, data, compress in members:  # fixed entry times, as for the PNG files
            info = zipfile.ZipInfo(name, date_time=moment.timetuple()[:6])
            info.compress_type, info.external_attr = compress, 0o644 << 16
            package.writestr(info, data)
    (out / 'ratings').mkdir(exist_ok=True)  # returned participant CSVs go here
    print(f'[make_user_study] {len(stimuli)} stimuli from {len(files)} asset file(s), build {build_id} -> {out}')
    print('[make_user_study] give participants ONLY participant_package.zip; keep private/ private.')


if __name__ == '__main__':
    main()
