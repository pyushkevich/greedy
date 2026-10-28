"""Compose the Greedy logo from a real greedy registration (see make_logo.sh),
in the same bookshelf layout as the Convert3D logo: the moving image, the moving
image resampled through the warp at several points of the optimization (with the
warped grid drawn on it), and the fixed image."""
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
from PIL import Image, ImageDraw, ImageFont, ImageFilter

src, font, out = sys.argv[1], sys.argv[2], sys.argv[3]
THUMB = len(sys.argv) > 4 and sys.argv[4] == "thumb"  # small version without captions
DARK, LIGHT = (0x16, 0x56, 0x7a), (0x00, 0xa8, 0xd8)
TEXT = (0x33, 0x33, 0x33)
OUTLINE = (34, 34, 34, 255)

# box cut from the 78 x 110 x 64 volume, as in the Convert3D logo
X0, X1, Y0, Y1, Z0, Z1 = 15, 63, 31, 49, 2, 64
U = 5          # face samples per voxel
SPACING = 5    # grid spacing, voxels


def load(name):
    return nib.load(f"{src}/{name}.nii.gz").get_fdata().squeeze()


def face_points():
    """Voxel coordinates (3, rows, cols) of the front, top and side faces, matching
    the orientation of the faces in the Convert3D logo."""
    def axis(a0, a1, rev=False):
        t = a0 - 0.5 + (np.arange((a1 - a0) * U) + 0.5) / U
        return t[::-1] if rev else t
    x, y, z = axis(X0, X1), axis(Y0, Y1), axis(Z0, Z1)
    front = np.stack(np.broadcast_arrays(x[None, :], Y0, axis(Z0, Z1, True)[:, None]))
    top = np.stack(np.broadcast_arrays(x[None, :], axis(Y0, Y1, True)[:, None], Z1 - 1))
    side = np.stack(np.broadcast_arrays(X1 - 1, y[None, :], axis(Z0, Z1, True)[:, None]))
    # which coordinates lie in the plane of each face
    return [(front, (0, 2)), (top, (0, 1)), (side, (1, 2))]


def grey_lut(ref, lo=0.5, hi=99.6):
    a, b = np.percentile(ref, [lo, hi])
    return lambda v: np.stack([np.clip((v - a) / (b - a), 0, 1) * 255] * 3, -1)


def faces(img, lut, warp=None):
    """Sample the image (moving image through the warp, if given) on each face and
    draw the moving image's grid, carried along by the warp."""
    res = []
    for p, plane in face_points():
        c = p.astype(float)
        if warp is not None:
            u = np.stack([map_coordinates(warp[..., a], c, order=1, mode="nearest") for a in range(3)])
            c = c + u
        rgb = lut(map_coordinates(img, c, order=3, mode="nearest"))
        if warp is not None:
            line = np.zeros(c.shape[1:])
            for a in plane:
                d = np.abs(c[a] / SPACING - np.round(c[a] / SPACING)) * SPACING
                line = np.maximum(line, np.clip((1.9 if THUMB else 1.2) - d * U / 0.9, 0, 1))
            line = 0.9 * line[..., None]
            rgb = rgb * (1 - line) + np.array(LIGHT) * line
        res.append(np.clip(rgb, 0, 255).astype(np.uint8))
    return res


def warp_face(face, origin, u, v, size):
    """Map face columns along vector u and rows along v, starting at origin."""
    # shrink the finely sampled face to about its size on the canvas first, so that
    # thin grid lines are antialiased rather than broken up by the affine resampling
    im = Image.fromarray(face)
    im = im.resize((max(2, round(np.hypot(*u))), max(2, round(np.hypot(*v)))), Image.LANCZOS)
    w, h = im.size
    M = np.array([[u[0], v[0]], [u[1], v[1]]], float)
    Mi = np.linalg.inv(M)
    # output (X, Y) -> (s, t) in [0,1]^2 -> input pixel (s*w, t*h)
    a, b = Mi[0] * w
    d, e = Mi[1] * h
    c = -(a * origin[0] + b * origin[1])
    f = -(d * origin[0] + e * origin[1])
    layer = im.transform(size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)
    mask = Image.new("L", size, 0)
    p = np.array(origin, float)
    ImageDraw.Draw(mask).polygon([tuple(p), tuple(p + u), tuple(p + u + v), tuple(p + v)], fill=255)
    return layer, mask


DEPTH = np.array([0.42, -0.3])  # screen direction of the j (depth) axis


def box(fcs, o, scale):
    """Upright slab: coronal face toward the viewer with its top left corner at o."""
    front, top, side = fcs
    nx, ny, nz = X1 - X0, Y1 - Y0, Z1 - Z0
    a = np.array([1.0, 0]) * scale * nx
    b = DEPTH * scale * ny
    c = np.array([0, 1.0]) * scale * nz
    o = np.array(o, float)
    return [
        (top, o + b, a, -b, 0.82),
        (side, o + a, b, c, 0.68),
        (front, o, a, c, 1.0),
    ]


W, H = 1000, 500
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)
def mono_font(size):
    for path in ("/System/Library/Fonts/Menlo.ttc", "DejaVuSansMono.ttf", "consola.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


mono = mono_font(13)


def render(fcs, o, scale, lean=0.0):
    """Draw a slab, tilted clockwise by `lean` degrees about the bottom right
    corner of its front face, like a book leaning on its neighbour."""
    quads = box(fcs, o, scale)
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for face, org, u, v, shade in quads:
        im, mask = warp_face((face * shade).astype(np.uint8), org, u, v, canvas.size)
        layer.paste(im, (0, 0), mask)
    for _, org, u, v, _ in quads:
        q = [org, org + u, org + u + v, org + v, org]
        ld.line([tuple(p) for p in q], fill=OUTLINE, width=1, joint="curve")
    _, org, a, c, _ = quads[2]
    pivot = tuple(org + a + c)
    layer = layer.rotate(-lean, resample=Image.BICUBIC, center=pivot)
    alpha = layer.getchannel("A")
    shadow = Image.new("L", canvas.size, 0)
    shadow.paste(alpha, (5, 8))
    shadow = shadow.filter(ImageFilter.GaussianBlur(9)).point(lambda x: x * 0.3)
    canvas.paste(Image.new("RGB", canvas.size, (20, 40, 60)), (0, 0), shadow)
    canvas.paste(layer, (0, 0), layer)


captions = []


def caption(text, cx, y):
    # drawn at the end; the thumbnail leaves them out (they would be too small to read)
    captions.append((text, cx, y))


def arrow(x, y, length):
    """Draw an arrow at 4x resolution and downsample it, since PIL does not antialias."""
    k, pad = 4, 12
    x0, y0 = int(x) - pad, int(y) - pad
    w, h = int(length) + 2 * pad, 2 * pad
    layer = Image.new("RGBA", (w * k, h * k), LIGHT + (0,))
    d = ImageDraw.Draw(layer)
    X, Y = (x - x0) * k, (y - y0) * k
    L = length * k
    d.rectangle((X + 4 * k, Y - 2.5 * k, X + L - 14 * k, Y + 2.5 * k), fill=LIGHT + (255,))
    d.polygon([(X + L - 4 * k, Y), (X + L - 18 * k, Y - 10 * k), (X + L - 18 * k, Y + 10 * k)], fill=LIGHT + (255,))
    layer = layer.resize((w, h), Image.LANCZOS)
    canvas.paste(layer, (x0, y0), layer)


nx, ny, nz = X1 - X0, Y1 - Y0, Z1 - Z0


def extent(scale):
    # bounding box relative to the top left corner of the front face
    return 0, (nx + DEPTH[0] * ny) * scale, DEPTH[1] * ny * scale, nz * scale


moving, fixed = load("moving"), load("fixed")
mlut = grey_lut(moving)
stages = [("200x5x0", "iter 205"), ("200x15x0", "iter 215"), ("200x40x0", "iter 240"), ("200x100x50", "iter 350")]
boxes = [("moving.nii.gz", faces(moving, mlut))]
boxes += [(text, faces(moving, mlut, load(f"warp_{n}"))) for n, text in stages]
boxes += [("fixed.nii.gz", faces(fixed, grey_lut(fixed, 1, 99.95)))]

# six slabs of the same size standing on a shelf: the moving image, an arrow, the
# registration in progress (the first two leaning on their neighbours like books),
# then an arrow to the fixed image it is being matched to
s, gap, arrow_w = 2.2, 30, 50  # gap includes the side face
leans = [0, 12, 6, 0, 0, 0]
l, r, t, b = extent(s)
hf, wf = nz * s, nx * s              # front face size
lean_dx = [hf * np.sin(np.radians(q)) for q in leans]
widths = [wf + max(0, r - wf), *[wf + max(gap, d + 10) for d in lean_dx[1:-1]], r]
widths[-2] = r                       # the last stage is followed by an arrow, like the input
total = sum(widths) + 2 * arrow_w
x = (W - total) / 2
shelf = 270                          # y of the front faces' bottom edge
x_first = x
for i, (text, fcs) in enumerate(boxes):
    render(fcs, (x, shelf - hf), s, leans[i])
    caption(text, x + wf / 2, shelf + 14)
    x += widths[i]
    if i in (0, len(boxes) - 2):
        arrow(x, shelf - hf / 2, arrow_w)
        x += arrow_w
right = x
draw.rounded_rectangle((x_first - 14, shelf + 1, right + 14, shelf + 6), 3, fill=(214, 224, 231))

f = ImageFont.truetype(font, 132)
w1, w2 = draw.textlength("Greedy", font=f), draw.textlength("Reg", font=f)
tx = (W - w1 - w2) / 2
draw.text((tx, 300), "Greedy", font=f, fill=DARK)
draw.text((tx + w1, 300), "Reg", font=f, fill=LIGHT)
if THUMB:
    # drop the captions (too small to read) and the space they take, stack the boxes
    # tightly over the wordmark, then pad to 2:1 and size for website cards (330 x 165
    # px on the page, 2x for high DPI screens)
    def bbox(top, bottom):
        a = np.asarray(canvas.convert("L"))[top:bottom] < 248
        rows, cols = np.nonzero(a.any(1))[0], np.nonzero(a.any(0))[0]
        return cols[0], top + rows[0], cols[-1] + 1, top + rows[-1] + 1
    shelves, mark = canvas.crop(bbox(0, shelf + 7)), canvas.crop(bbox(shelf + 40, H))
    gap, margin = 22, 24
    cw = max(shelves.width, mark.width) + 2 * margin
    ch = shelves.height + gap + mark.height + 2 * margin
    cw, ch = max(cw, 2 * ch), max(ch, cw // 2)
    thumb = Image.new("RGB", (cw, ch), "white")
    y0 = (ch - shelves.height - gap - mark.height) // 2
    thumb.paste(shelves, ((cw - shelves.width) // 2, y0))
    thumb.paste(mark, ((cw - mark.width) // 2, y0 + shelves.height + gap))
    thumb.resize((660, 330), Image.LANCZOS).save(out, quality=92)
    sys.exit()
for text, cx, y in captions:
    w = draw.textlength(text, font=mono)
    draw.text((cx - w / 2, y), text, font=mono, fill=TEXT)
canvas.save(out, quality=92)
