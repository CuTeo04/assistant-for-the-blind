# Plan: Mô Tả Không Gian Bằng Doubly Linked List

> **Mục tiêu:** Thay thế output visualize của pipeline hiện tại bằng mô tả không gian tự nhiên bằng tiếng Việt, phục vụ hỗ trợ người khiếm thị.

---

## 1. Dữ liệu đầu vào

Từ pipeline hiện có (`process_image`), mỗi vật thể cho ra các trường sau:

| Trường | Nguồn | Đơn vị |
|---|---|---|
| `label` | YOLO11n | chuỗi |
| `x1, y1, x2, y2` | YOLO11n | pixel |
| `cx, cy` | tính từ box | pixel |
| `conf` | YOLO11n | 0–1 |
| `depth_m` | Depth Anything V2 (calibrated) | mét |
| `focal_length` | tính từ bàn tay (MediaPipe) | pixel |
| `img_h, img_w` | ảnh sau resize | pixel |

---

## 2. Lọc vật hợp lệ

Chỉ giữ vật thỏa **đồng thời** các điều kiện sau:

```python
conf > 0.5
0.3 <= depth_m <= 8.0
```

Sau khi lọc, tính **priority score** và giữ tối đa **5 vật** có điểm cao nhất:

```python
BG_CLASSES = {"wall", "ceiling", "floor", "window"}

def priority_score(obj):
    bg_penalty = 0.5 if obj["label"] in BG_CLASSES else 1.0
    return bg_penalty * obj["conf"] / (obj["Z"] + 0.1)

objects = sorted(objects, key=priority_score, reverse=True)[:5]
```

> **Lý do giới hạn 5 vật:** Câu mô tả quá dài sẽ khó nghe và khó hiểu cho người dùng khiếm thị.

---

## 3. Quy mỗi vật ra tọa độ 3D thực

Từ `cx`, `cy`, `depth_m`, `focal_length`:

```python
X_real =  cx              * depth_m / focal_length   # ngang: trái(-) / phải(+)
Y_real = (img_h - cy)     * depth_m / focal_length   # dọc: dưới(-) / trên(+)  ← flip cy
Z_real =  depth_m                                    # sâu: trước(-) / sau(+)
```

> **Lưu ý quan trọng — flip trục Y:** Trong OpenCV, `cy` nhỏ = ở trên ảnh. Nếu không flip, phép tính `ΔY` sẽ cho kết quả ngược (vật ở trên bị mô tả là ở dưới). Dùng `(img_h - cy)` để đảm bảo `Y_real` lớn hơn = cao hơn trong thực tế.

Mỗi vật trở thành một điểm 3D `(X_real, Y_real, Z_real)` tính bằng **mét**.

---

## 4. Tính khoảng cách giữa hai vật

Với vật A và vật B:

```python
dX = X_B - X_A    # B lệch ngang so với A
dY = Y_B - Y_A    # B lệch dọc so với A (sau khi flip)
dZ = Z_B - Z_A    # B lệch sâu so với A

dist_ngang     = sqrt(dX**2 + dZ**2)          # khoảng cách trên mặt sàn
dist_khong_gian = sqrt(dX**2 + dY**2 + dZ**2)  # khoảng cách 3D đầy đủ
```

Dùng `dist_ngang` để mô tả trái/phải/trước/sau.
Dùng `dY` riêng để mô tả trên/dưới.

---

## 5. Tính ngưỡng phân loại (động)

Thay vì dùng hằng số cứng, tính ngưỡng theo phân phối thực tế của cảnh:

```python
depths = [obj["Z"] for obj in objects]
xs     = [obj["X"] for obj in objects]

q25, q75 = np.percentile(depths, [25, 75])

THRESH_DEEP  = max(0.40, (q75 - q25) * 0.30)   # ngưỡng phân biệt trước/sau
THRESH_HORIZ = max(0.15, np.std(xs)   * 0.25)   # ngưỡng phân biệt trái/phải
THRESH_VERT  = 0.30                              # ngưỡng trên/dưới (giữ tĩnh)
```

> **Lý do dùng ngưỡng động:** Ngưỡng tĩnh dễ fail khi cảnh rộng (phòng lớn, ΔZ lên tới 5m) hoặc hẹp (bàn làm việc, ΔZ chỉ 0.3m). Dùng IQR depth làm cơ sở giúp pipeline tự thích nghi.

---

## 6. Xác định quan hệ không gian

### 6.1 Quan hệ ngang (trái / phải)

```python
if   dX > +THRESH_HORIZ:  relation = "phải"
elif dX < -THRESH_HORIZ:  relation = "trái"
else:                      relation = "thẳng hàng"
```

### 6.2 Quan hệ sâu (trước / sau)

```python
if   dZ > +THRESH_DEEP:  relation = "sau"
elif dZ < -THRESH_DEEP:  relation = "trước"
else:                     relation = "cùng hàng"
```

### 6.3 Quan hệ dọc (trên / dưới)

Chỉ kết luận trên/dưới khi hai bounding box **chồng lấp theo trục X** (tránh nhầm vật ở xa sang trái thành "ở trên"):

```python
def x_overlap(a, b):
    return not (a["x2"] < b["x1"] or b["x2"] < a["x1"])

if x_overlap(a, b) and abs(dY) > THRESH_VERT:
    if dY > 0:
        a["above"].append(b)   # B cao hơn A
    else:
        a["below"].append(b)   # B thấp hơn A
```

---

## 7. Xây dựng Doubly Linked List

### 7.1 Cấu trúc node

```python
Node {
    label   : str          # "bàn"
    X, Y, Z : float        # tọa độ 3D thực (mét)
    cx, cy  : int          # tọa độ pixel trung tâm box
    x1,y1,x2,y2 : int     # bounding box pixel

    prev         : Node | None    # node liền bên trái
    next         : Node | None    # node liền bên phải
    dist_to_next : float          # dist_ngang đến node kế (mét)

    above  : list[Node]    # các vật nằm trên vật này
    below  : list[Node]    # các vật nằm dưới vật này
    behind : list[Node]    # các vật nằm sau vật này
}
```

### 7.2 Các bước xây dựng

**Bước 1 — Tách nhóm ngang và nhóm dọc/sâu**

Các vật có `|dZ| < THRESH_DEEP` với nhau → cùng nhóm ngang.
Khi có nhiều nhóm, chỉ lấy nhóm chứa vật gần nhất làm trục DLL chính.

**Bước 2 — Sắp xếp nhóm ngang theo `X_real` tăng dần**

```python
group_main = sorted(group_main, key=lambda o: o["X"])
# [tủ(X=0.1), bàn(X=0.8), ghế(X=1.1), giường(X=1.8)]
#  trái nhất                              phải nhất
```

**Bước 3 — Nối thành DLL**

```python
for i in range(len(group_main)):
    if i > 0:
        group_main[i].prev = group_main[i-1]
        group_main[i-1].next = group_main[i]
        group_main[i-1].dist_to_next = dist_ngang(group_main[i-1], group_main[i])
```

```
NULL ← tủ ←→ bàn ←→ ghế ←→ giường → NULL
```

**Bước 4 — Gắn quan hệ dọc và sâu**

Với mỗi node trong DLL, duyệt tất cả vật còn lại và gắn vào `above`, `below`, `behind` theo điều kiện mục 6.

### 7.3 Edge case: tất cả vật cùng Z

Khi `max(depths) - min(depths) < THRESH_DEEP`, toàn bộ vật nằm trên cùng mặt phẳng — không có "phía sau". Xử lý:

```python
all_same_plane = (max(depths) - min(depths)) < THRESH_DEEP
if all_same_plane:
    for node in dll:
        node.behind = []   # bỏ qua trục sâu, chỉ mô tả ngang
```

### 7.4 Edge case: chỉ 1 vật

```python
if len(objects) == 1:
    return f"Trước mặt là cái {objects[0]['label']}, cách {objects[0]['Z']:.1f}m. Không thấy vật nào khác xung quanh."
```

---

## 8. Chọn điểm neo

Điểm neo là vật được nhắc đến đầu tiên trong câu mô tả.

**Tiêu chí chọn (ưu tiên theo thứ tự):**

1. `Z` nhỏ nhất (gần camera nhất)
2. Nếu bằng nhau → `|cx - img_w/2|` nhỏ nhất (gần tâm ảnh nhất)

```python
def neo_score(obj):
    center_dist = abs(obj["cx"] - img_w / 2) / img_w   # chuẩn hóa 0–1
    return obj["Z"] + center_dist * 0.1                 # Z là yếu tố chính

neo = min(objects, key=neo_score)
```

---

## 9. Sinh cấu trúc JSON trung gian

Bước này hoàn toàn deterministic — không có rủi ro sai thông tin.

```python
scene_json = {
    "neo": neo.label,
    "neo_depth": round(neo.Z, 1),
    "chuoi_ngang": [],
    "tren_duoi": [],
    "phia_sau": []
}

# Duyệt sang phải từ neo
cur = neo.next
prev_label = neo.label
while cur:
    scene_json["chuoi_ngang"].append({
        "từ": cur.label,
        "hướng": "phải",
        "của": prev_label,
        "dist": round(cur.prev.dist_to_next, 1)
    })
    prev_label = cur.label
    cur = cur.next

# Duyệt sang trái từ neo
cur = neo.prev
prev_label = neo.label
while cur:
    scene_json["chuoi_ngang"].append({
        "từ": cur.label,
        "hướng": "trái",
        "của": prev_label,
        "dist": round(cur.next.dist_to_next, 1)
    })
    prev_label = cur.label
    cur = cur.prev

# Quan hệ trên/dưới và sau
for node in dll:
    for above_node in node.above:
        scene_json["tren_duoi"].append({
            "vật": above_node.label,
            "quan_hệ": "trên",
            "của": node.label
        })
    for behind_node in node.behind:
        scene_json["phia_sau"].append({
            "vật": behind_node.label,
            "sau": node.label,
            "dist": round(dist_ngang(node, behind_node), 1)
        })
```

**Ví dụ output JSON:**

```json
{
  "neo": "bàn",
  "neo_depth": 1.2,
  "chuoi_ngang": [
    {"từ": "tủ",     "hướng": "trái",  "của": "bàn", "dist": 0.5},
    {"từ": "ghế",    "hướng": "phải",  "của": "bàn", "dist": 0.3},
    {"từ": "giường", "hướng": "phải",  "của": "ghế", "dist": 0.4}
  ],
  "tren_duoi": [
    {"vật": "ly", "quan_hệ": "trên", "của": "bàn"}
  ],
  "phia_sau": [
    {"vật": "cửa sổ", "sau": "ghế", "dist": 1.0}
  ]
}
```

---

## 10. Sinh câu tiếng Việt từ JSON

### 10.1 Template rule-based

```python
def generate_description(scene_json: dict) -> str:
    parts = []

    # Câu 1: điểm neo
    parts.append(
        f"Trước mặt là cái {scene_json['neo']}, "
        f"cách {scene_json['neo_depth']:.1f}m."
    )

    # Câu 2+: quan hệ ngang
    for rel in scene_json["chuoi_ngang"]:
        parts.append(
            f"Bên {rel['hướng']} cái {rel['của']} "
            f"{rel['dist']:.1f}m là cái {rel['từ']}."
        )

    # Quan hệ trên/dưới
    for rel in scene_json["tren_duoi"]:
        parts.append(
            f"Bên trên cái {rel['của']} là cái {rel['vật']}."
        )

    # Quan hệ sau
    for rel in scene_json["phia_sau"]:
        parts.append(
            f"Phía sau cái {rel['sau']} "
            f"{rel['dist']:.1f}m là cái {rel['vật']}."
        )

    return " ".join(parts)
```

### 10.2 Ví dụ output hoàn chỉnh

```
Trước mặt là cái bàn, cách 1.2m.
Bên trái cái bàn 0.5m là cái tủ.
Bên phải cái bàn 0.3m là cái ghế.
Bên phải cái ghế 0.4m là cái giường.
Bên trên cái bàn là cái ly.
Phía sau cái ghế 1.0m là cái cửa sổ.
```

### 10.3 Minh họa DLL tương ứng

```
        [ly]
         ↑ above
tủ ←0.5m→ bàn★ ←0.3m→ ghế ←0.4m→ giường
                         ↓ behind
                      cửa sổ (1.0m)
```

---

## 11. Bảng xử lý edge cases

| Tình huống | Cách xử lý |
|---|---|
| Chỉ 1 vật | Mô tả neo + "Không thấy vật nào khác xung quanh" |
| 2 vật cùng `X_real` | Ưu tiên vật có `Z` nhỏ hơn đứng trước trong DLL |
| Tất cả vật cùng `Z` | Bỏ qua trục sâu, chỉ mô tả ngang |
| Quá nhiều vật (>5) | Giữ 5 vật theo `priority_score` (gần + conf cao + không phải background) |
| `ΔZ` lớn nhưng `ΔX` cũng lớn | Dùng `dist_ngang` để phán xét, không tách riêng trục |
| Vật chồng bounding box | Dùng `Z` để xác định vật nào trước, vật nào sau |
| `depth_m` không hợp lệ | Lọc bỏ nếu `depth < 0.3m` hoặc `> 8m` |
| Không detect được bàn tay | Fallback: dùng focal length ước tính từ EXIF hoặc giá trị mặc định |

---

## 12. Tích hợp vào `process_image`

Thay toàn bộ khối `# ---- DRAW ----` và `plt.show()` bằng:

```python
# ---- BUILD SCENE & DESCRIBE ----
start = time.perf_counter()

valid_objects = filter_objects(object_data, depth_final, focal_length, h, w)
dll_head      = build_dll(valid_objects)
scene_json    = build_scene_json(dll_head)
description   = generate_description(scene_json)
describe_time = time.perf_counter() - start

print(f"\n{'='*65}")
print("🗣️  MÔ TẢ KHÔNG GIAN:")
print(description)
print(f"\n⏱️  Sinh mô tả: {describe_time:.3f}s")
print(f"{'='*65}")
```

---

## 13. Sơ đồ pipeline tổng thể

```
YOLO11n + Depth Anything V2 + MediaPipe Hands
              ↓
    [label, box, conf, depth_m] × N vật
              ↓
    Lọc: conf>0.5, 0.3m < depth < 8m
    Giữ top-5 theo priority_score
              ↓
    Tính (X_real, Y_real★flip, Z_real) cho mỗi vật
              ↓
    Tính ngưỡng động (IQR depth, std X)
              ↓
    Phân quan hệ: ngang(ΔX) · sâu(ΔZ) · dọc(ΔY + overlap-X)
              ↓
    Sắp xếp nhóm ngang theo X_real tăng dần
              ↓
    Xây Doubly Linked List (prev ←→ next)
    Gắn above / below / behind vào từng node
              ↓
    Chọn điểm neo (Z nhỏ nhất → gần tâm ảnh nhất)
              ↓
    Duyệt DLL → sinh JSON trung gian (deterministic)
              ↓
    Template rule-based → câu tiếng Việt
              ↓
    "Trước mặt là cái bàn, cách 1.2m..."
```