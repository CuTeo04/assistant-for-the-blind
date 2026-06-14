# Client API

Tài liệu này mô tả contract giữa client test và `POST /process`.

## Endpoint

`POST /process`

Content type:

- `multipart/form-data`

## Multipart Fields

### Bắt buộc

- `audio`: file âm thanh chứa lệnh tiếng Việt.
- `image`: file ảnh camera hoặc ảnh test.

### Tùy chọn

- `focal_length_px` hoặc `f`: tiêu cự pixel client gửi kèm cho request bình thường.
- `depth_scale`: hệ số hiệu chỉnh depth client gửi kèm cho request bình thường.
- `hand_distance_cm`: khoảng cách thực từ camera đến tay khi chạy `THIET_LAP_CAU_HINH`.

## Mode `setup`

Mode này dùng khi người dùng nói:

- `thiết lập camera`
- `thiết lập cấu hình`

Luồng:

1. Client gửi `audio` + `image` có tay.
2. Client có thể gửi `hand_distance_cm`.
3. Nếu không gửi `hand_distance_cm`, server dùng mặc định `43 cm`.
4. Server chạy Task 2 calibration lane, detect tay, tính:
   - `focal_length_px`
   - `depth_scale`
5. Task 1 vẫn chạy song song để xác nhận intent.
6. Server trả về calibration cho client.
7. Client lưu calibration để dùng lại ở request sau.

Ví dụ request:

```python
files = {
    "audio": ("setup.wav", audio_f, "audio/wav"),
    "image": ("hand.jpg", image_f, "image/jpeg"),
}
data = {
    "hand_distance_cm": "43",
}
requests.post(url, files=files, data=data)
```

## Mode `normal`

Mode này dùng cho các request bình thường sau khi đã có calibration.

Client gửi:

- `audio`
- `image`
- `focal_length_px` hoặc `f`
- `depth_scale`

Nếu không gửi hai giá trị này, server sẽ dùng giá trị mặc định trong cấu hình:

- `vision.fixed_focal_length_px`
- `vision.default_depth_scale`

Ví dụ request:

```python
files = {
    "audio": ("cmd.wav", audio_f, "audio/wav"),
    "image": ("scene.jpg", image_f, "image/jpeg"),
}
data = {
    "focal_length_px": "723.137184602688",
    "depth_scale": "0.43492076031705035",
}
requests.post(url, files=files, data=data)
```

## Response

### Normal response

```json
{
  "api": "O_PHIA_TRUOC_CO_GI",
  "text": "Phía trước có một người cách bạn 44 xen-ti-mét."
}
```

### Calibration response

Khi `api = THIET_LAP_CAU_HINH`, server trả thêm `camera_calibration`:

```json
{
  "api": "THIET_LAP_CAU_HINH",
  "text": "Đã thiết lập camera. Tiêu cự sử dụng 723.1 pixel. Hệ số chuẩn hóa depth 0.435.",
  "camera_calibration": {
    "focal_length_px": 723.137184602688,
    "depth_scale": 0.43492076031705035,
    "hand_depth_raw_m": 0.9886858463287354,
    "known_distance_m": 0.43,
    "hand_center": {
      "x": 132,
      "y": 598
    },
    "pixel_hand_span": 117.72000679578642,
    "used_default_hand_distance": true
  }
}
```

## File Calibration

`test/client_test.py` lưu calibration vào `camera_calibration.json` với 3 khóa:

- `focal_length_px`
- `depth_scale`
- `known_distance_m`

Client `normal` mode sẽ đọc file này nếu tồn tại và tự đưa các giá trị vào request.

## Ví dụ Chạy

Setup:

```powershell
python test\client_test.py --mode setup --image hand.jpg
```

Normal:

```powershell
python test\client_test.py --mode normal --image test_room.jpg
```

Override tay:

```powershell
python test\client_test.py --mode normal --image test_room.jpg --focal-length-px 730 --depth-scale 1.0
```
