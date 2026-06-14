from __future__ import annotations

import re

YOLO_LABEL_VI = {
    "bed": "giường",
    "book": "cuốn sách",
    "bottle": "chai",
    "bowl": "cái chén",
    "bucket": "xô",
    "cabinet": "tủ",
    "cardboard box": "thùng carton",
    "ceiling fan": "quạt trần",
    "chair": "cái ghế",
    "charger": "bộ sạc",
    "chopsticks": "đũa",
    "clothes hanger": "móc treo quần áo",
    "clothes rack": "giá treo quần áo",
    "cooking pot": "nồi",
    "couch": "ghế sofa",
    "cup": "cái cốc",
    "curtain": "rèm cửa",
    "cutting board": "thớt",
    "desk": "cái bàn",
    "dining table": "bàn ăn",
    "door": "cửa",
    "dustpan": "hót rác",
    "electric kettle": "ấm đun nước điện",
    "electrical outlet": "ổ cắm điện",
    "extension cord": "dây nối dài",
    "fan": "quạt",
    "floor mat": "tấm lót sàn",
    "frying pan": "chảo",
    "gas stove": "bếp gas",
    "glasses": "kính mắt",
    "handrail": "tay vịn",
    "hanging clothes": "quần áo treo",
    "helmet": "mũ bảo hiểm",
    "keyboard": "bàn phím",
    "keys": "chùm chìa khóa",
    "lamp": "đèn",
    "laptop": "máy tính xách tay",
    "light switch": "công tắc đèn",
    "medicine": "thuốc",
    "mirror": "gương",
    "mop": "cây lau nhà",
    "mouse": "chuột máy tính",
    "mosquito net": "màn chống muỗi",
    "oven": "lò nướng",
    "pillow": "gối",
    "person": "người",
    "phone": "điện thoại",
    "plastic bag": "túi nhựa",
    "plastic basin": "chậu nhựa",
    "power bank": "pin sạc dự phòng",
    "power outlet": "ổ cắm điện",
    "power strip": "ổ cắm kéo dài",
    "refrigerator": "tủ lạnh",
    "remote": "điều khiển",
    "rice cooker": "nồi cơm điện",
    "sandals": "dép",
    "shelf": "kệ",
    "shoe rack": "kệ giày",
    "shopping bag": "túi mua sắm",
    "slippers": "dép lê",
    "sofa": "ghế sofa",
    "spoon": "thìa",
    "stairs": "cầu thang",
    "staircase": "cầu thang",
    "standing fan": "quạt đứng",
    "step ladder": "thang bậc",
    "storage box": "hộp lưu trữ",
    "suitcase": "vali",
    "table": "cái bàn",
    "teddy bear": "gấu bông",
    "toaster": "lò nướng bánh mì",
    "toothbrush": "bàn chải đánh răng",
    "trash bin": "thùng rác",
    "tv": "ti vi",
    "umbrella": "ô",
    "vase": "lọ hoa",
    "wallet": "ví",
    "wardrobe": "tủ quần áo",
    "washing machine": "máy giặt",
    "window": "cửa sổ",
    "wine glass": "ly rượu vang",
}


def translate_label(label: str) -> str:
    normalized = str(label or "").strip()
    if not normalized:
        return normalized
    return YOLO_LABEL_VI.get(normalized.lower(), normalized)


def translate_label_with_id(name: str) -> str:
    text = re.sub(r"^(cái|chiếc)\s+", "", (name or "").strip(), flags=re.IGNORECASE)
    match = re.match(r"(.+?)\s+(\d+)$", text)
    if match:
        label = match.group(1).strip()
        object_id = match.group(2)
    else:
        label = text
        object_id = ""

    translated = translate_label(label)
    return f"{translated} {object_id}".strip()
