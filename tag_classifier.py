import re
from typing import Any

# Canonical 8-slot taxonomy (preserves existing 4 slots + discovered, adds outfit/scene/appearance)
SLOT_ORDER: list[str] = [
    "camera",
    "pose",
    "expression",
    "lighting",
    "outfit",
    "scene",
    "appearance",
    "discovered",
]

# Natural ordering when assembling final Stable Diffusion / Anima prompt strings
PROMPT_OUTPUT_ORDER: list[str] = [
    "camera",
    "appearance",
    "outfit",
    "pose",
    "expression",
    "scene",
    "lighting",
    "discovered",
]

# Soft per-generation caps per slot (used to avoid semantic collisions, NOT rigid quotas)
SLOT_SOFT_CAPS: dict[str, int] = {
    "camera": 1,
    "lighting": 1,
    "expression": 1,
    "pose": 2,
    "outfit": 2,
    "scene": 2,
    "appearance": 2,
    "discovered": 2,
}

SLOT_META: dict[str, dict[str, Any]] = {
    "camera": {"label": "镜头", "color": "#7dd3fc", "soft_cap": 1},
    "pose": {"label": "姿势", "color": "#fbbf24", "soft_cap": 2},
    "expression": {"label": "表情", "color": "#f472b6", "soft_cap": 1},
    "lighting": {"label": "光影", "color": "#fb923c", "soft_cap": 1},
    "outfit": {"label": "服装", "color": "#34d399", "soft_cap": 2},
    "scene": {"label": "环境", "color": "#2dd4bf", "soft_cap": 2},
    "appearance": {"label": "外观", "color": "#e879f9", "soft_cap": 2},
    "discovered": {"label": "其他", "color": "#94a3b8", "soft_cap": 2},
}

CLUSTER_DEFS: list[dict[str, str]] = [
    # camera
    {"cluster": "cam_framing", "slot": "camera", "label": "景别构图"},
    {"cluster": "cam_low", "slot": "camera", "label": "仰拍透视"},
    {"cluster": "cam_high", "slot": "camera", "label": "俯拍倾斜"},
    {"cluster": "cam_orient", "slot": "camera", "label": "方位取景"},
    {"cluster": "cam_misc", "slot": "camera", "label": "其他镜头"},
    # pose
    {"cluster": "pose_stand", "slot": "pose", "label": "站姿体态"},
    {"cluster": "pose_sit_lie", "slot": "pose", "label": "坐卧姿态"},
    {"cluster": "pose_dynamic", "slot": "pose", "label": "动态动作"},
    {"cluster": "pose_hands", "slot": "pose", "label": "手部交互"},
    {"cluster": "pose_misc", "slot": "pose", "label": "其他姿态"},
    # expression
    {"cluster": "expr_gaze", "slot": "expression", "label": "眼神交互"},
    {"cluster": "expr_mood", "slot": "expression", "label": "神态情绪"},
    {"cluster": "expr_misc", "slot": "expression", "label": "其他神态"},
    # lighting
    {"cluster": "light_dramatic", "slot": "lighting", "label": "戏剧光影"},
    {"cluster": "light_soft", "slot": "lighting", "label": "柔和氛围"},
    {"cluster": "light_ambient", "slot": "lighting", "label": "环境光线状态"},
    # outfit
    {"cluster": "outfit_clothes", "slot": "outfit", "label": "服装款式"},
    {"cluster": "outfit_swim_inner", "slot": "outfit", "label": "泳装贴身"},
    {"cluster": "outfit_accessories", "slot": "outfit", "label": "配饰鞋袜"},
    {"cluster": "outfit_state", "slot": "outfit", "label": "穿着状态材质"},
    # scene
    {"cluster": "scene_indoor", "slot": "scene", "label": "室内与陈设"},
    {"cluster": "scene_outdoor", "slot": "scene", "label": "户外与自然"},
    {"cluster": "scene_props_fx", "slot": "scene", "label": "物件与氛围"},
    # appearance
    {"cluster": "appear_hair", "slot": "appearance", "label": "发色与发型"},
    {"cluster": "appear_eyes_face", "slot": "appearance", "label": "瞳色与面部"},
    {"cluster": "appear_body_style", "slot": "appearance", "label": "体态与特征"},
    # discovered
    {"cluster": "discovered_misc", "slot": "discovered", "label": "未归类新词"},
]

DEFAULT_CLUSTER_FOR_SLOT: dict[str, str] = {
    "camera": "cam_misc",
    "pose": "pose_misc",
    "expression": "expr_mood",
    "lighting": "light_ambient",
    "outfit": "outfit_clothes",
    "scene": "scene_indoor",
    "appearance": "appear_body_style",
    "discovered": "discovered_misc",
}

# Neutral, high-utility Danbooru expansion packs users can import in one click
CURATED_EXPANSION_PACKS: dict[str, dict[str, Any]] = {
    "outfit": {
        "key": "outfit",
        "label": "服装穿搭拓展包",
        "description": "中性通用的日常穿搭、礼服与鞋袜词汇（14 词）",
        "tags": [
            {"tag": "sundress", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "white shirt", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "pleated skirt", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "cardigan", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "school uniform", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "kimono", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "sweater", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "off-shoulder dress", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "hoodie", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "blazer", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "trench coat", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "ribbed sweater", "slot": "outfit", "cluster": "outfit_clothes"},
            {"tag": "thighhighs", "slot": "outfit", "cluster": "outfit_accessories"},
            {"tag": "ribbon", "slot": "outfit", "cluster": "outfit_accessories"},
        ],
    },
    "scene": {
        "key": "scene",
        "label": "环境场景拓展包",
        "description": "弥补纯白背景偏科的室内/户外环境词（14 词）",
        "tags": [
            {"tag": "classroom", "slot": "scene", "cluster": "scene_indoor"},
            {"tag": "by window", "slot": "scene", "cluster": "scene_indoor"},
            {"tag": "cafe", "slot": "scene", "cluster": "scene_indoor"},
            {"tag": "library", "slot": "scene", "cluster": "scene_indoor"},
            {"tag": "balcony", "slot": "scene", "cluster": "scene_indoor"},
            {"tag": "rainy window", "slot": "scene", "cluster": "scene_indoor"},
            {"tag": "city street", "slot": "scene", "cluster": "scene_outdoor"},
            {"tag": "night city", "slot": "scene", "cluster": "scene_outdoor"},
            {"tag": "sunset beach", "slot": "scene", "cluster": "scene_outdoor"},
            {"tag": "cherry blossoms", "slot": "scene", "cluster": "scene_outdoor"},
            {"tag": "forest", "slot": "scene", "cluster": "scene_outdoor"},
            {"tag": "rooftop", "slot": "scene", "cluster": "scene_outdoor"},
            {"tag": "flower field", "slot": "scene", "cluster": "scene_outdoor"},
            {"tag": "seaside", "slot": "scene", "cluster": "scene_outdoor"},
        ],
    },
    "lighting_ambient": {
        "key": "lighting_ambient",
        "label": "环境光线状态包",
        "description": "时间段、环境光源与空气光效状态词（12 词）",
        "tags": [
            {"tag": "sunset", "slot": "lighting", "cluster": "light_ambient"},
            {"tag": "dusk", "slot": "lighting", "cluster": "light_ambient"},
            {"tag": "moonlight", "slot": "lighting", "cluster": "light_ambient"},
            {"tag": "neon lights", "slot": "lighting", "cluster": "light_ambient"},
            {"tag": "window light", "slot": "lighting", "cluster": "light_ambient"},
            {"tag": "sunbeams", "slot": "lighting", "cluster": "light_dramatic"},
            {"tag": "lens flare", "slot": "lighting", "cluster": "light_dramatic"},
            {"tag": "light particles", "slot": "lighting", "cluster": "light_soft"},
            {"tag": "city lights", "slot": "lighting", "cluster": "light_ambient"},
            {"tag": "warm lighting", "slot": "lighting", "cluster": "light_soft"},
            {"tag": "cool lighting", "slot": "lighting", "cluster": "light_ambient"},
            {"tag": "overcast", "slot": "lighting", "cluster": "light_ambient"},
        ],
    },
}

# Exact phrase overrides with strong Danbooru semantics (case-insensitive)
_EXACT_MAP: dict[str, tuple[str, str]] = {
    # Camera
    "pov": ("camera", "cam_orient"),
    "pov hands": ("camera", "cam_orient"),
    "pov crotch": ("camera", "cam_framing"),
    "pov doorway": ("camera", "cam_orient"),
    "portrait": ("camera", "cam_framing"),
    "profile": ("camera", "cam_orient"),
    "fisheye": ("camera", "cam_misc"),
    "letterboxed": ("camera", "cam_framing"),
    "feet out of frame": ("camera", "cam_framing"),
    "lower body": ("camera", "cam_framing"),
    "top-down bottom-up": ("camera", "cam_high"),
    "solo focus": ("camera", "cam_framing"),
    "male focus": ("camera", "cam_framing"),
    "ass focus": ("camera", "cam_framing"),
    "x-ray": ("camera", "cam_misc"),
    "motion lines": ("camera", "cam_misc"),
    # Lighting & ambient light state
    "sunset": ("lighting", "light_ambient"),
    "sunrise": ("lighting", "light_ambient"),
    "evening": ("lighting", "light_ambient"),
    "dusk": ("lighting", "light_ambient"),
    "dawn": ("lighting", "light_ambient"),
    "twilight": ("lighting", "light_ambient"),
    "night": ("lighting", "light_ambient"),
    "day": ("lighting", "light_ambient"),
    "morning": ("lighting", "light_ambient"),
    "sun": ("lighting", "light_ambient"),
    "moonlight": ("lighting", "light_ambient"),
    "overcast": ("lighting", "light_ambient"),
    "lamp": ("lighting", "light_ambient"),
    "desk lamp": ("lighting", "light_ambient"),
    "candle": ("lighting", "light_ambient"),
    "chandelier": ("lighting", "light_ambient"),
    "city lights": ("lighting", "light_ambient"),
    "neon lights": ("lighting", "light_ambient"),
    "window light": ("lighting", "light_ambient"),
    "warm lighting": ("lighting", "light_soft"),
    "cool lighting": ("lighting", "light_ambient"),
    "light particles": ("lighting", "light_soft"),
    "lens flare": ("lighting", "light_dramatic"),
    "sunbeams": ("lighting", "light_dramatic"),
    "silhouette": ("lighting", "light_dramatic"),
    # Outfit accessories & jewelry
    "jewelry": ("outfit", "outfit_accessories"),
    "earrings": ("outfit", "outfit_accessories"),
    "stud earrings": ("outfit", "outfit_accessories"),
    "hoop earrings": ("outfit", "outfit_accessories"),
    "cross earrings": ("outfit", "outfit_accessories"),
    "piercing": ("outfit", "outfit_accessories"),
    "ear piercing": ("outfit", "outfit_accessories"),
    "choker": ("outfit", "outfit_accessories"),
    "black choker": ("outfit", "outfit_accessories"),
    "necklace": ("outfit", "outfit_accessories"),
    "bracelet": ("outfit", "outfit_accessories"),
    "bead bracelet": ("outfit", "outfit_accessories"),
    "beads": ("outfit", "outfit_accessories"),
    "ring": ("outfit", "outfit_accessories"),
    "armlet": ("outfit", "outfit_accessories"),
    "wrist cuffs": ("outfit", "outfit_accessories"),
    "gloves": ("outfit", "outfit_accessories"),
    "black gloves": ("outfit", "outfit_accessories"),
    "white gloves": ("outfit", "outfit_accessories"),
    "elbow gloves": ("outfit", "outfit_accessories"),
    "fingerless gloves": ("outfit", "outfit_accessories"),
    "hair ornament": ("outfit", "outfit_accessories"),
    "hairband": ("outfit", "outfit_accessories"),
    "black hairband": ("outfit", "outfit_accessories"),
    "hairclip": ("outfit", "outfit_accessories"),
    "hair ribbon": ("outfit", "outfit_accessories"),
    "scrunchie": ("outfit", "outfit_accessories"),
    "wrist scrunchie": ("outfit", "outfit_accessories"),
    "maid headdress": ("outfit", "outfit_accessories"),
    "halo": ("outfit", "outfit_accessories"),
    "veil": ("outfit", "outfit_accessories"),
    "ribbon": ("outfit", "outfit_accessories"),
    "bow": ("outfit", "outfit_accessories"),
    "yellow bow": ("outfit", "outfit_accessories"),
    "bowtie": ("outfit", "outfit_accessories"),
    "yellow bowtie": ("outfit", "outfit_accessories"),
    "belt": ("outfit", "outfit_accessories"),
    "sash": ("outfit", "outfit_accessories"),
    "obi": ("outfit", "outfit_accessories"),
    "backpack": ("outfit", "outfit_accessories"),
    "bag": ("outfit", "outfit_accessories"),
    "randoseru": ("outfit", "outfit_accessories"),
    "gem": ("outfit", "outfit_accessories"),
    "socks": ("outfit", "outfit_accessories"),
    "white socks": ("outfit", "outfit_accessories"),
    "thighhighs": ("outfit", "outfit_accessories"),
    "white thighhighs": ("outfit", "outfit_accessories"),
    "black thighhighs": ("outfit", "outfit_accessories"),
    "pantyhose": ("outfit", "outfit_accessories"),
    "black pantyhose": ("outfit", "outfit_accessories"),
    "brown pantyhose": ("outfit", "outfit_accessories"),
    "white pantyhose": ("outfit", "outfit_accessories"),
    "thighband pantyhose": ("outfit", "outfit_accessories"),
    "panties under pantyhose": ("outfit", "outfit_accessories"),
    "lace trim": ("outfit", "outfit_state"),
    "frills": ("outfit", "outfit_state"),
    # Outfit state / fit / coverage
    "see-through": ("outfit", "outfit_state"),
    "see-through silhouette": ("outfit", "outfit_state"),
    "strapless": ("outfit", "outfit_state"),
    "sleeveless": ("outfit", "outfit_state"),
    "short sleeves": ("outfit", "outfit_state"),
    "long sleeves": ("outfit", "outfit_state"),
    "wide sleeves": ("outfit", "outfit_state"),
    "detached sleeves": ("outfit", "outfit_state"),
    "detached collar": ("outfit", "outfit_state"),
    "halterneck": ("outfit", "outfit_state"),
    "off shoulder": ("outfit", "outfit_state"),
    "skin tight": ("outfit", "outfit_state"),
    "highleg": ("outfit", "outfit_state"),
    "button gap": ("outfit", "outfit_state"),
    "zipper": ("outfit", "outfit_state"),
    "shirt tucked in": ("outfit", "outfit_state"),
    "wedgie": ("outfit", "outfit_state"),
    "pantylines": ("outfit", "outfit_state"),
    "covered navel": ("outfit", "outfit_state"),
    "covered nipples": ("outfit", "outfit_state"),
    "areola slip": ("outfit", "outfit_state"),
    "skirt lift": ("outfit", "outfit_state"),
    "dress lift": ("outfit", "outfit_state"),
    "shirt lift": ("outfit", "outfit_state"),
    "sweater lift": ("outfit", "outfit_state"),
    "clothes lift": ("outfit", "outfit_state"),
    "clothes pull": ("outfit", "outfit_state"),
    "pantyhose pull": ("outfit", "outfit_state"),
    "clothing aside": ("outfit", "outfit_state"),
    "panties aside": ("outfit", "outfit_state"),
    "leotard aside": ("outfit", "outfit_state"),
    "bikini bottom aside": ("outfit", "outfit_state"),
    "open clothes": ("outfit", "outfit_state"),
    "open cardigan": ("outfit", "outfit_state"),
    "torn clothes": ("outfit", "outfit_state"),
    "torn pantyhose": ("outfit", "outfit_state"),
    "wet clothes": ("outfit", "outfit_state"),
    "wet dress": ("outfit", "outfit_state"),
    "wet shirt": ("outfit", "outfit_state"),
    "wet panties": ("outfit", "outfit_state"),
    "topless": ("outfit", "outfit_state"),
    "bottomless": ("outfit", "outfit_state"),
    "nude": ("outfit", "outfit_state"),
    "completely nude": ("outfit", "outfit_state"),
    "no panties": ("outfit", "outfit_state"),
    "no pants": ("outfit", "outfit_state"),
    "no shoes": ("outfit", "outfit_state"),
    "underwear only": ("outfit", "outfit_state"),
    "bikini bottom only": ("outfit", "outfit_state"),
    "bikini top only": ("outfit", "outfit_state"),
    "pasties": ("outfit", "outfit_state"),
    "side slit": ("outfit", "outfit_state"),
    "pelvic curtain": ("outfit", "outfit_clothes"),
    "alternate costume": ("outfit", "outfit_clothes"),
    "playboy bunny": ("outfit", "outfit_clothes"),
    "cow print": ("outfit", "outfit_clothes"),
    "animal print": ("outfit", "outfit_clothes"),
    "striped clothes": ("outfit", "outfit_clothes"),
    "denim": ("outfit", "outfit_clothes"),
    "babydoll": ("outfit", "outfit_swim_inner"),
    "lingerie": ("outfit", "outfit_swim_inner"),
    "underwear": ("outfit", "outfit_swim_inner"),
    "impossible leotard": ("outfit", "outfit_swim_inner"),
    # Pose & hands (resolve before hair/outfit/scene collisions)
    "hair lift": ("pose", "pose_hands"),
    "holding hair": ("pose", "pose_hands"),
    "hair tucking": ("pose", "pose_hands"),
    "adjusting hair": ("pose", "pose_hands"),
    "hand in own hair": ("pose", "pose_hands"),
    "wind-blown hair": ("pose", "pose_dynamic"),
    "hand on own face": ("pose", "pose_hands"),
    "hand on own cheek": ("pose", "pose_hands"),
    "hand on own hip": ("pose", "pose_hands"),
    "hand on another's face": ("pose", "pose_hands"),
    "hand on another's cheek": ("pose", "pose_hands"),
    "hand up": ("pose", "pose_hands"),
    "arms up": ("pose", "pose_hands"),
    "arms behind back": ("pose", "pose_hands"),
    "crossed arms": ("pose", "pose_hands"),
    "arms under breasts": ("pose", "pose_hands"),
    "outstretched arms": ("pose", "pose_hands"),
    "outstretched arm": ("pose", "pose_hands"),
    "outstretched hand": ("pose", "pose_hands"),
    "index finger raised": ("pose", "pose_hands"),
    "holding": ("pose", "pose_hands"),
    "holding hands": ("pose", "pose_hands"),
    "holding legs": ("pose", "pose_hands"),
    "holding condom": ("pose", "pose_hands"),
    "sheet grab": ("pose", "pose_hands"),
    "leg grab": ("pose", "pose_hands"),
    "torso grab": ("pose", "pose_hands"),
    "hugging object": ("pose", "pose_hands"),
    "hugging own legs": ("pose", "pose_hands"),
    "pillow hug": ("pose", "pose_hands"),
    "incoming hug": ("pose", "pose_hands"),
    "bound": ("pose", "pose_hands"),
    "bound wrists": ("pose", "pose_hands"),
    "reaching": ("pose", "pose_hands"),
    "reaching towards viewer": ("pose", "pose_hands"),
    "presenting": ("pose", "pose_stand"),
    "bent over": ("pose", "pose_stand"),
    "standing on one leg": ("pose", "pose_stand"),
    "tiptoes": ("pose", "pose_stand"),
    "against wall": ("pose", "pose_stand"),
    "squatting": ("pose", "pose_sit_lie"),
    "wariza": ("pose", "pose_sit_lie"),
    "all fours": ("pose", "pose_sit_lie"),
    "lying": ("pose", "pose_sit_lie"),
    "on back": ("pose", "pose_sit_lie"),
    "on stomach": ("pose", "pose_sit_lie"),
    "on side": ("pose", "pose_sit_lie"),
    "knees up": ("pose", "pose_sit_lie"),
    "knees to chest": ("pose", "pose_sit_lie"),
    "legs up": ("pose", "pose_dynamic"),
    "leg lift": ("pose", "pose_dynamic"),
    "leg up": ("pose", "pose_dynamic"),
    "legs together": ("pose", "pose_sit_lie"),
    "spread legs": ("pose", "pose_sit_lie"),
    "split": ("pose", "pose_dynamic"),
    "standing split": ("pose", "pose_dynamic"),
    "head back": ("pose", "pose_dynamic"),
    "straddling": ("pose", "pose_sit_lie"),
    "girl on top": ("pose", "pose_misc"),
    "boy on top": ("pose", "pose_misc"),
    "missionary": ("pose", "pose_misc"),
    "doggystyle": ("pose", "pose_misc"),
    "prone bone": ("pose", "pose_misc"),
    "cowgirl position": ("pose", "pose_misc"),
    "squatting cowgirl position": ("pose", "pose_misc"),
    "reverse cowgirl position": ("pose", "pose_misc"),
    "standing sex": ("pose", "pose_misc"),
    "sex from behind": ("pose", "pose_misc"),
    "full nelson": ("pose", "pose_misc"),
    "mating press": ("pose", "pose_misc"),
    "reverse suspended congress": ("pose", "pose_misc"),
    "folded": ("pose", "pose_misc"),
    "face in pillow": ("pose", "pose_sit_lie"),
    "breast press": ("pose", "pose_misc"),
    "masturbation": ("pose", "pose_misc"),
    "strangling": ("pose", "pose_misc"),
    "kiss": ("pose", "pose_misc"),
    "french kiss": ("pose", "pose_misc"),
    "oral": ("pose", "pose_misc"),
    "spread pussy": ("pose", "pose_hands"),
    "penis on head": ("pose", "pose_misc"),
    "penis on ass": ("pose", "pose_misc"),
    "penis on pussy": ("pose", "pose_misc"),
    # Expression (resolve before generic eyes/face)
    "closed eyes": ("expression", "expr_mood"),
    "half-closed eyes": ("expression", "expr_mood"),
    "one eye closed": ("expression", "expr_mood"),
    "rolling eyes": ("expression", "expr_mood"),
    "facing away": ("expression", "expr_gaze"),
    "sideways": ("expression", "expr_gaze"),
    "looking down": ("expression", "expr_gaze"),
    "looking to the side": ("expression", "expr_gaze"),
    "looking at another": ("expression", "expr_gaze"),
    "looking through legs": ("expression", "expr_gaze"),
    "glancing sideways": ("expression", "expr_gaze"),
    "smile": ("expression", "expr_mood"),
    "grin": ("expression", "expr_mood"),
    ":d": ("expression", "expr_mood"),
    "blush": ("expression", "expr_mood"),
    "nose blush": ("expression", "expr_mood"),
    "ear blush": ("expression", "expr_mood"),
    "closed mouth": ("expression", "expr_mood"),
    "open mouth": ("expression", "expr_mood"),
    "tongue": ("expression", "expr_mood"),
    "tongue out": ("expression", "expr_mood"),
    "long tongue": ("expression", "expr_mood"),
    "teeth": ("expression", "expr_mood"),
    "upper teeth only": ("expression", "expr_mood"),
    "parted lips": ("expression", "expr_mood"),
    "tears": ("expression", "expr_mood"),
    "crying": ("expression", "expr_mood"),
    "sleeping": ("expression", "expr_mood"),
    "trembling": ("expression", "expr_mood"),
    "shushing": ("expression", "expr_mood"),
    "ahegao": ("expression", "expr_mood"),
    "torogao": ("expression", "expr_mood"),
    "fucked silly": ("expression", "expr_mood"),
    "sweatdrop": ("expression", "expr_mood"),
    "saliva": ("expression", "expr_mood"),
    "saliva trail": ("expression", "expr_mood"),
    "orgasm": ("expression", "expr_mood"),
    "female orgasm": ("expression", "expr_mood"),
    # Scene & props
    "indoors": ("scene", "scene_indoor"),
    "outdoors": ("scene", "scene_outdoor"),
    "simple background": ("scene", "scene_indoor"),
    "white background": ("scene", "scene_indoor"),
    "black background": ("scene", "scene_indoor"),
    "grey background": ("scene", "scene_indoor"),
    "blurry background": ("scene", "scene_indoor"),
    "bed": ("scene", "scene_indoor"),
    "on bed": ("scene", "scene_indoor"),
    "bed sheet": ("scene", "scene_indoor"),
    "stained sheets": ("scene", "scene_indoor"),
    "pillow": ("scene", "scene_indoor"),
    "curtains": ("scene", "scene_indoor"),
    "window": ("scene", "scene_indoor"),
    "window blinds": ("scene", "scene_indoor"),
    "stained glass": ("scene", "scene_indoor"),
    "couch": ("scene", "scene_indoor"),
    "on couch": ("scene", "scene_indoor"),
    "desk": ("scene", "scene_indoor"),
    "doorway": ("scene", "scene_indoor"),
    "ceiling": ("scene", "scene_indoor"),
    "on floor": ("scene", "scene_indoor"),
    "tiles": ("scene", "scene_indoor"),
    "tile wall": ("scene", "scene_indoor"),
    "bar (place)": ("scene", "scene_indoor"),
    "toilet": ("scene", "scene_indoor"),
    "restroom": ("scene", "scene_indoor"),
    "pool": ("scene", "scene_indoor"),
    "yellow theme": ("scene", "scene_indoor"),
    "sky": ("scene", "scene_outdoor"),
    "blue sky": ("scene", "scene_outdoor"),
    "orange sky": ("scene", "scene_outdoor"),
    "cloud": ("scene", "scene_outdoor"),
    "horizon": ("scene", "scene_outdoor"),
    "ocean": ("scene", "scene_outdoor"),
    "beach": ("scene", "scene_outdoor"),
    "water": ("scene", "scene_outdoor"),
    "wading": ("scene", "scene_outdoor"),
    "partially submerged": ("scene", "scene_outdoor"),
    "forest": ("scene", "scene_outdoor"),
    "tree": ("scene", "scene_outdoor"),
    "palm tree": ("scene", "scene_outdoor"),
    "grass": ("scene", "scene_outdoor"),
    "field": ("scene", "scene_outdoor"),
    "flower field": ("scene", "scene_outdoor"),
    "nature": ("scene", "scene_outdoor"),
    "cityscape": ("scene", "scene_outdoor"),
    "flower": ("scene", "scene_props_fx"),
    "petals": ("scene", "scene_props_fx"),
    "plant": ("scene", "scene_props_fx"),
    "potted plant": ("scene", "scene_props_fx"),
    "mushroom": ("scene", "scene_props_fx"),
    "vase": ("scene", "scene_props_fx"),
    "teapot": ("scene", "scene_props_fx"),
    "pen": ("scene", "scene_props_fx"),
    "candlestand": ("scene", "scene_props_fx"),
    "heart": ("scene", "scene_props_fx"),
    "spoken heart": ("scene", "scene_props_fx"),
    "wet": ("scene", "scene_props_fx"),
    "dripping": ("scene", "scene_props_fx"),
    # Appearance
    "long hair": ("appearance", "appear_hair"),
    "very long hair": ("appearance", "appear_hair"),
    "short hair": ("appearance", "appear_hair"),
    "medium hair": ("appearance", "appear_hair"),
    "wavy hair": ("appearance", "appear_hair"),
    "floating hair": ("appearance", "appear_hair"),
    "wet hair": ("appearance", "appear_hair"),
    "hair spread out": ("appearance", "appear_hair"),
    "hair over one eye": ("appearance", "appear_hair"),
    "hair behind ear": ("appearance", "appear_hair"),
    "hair between eyes": ("appearance", "appear_hair"),
    "blunt bangs": ("appearance", "appear_hair"),
    "parted bangs": ("appearance", "appear_hair"),
    "ponytail": ("appearance", "appear_hair"),
    "twintails": ("appearance", "appear_hair"),
    "braid": ("appearance", "appear_hair"),
    "bob cut": ("appearance", "appear_hair"),
    "ahoge": ("appearance", "appear_hair"),
    "hair bun": ("appearance", "appear_hair"),
    "double bun": ("appearance", "appear_hair"),
    "colored inner hair": ("appearance", "appear_hair"),
    "multicolored hair": ("appearance", "appear_hair"),
    "two-tone hair": ("appearance", "appear_hair"),
    "dark skin": ("appearance", "appear_body_style"),
    "dark-skinned female": ("appearance", "appear_body_style"),
    "dark-skinned male": ("appearance", "appear_body_style"),
    "shiny skin": ("appearance", "appear_body_style"),
    "barefoot": ("appearance", "appear_body_style"),
    "bare shoulders": ("appearance", "appear_body_style"),
    "bare legs": ("appearance", "appear_body_style"),
    "bare arms": ("appearance", "appear_body_style"),
    "feet": ("appearance", "appear_body_style"),
    "soles": ("appearance", "appear_body_style"),
    "toes": ("appearance", "appear_body_style"),
    "toenails": ("appearance", "appear_body_style"),
    "toenail polish": ("appearance", "appear_body_style"),
    "nail polish": ("appearance", "appear_body_style"),
    "black nails": ("appearance", "appear_body_style"),
    "red nails": ("appearance", "appear_body_style"),
    "fingernails": ("appearance", "appear_body_style"),
    "sweat": ("appearance", "appear_body_style"),
    "very sweaty": ("appearance", "appear_body_style"),
    "steaming body": ("appearance", "appear_body_style"),
    "skindentation": ("appearance", "appear_body_style"),
    "slap mark": ("appearance", "appear_body_style"),
    "tattoo": ("appearance", "appear_body_style"),
    "scales": ("appearance", "appear_body_style"),
    "veins": ("appearance", "appear_body_style"),
    "veiny arms": ("appearance", "appear_body_style"),
    "collarbone": ("appearance", "appear_body_style"),
    "navel": ("appearance", "appear_body_style"),
    "armpits": ("appearance", "appear_body_style"),
    "stomach": ("appearance", "appear_body_style"),
    "back": ("appearance", "appear_body_style"),
    "wide hips": ("appearance", "appear_body_style"),
    "legs": ("appearance", "appear_body_style"),
    "thighs": ("appearance", "appear_body_style"),
    "thick thighs": ("appearance", "appear_body_style"),
    "thigh gap": ("appearance", "appear_body_style"),
    "kneepits": ("appearance", "appear_body_style"),
    "pubic hair": ("appearance", "appear_body_style"),
    "female pubic hair": ("appearance", "appear_body_style"),
    "male pubic hair": ("appearance", "appear_body_style"),
    "black pubic hair": ("appearance", "appear_body_style"),
    "anal hair": ("appearance", "appear_body_style"),
    "pubic hair peek": ("appearance", "appear_body_style"),
    "large testicle": ("appearance", "appear_body_style"),
    "erection": ("appearance", "appear_body_style"),
    "perineum": ("appearance", "appear_body_style"),
    "forehead": ("appearance", "appear_eyes_face"),
    "mole": ("appearance", "appear_eyes_face"),
    "mole under eye": ("appearance", "appear_eyes_face"),
    "mole on ass": ("appearance", "appear_body_style"),
    "animal ears": ("appearance", "appear_eyes_face"),
    "fake animal ears": ("appearance", "appear_eyes_face"),
    "rabbit ears": ("appearance", "appear_eyes_face"),
    "pointy ears": ("appearance", "appear_eyes_face"),
    "horns": ("appearance", "appear_eyes_face"),
    "dragon horns": ("appearance", "appear_eyes_face"),
    "elf": ("appearance", "appear_body_style"),
    "dragon girl": ("appearance", "appear_body_style"),
    "demon girl": ("appearance", "appear_body_style"),
    "gyaru": ("appearance", "appear_body_style"),
    "mature female": ("appearance", "appear_body_style"),
    "office lady": ("appearance", "appear_body_style"),
    "loli": ("appearance", "appear_body_style"),
    "otoko no ko": ("appearance", "appear_body_style"),
    "futanari": ("appearance", "appear_body_style"),
    "newhalf": ("appearance", "appear_body_style"),
    "bald": ("appearance", "appear_hair"),
    "faceless": ("appearance", "appear_eyes_face"),
    "faceless male": ("appearance", "appear_body_style"),
    "body writing": ("appearance", "appear_body_style"),
    "tally": ("appearance", "appear_body_style"),
    "heart-shaped pupils": ("appearance", "appear_eyes_face"),
    "symbol-shaped pupils": ("appearance", "appear_eyes_face"),
    "slit pupils": ("appearance", "appear_eyes_face"),
}

_HAIR_RE = re.compile(
    r"\b(hair|bangs|ponytail|twintails|braid|ahoge|hair bun|double bun|bob cut)\b"
)
_EYES_RE = re.compile(r"\b(eyes|pupils|eyelashes|eyebrows|heterochromia)\b")
_LIGHT_RE = re.compile(
    r"\b(lighting|rim light|backlighting|chiaroscuro|tyndall|bokeh|depth of field|"
    r"sunlight|sunbeams|moonlight|lens flare|light particles|city lights|neon lights|"
    r"window light|shadow|shadows|sunset|sunrise|evening|dusk|dawn|twilight|overcast|lamp|candle|chandelier)\b"
)
_CAM_RE = re.compile(
    r"\b(close-up|upper body|lower body|cowboy shot|full body|wide shot|low angle|"
    r"high angle|from below|from above|from side|from behind|overhead|dutch angle|"
    r"foreshortening|worm's-eye|over-the-shoulder|pov|portrait|fisheye|letterboxed|"
    r"out of frame|focus|top-down|bottom-up)\b"
)
_OUTFIT_ACC_RE = re.compile(
    r"\b(earrings|choker|necklace|bracelet|jewelry|piercing|ring|armlet|wrist cuffs|"
    r"gloves|thighhighs|pantyhose|stockings|socks|shoes|boots|heels|sandals|slippers|"
    r"ribbon|bowtie|bow|hairband|hairclip|hair ornament|scrunchie|headdress|hat|cap|"
    r"glasses|veil|belt|sash|obi|backpack|bag|halo)\b"
)
_OUTFIT_SWIM_RE = re.compile(
    r"\b(bikini|swimsuit|leotard|panties|bra|lingerie|underwear|thong|bodysuit|babydoll)\b"
)
_OUTFIT_CLOTHES_RE = re.compile(
    r"\b(dress|shirt|t-shirt|skirt|sweater|jacket|coat|cardigan|hoodie|blazer|vest|"
    r"uniform|kimono|yukata|japanese clothes|camisole|crop top|shorts|pants|jeans|"
    r"turtleneck|shrug|gown|clothes|outfit|costume|apron|cape|scarf|suit|blouse)\b"
)
_SCENE_INDOOR_RE = re.compile(
    r"\b(indoors|room|classroom|library|cafe|bedroom|bed|pillow|curtains|window|couch|"
    r"chair|desk|doorway|wall|floor|ceiling|toilet|restroom|tiles|background)\b"
)
_SCENE_OUTDOOR_RE = re.compile(
    r"\b(outdoors|sky|cloud|clouds|ocean|sea|beach|seaside|water|forest|tree|grass|"
    r"field|cityscape|street|rooftop|balcony|blossoms|horizon|nature)\b"
)
_POSE_RE = re.compile(
    r"\b(standing|sitting|lying|kneeling|crouching|squatting|leaning|bending|bent over|"
    r"all fours|straddling|reaching|holding|grab|hug|arms|hand on|hands on|hands behind|"
    r"legs up|leg lift|split|tiptoes|bound|looking over|mating press|full nelson|"
    r"missionary|doggystyle|cowgirl|prone bone)\b"
)
_EXPR_RE = re.compile(
    r"\b(smile|smiling|grin|blush|mouth|lips|tongue|teeth|looking|glancing|gaze|"
    r"expression|crying|tears|sleeping|trembling|shushing|ahegao|torogao|fucked silly)\b"
)
_BODY_RE = re.compile(
    r"\b(breasts|cleavage|sideboob|underboob|backboob|nipples|areola|ass|butt|legs|thighs|"
    r"hips|navel|collarbone|stomach|armpits|shoulders|feet|soles|toes|nails|toenails|"
    r"skin|mole|tattoo|scales|veins|vulva|pussy|clitoris|anus|perineum|penis|erection|testicle|testicles|pubic|cameltoe|bulge|groin)\b"
)


def build_seed_lookup(
    seed_tags: list[dict[str, str]] | None = None,
) -> dict[str, tuple[str, str]]:
    """Builds lowercase tag -> (slot, cluster) exact lookup from seed and expansion packs."""
    lookup: dict[str, tuple[str, str]] = {}
    if seed_tags:
        for item in seed_tags:
            tag = str(item.get("tag", "")).strip().lower()
            slot = str(item.get("slot", "")).strip()
            cluster = str(item.get("cluster", "")).strip()
            if tag and slot in SLOT_META:
                lookup[tag] = (slot, cluster or DEFAULT_CLUSTER_FOR_SLOT[slot])
    for pack in CURATED_EXPANSION_PACKS.values():
        for item in pack.get("tags", []):
            tag = str(item.get("tag", "")).strip().lower()
            slot = str(item.get("slot", "")).strip()
            cluster = str(item.get("cluster", "")).strip()
            if tag and slot in SLOT_META and tag not in lookup:
                lookup[tag] = (slot, cluster or DEFAULT_CLUSTER_FOR_SLOT[slot])
    return lookup


def classify_tag(
    tag: str,
    seed_lookup: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """Classifies a Danbooru/WD14 tag into (slot, cluster) with strict disambiguation."""
    tl = str(tag or "").strip().lower()
    if not tl:
        return ("discovered", "discovered_misc")

    if seed_lookup and tl in seed_lookup:
        return seed_lookup[tl]

    if tl in _EXACT_MAP:
        return _EXACT_MAP[tl]

    # Disambiguation rules ordered to prevent substring traps (e.g. 'light brown hair')
    if _OUTFIT_ACC_RE.search(tl):
        return ("outfit", "outfit_accessories")
    if _CAM_RE.search(tl):
        return ("camera", "cam_misc")
    if "pubic" in tl or tl == "anal hair":
        return ("appearance", "appear_body_style")
    if _HAIR_RE.search(tl):
        return ("appearance", "appear_hair")
    if _EYES_RE.search(tl):
        if any(k in tl for k in ("closed", "rolling", "contact")):
            return ("expression", "expr_mood")
        return ("appearance", "appear_eyes_face")
    if _EXPR_RE.search(tl):
        if any(k in tl for k in ("looking", "glancing", "gaze", "away")):
            return ("expression", "expr_gaze")
        return ("expression", "expr_mood")
    if _POSE_RE.search(tl):
        if any(k in tl for k in ("hand", "arm", "grab", "hug", "holding", "reaching", "bound")):
            return ("pose", "pose_hands")
        if any(k in tl for k in ("sitting", "lying", "kneeling", "squatting", "all fours", "straddling")):
            return ("pose", "pose_sit_lie")
        if any(k in tl for k in ("standing", "leaning", "bent", "tiptoes")):
            return ("pose", "pose_stand")
        return ("pose", "pose_misc")
    if _OUTFIT_SWIM_RE.search(tl):
        if any(k in tl for k in ("aside", "lift", "pull", "wet", "only", "open", "torn")):
            return ("outfit", "outfit_state")
        return ("outfit", "outfit_swim_inner")
    if _OUTFIT_CLOTHES_RE.search(tl):
        if any(k in tl for k in ("aside", "lift", "pull", "wet", "see-through", "open", "torn", "tucked", "tight")):
            return ("outfit", "outfit_state")
        return ("outfit", "outfit_clothes")
    if _BODY_RE.search(tl):
        return ("appearance", "appear_body_style")
    if _LIGHT_RE.search(tl):
        return ("lighting", "light_ambient")
    if _SCENE_OUTDOOR_RE.search(tl):
        return ("scene", "scene_outdoor")
    if _SCENE_INDOOR_RE.search(tl):
        return ("scene", "scene_indoor")

    return ("discovered", "discovered_misc")
