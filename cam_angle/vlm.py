from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import httpx
import numpy as np

from .config import Config
from .prompts_compact import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from .prompts_compact import USER_PROMPT as DEFAULT_USER_PROMPT


CANONICAL_ELEMENTS: Dict[str, Dict[str, Any]] = {
    "oil_fryer": {"label": "油锅", "weight": 0.24},
    "cash_register": {"label": "收银机", "weight": 0.24},
    "range_hood": {"label": "油烟机", "weight": 0.18},
    "counter": {"label": "柜台/操作台", "weight": 0.14},
    "refrigerator": {"label": "冰箱/冷柜", "weight": 0.18},
    "display_case": {"label": "展示柜/保温柜/保酥柜", "weight": 0.12},
    "sink": {"label": "水池/清洗槽", "weight": 0.18},
    "prep_area": {"label": "切配/备餐/腌制区", "weight": 0.12},
    "breading_area": {"label": "裹粉区/裹粉台", "weight": 0.12},
    "employee": {"label": "员工", "weight": 0.10},
}

EXPECTED_ELEMENTS: Dict[str, List[str]] = {
    "kitchen": [
        "oil_fryer", "range_hood", "counter", "refrigerator",
        "display_case", "sink", "prep_area", "breading_area",
    ],
    "cashier": ["cash_register", "counter", "display_case", "employee"],
    "kitchen_cashier": [
        "oil_fryer", "cash_register", "range_hood", "counter",
        "refrigerator", "display_case", "sink", "prep_area", "breading_area",
    ],
    "dining": ["counter", "employee"],
    "storage": ["counter"],
    "other_indoor": ["counter", "employee"],
}

_LEGACY_SYSTEM_PROMPT = """你是门店监控摄像头画面可用性审核专家。你要判断的不是“店里现在是否有人”，而是“该镜头是否持续提供足够清楚、足够完整的业务区域证据”。

只使用一条通用判定链：
1. 业务锚点是否存在：目标业务区域本身是否在画面中，而不是只有墙面、吊顶、装饰、告示、设备局部或与业务无关的室内结构。
2. 可用覆盖是否充足：目标区域的可观察部分占画面多少，关键工作区或通道是否相对连续。主体不必几何居中；当前无人不影响锚点判断。
3. 前景侵入是否隔离了镜头与目标：任何颜色、材质、类别的物体，只要从画面边缘进入、明显离镜头更近、连续切断场景，并遮住核心区域，都必须计入遮挡。对象身份只用于写 evidence，不用于决定是否计入。
4. 全局可读性是否足够：设备边缘、工作区边界、人员肢体或通道是否能辨认；雾化、强光、过曝、脏污、低对比度或全局虚焦都按实际可读性判断。
5. 所有字段是否与文字证据一致。

通用量化口径：
- 业务主体无像素居中的要求。目标区域位于中央、下部或侧中部均可。
- subject_coverage：large 表示可用业务区占画面主要部分；medium 表示占比适中但核心设备、工作区或通道可观察；small 表示仅剩不足以持续监控的一小部分；none 表示业务区不可见。
- 合格覆盖只需要一个完整、可用的业务锚点组合：该组合约占画面三分之一，或连续经过画面中部/下部并足以观察主要活动。不要要求整家门店或整个目标区域全部入镜。
- 墙体、道路、门头、装饰和背景结构可以与目标区域共存；它们占据部分画面时只降低覆盖比例，不应仅因此把目标区域判为不存在或 wrong_area。
- 先确定目标业务区域，再估计前景侵入。遮挡比例是“中间层物体覆盖了多少目标业务区域”，不是占了多少整幅画面，也不是对象外接框面积。
- 中间层物体遮挡目标业务区约 20% 以上；或把目标区域切断到只剩局部；或使关键设备、工作区或通道无法完整观察时，属于 major obstruction，必须判 view_valid=false。
- 场景本身的边界、墙体、柜台侧面、道路边缘等如果不位于镜头与目标之间、没有遮住目标业务区，即使面积大也不计入前景遮挡。
- 窄长、细薄、横向或斜向的中间层结构，如果其横向厚度很小且没有连续盖住关键设备或工作区，只记局部干扰，不单独构成 major obstruction。
- 细小线缆、细杆、透明边缘等如果只造成局部视觉干扰，且目标仍完整可读，不算 major obstruction。
- 正常场景组成物只有在不隔离镜头与目标时才是正常内容；如果它实际上挡住了主业务区，仍按遮挡处理，不因类别常见而忽略。

结论矩阵：
- 合格：有明确业务锚点，可用覆盖为 medium/large，无 major obstruction，无影响主体识别的全局质量退化，朝向正确。
- 不合格：没有业务锚点；可用覆盖仅 small/none；存在 major obstruction；全局模糊/雾化/强光使主体难以辨认；明确对天、对墙或拍错区域。
- 复核：只有在完成上述逐项检查后，证据仍然真正互相冲突且无法判断时才使用。当前无人、动作不适用、对象类别陌生、轻微倾斜都不是 unknown 的理由。

一致性不变量：
- view_valid=true 必须同时满足 target_area_visible=true、subject_coverage=medium/large、view_issue=proper，并且没有影响核心区域的 major obstruction 或全局可读性问题。
- 文字 evidence 是事实来源。若 evidence 表明“没有业务锚点、只剩局部、被近景物体切断、主体难以辨认”，字段必须按不合格输出，不能写成合格。
- 某个设备局部在角落可见，不等于业务区域可用。
- 只输出一个合法 JSON 对象，不要 Markdown，不要推理过程。"""

_LEGACY_USER_PROMPT = """审核这张门店监控抓图。先识别固定业务区域，再判断画面覆盖是否足以持续监控，不要审核当前客流或工作繁忙程度。

camera_type 只能选择：
- store_outdoor：店外/门头/街道/停车区域
- store_entrance：门店门口、进出通道或人流经过区域
- kitchen：后厨
- cashier：收银/服务台/打包台
- kitchen_cashier：后厨和收银同框
- dining：客休/用餐区
- storage：仓库/储物区
- other_indoor：其他可确认的门店室内业务区域
- unknown：只有完全无法确认门店业务场景时才使用

第一步：确定主业务区域
- 后厨以烹饪设备、操作台、备餐/裹粉区、水池、冷柜为锚点。
- 收银以收银机、柜台、打包台或交易区为锚点。
- 用餐区以主要桌椅、顾客活动区和用餐通道为锚点。
- 仓库以货架、货物、储物区和通道为锚点。
- 门口/店外以门口、进出通道、店内外交界、门头、人行道或停车区为锚点。
- 必须先使用固定设施锚点。当前没有员工或顾客时，仍可依据固定设施确认区域和角度。
- 门头、墙体、街道或无关背景可以同时出现在画面。若门口、通道、相邻等候/用餐区或目标外部区域中至少一组完整可见，仍算业务锚点存在。

第二步：判断覆盖
- view_valid=true 表示主业务区域被充分覆盖，可以持续监控关键设备、工作区或通道。业务区域可以在画面中央、下方、侧中部或延伸到边缘，不要求几何居中。
- primary_subject_central=true 表示主业务区域属于画面中明显、有效的监控内容，不表示恰好位于正中央。只有目标被挤到最外侧窄边、画面绝大部分是无关区域时才填 false。
- subject_coverage 只能是 none、small、medium、large。正常可监控画面通常为 medium 或 large；只有目标仅剩一小角、无法承担监控用途时才填 small。
- view_issue 只能是 proper、wrong_area、too_close、wall_ceiling、obstructed、blurred、unrecognized、unknown。
- 一致规则：有效覆盖时填 view_valid=true、target_area_visible=true、primary_subject_central=true、subject_coverage=medium 或 large、view_issue=proper，并写清看见的固定业务锚点。
- 先检查是否存在从画面边界进入、明显比目标更靠近镜头的中间层区域。对象类别、颜色和材质不参与是否计入的判断，只看它是否连续切断场景并影响核心业务区。
- 窄长、细薄、斜向结构如果只跨过画面但没有连续覆盖关键设备、工作区或通道，记局部遮挡，不单独判 view_valid=false。
- 一个完整业务锚点组合约占画面三分之一，或连续贯穿中部/下部并可监控主要活动时，subject_coverage 至少为 medium，不应因旁边有墙、道路或门头而判不存在。
- 细小且透明的视觉干扰如果不影响目标完整性，可按有效覆盖处理；大面积前景侵入使目标只剩局部时，必须判无效。
- view_valid=true 必须同时满足：有明确业务锚点、业务工作区相对完整可观察、没有被中间层区域大面积切断、没有全局可读性问题。看见某台设备的一角不满足该条件。
- 如果 view_evidence 准备描述“无业务锚点”“无法确认业务区域”或“业务区只剩局部”，则 target_area_visible=false、view_valid=false、subject_coverage=none，camera_type 通常为 unknown。

第三步：判断遮挡
occlusion 只能是 none、partial、severe、unknown。
- none：没有遮挡核心业务区的前景物。
- partial：有前景物覆盖少量区域，但关键设备、主要工作区或通道仍可观察。
- severe：遮挡大面积覆盖关键设备、主要操作区或进出通道，导致无法监控。
- unknown：遮挡物属性或覆盖范围确实无法判断；不能因为正常家具、设备、立柱、线缆而填 unknown。
occlusion_area_ratio 为 0 到 1 的估计值，表示从镜头到目标之间的中间层区域覆盖了多少目标业务区域。场景自身的边界结构如果不遮挡目标，不计入该比例。
occlusion_scope 只能是 none、peripheral、critical、unknown。未影响目标完整性的细小干扰为 peripheral；遮挡目标业务区约 20% 以上，或使核心区只剩局部可见时为 critical。
occlusion_affects_subject 在核心设备、人员手部/操作区、主要业务区域或进出通道实际被近景物体挡住时为 true。

第四步：判断模糊和人员操作
blur 只能是 none、slight、severe、unknown。
- none：关键设备、人物肢体、手部或通道边缘清楚。
- slight：只有远处文字、细小纹理、水印或轻微虚焦；具体设备、人员动作和通道仍可辨认。
- severe：设备边缘、人体肢体、手部或业务动作明显拖影、雾化、过曝，无法可靠判断。
- 大面积亮雾、强光光斑、镜头脏污、整体泛白或低对比度导致设备边缘、工作区或通道难以辨认时，按 severe + global 处理，不得因为还能看见颜色或轮廓就降为 slight。
- 无人画面应根据设备和通道清晰度判断，不得因无人填 unknown。
blur_scope 只能是 none、localized、global、unknown。只有背景、边缘或非关键区域发虚时为 localized。
operation_visibility 只能是 clear、partial、obscured、not_applicable、unknown。
- 无员工时填 not_applicable，不要填 clear、obscured 或 unknown。
- 有员工且身体、手臂、手部和具体工作动作可辨认时填 clear。
- 仅有少量非关键细节不清楚时填 partial。
- 只能看到人物轮廓、无法可靠判断手部位置或动作时填 obscured。
hand_action_visible 表示当前画面中是否至少有一名员工的主要手部和具体工作动作清楚可见。
面部遮挡、背对镜头、光线照到头部、人物在画面侧边，只要手臂、手部和操作动作仍可辨认，就不能判 obscured。
如果没有任务相关的操作环境，或者画面里没有员工，不应为了“人员操作清楚”而强行输出 clear。

其他字段：
target_area_visible 表示主业务区域是否可见，不表示当前是否有人。
monitoring_subject_affected 表示关键设备、主要工作区或进出通道是否实际难以观察；正常有效画面填 false。
outdoor_view 只能是 proper、facing_sky、facing_wall、unknown。
angle_quality 只能是 good、slightly_off、bad、unknown。轻微倾斜不影响合格。

visible_elements 只从以下 id 选择，并且只在有明确视觉证据时返回：
oil_fryer（油锅/炸锅）、cash_register（收银机/收银屏）、range_hood（油烟机/排烟罩）、counter（柜台/操作台/备餐台/打包台/前台木质台面）、refrigerator（冰箱/冷柜/冷藏柜/冷冻柜）、display_case（展示柜/保温柜/保酥柜/酥脆柜）、sink（水池/清洗槽）、prep_area（切配台/备餐区/腌制备餐区）、breading_area（裹粉区/裹粉台）、employee（员工）。
visible_elements 只返回 visible=true 的元素，不可见元素直接省略。每个元素给出 0 到 1 的 confidence 和不超过 40 字的 evidence。元素必须真实可见，不能根据餐饮常识补全。

输出必须是一个 JSON 对象，并完整包含系统响应结构要求的全部字段。不要复制任何固定的正向模板；所有布尔值、枚举、面积比例和 evidence 必须由当前图片重新判断。business_scene_summary 必须与 view_valid、target_area_visible、occlusion、blur 和 monitoring_subject_affected 一致。"""

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "camera_type": {
            "type": "string",
            "enum": [
                "store_outdoor", "store_entrance", "kitchen", "cashier", "kitchen_cashier",
                "dining", "storage", "other_indoor", "unknown",
            ],
        },
        "is_store_outdoor": {"type": "boolean"},
        "is_entrance_flow": {"type": "boolean"},
        "target_area_visible": {"type": "boolean"},
        "view_valid": {"type": "boolean"},
        "view_issue": {
            "type": "string",
            "enum": [
                "proper",
                "wrong_area",
                "too_close",
                "wall_ceiling",
                "obstructed",
                "blurred",
                "unrecognized",
                "unknown",
            ],
        },
        "primary_subject_central": {"type": "boolean"},
        "subject_coverage": {
            "type": "string",
            "enum": ["none", "small", "medium", "large"],
        },
        "view_evidence": {"type": "string", "maxLength": 120},
        "occlusion": {
            "type": "string",
            "enum": ["none", "partial", "severe", "unknown"],
        },
        "occlusion_evidence": {"type": "string", "maxLength": 120},
        "occlusion_area_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "occlusion_affects_subject": {"type": "boolean"},
        "occlusion_scope": {
            "type": "string",
            "enum": ["none", "peripheral", "critical", "unknown"],
        },
        "blur": {
            "type": "string",
            "enum": ["none", "slight", "severe", "unknown"],
        },
        "blur_evidence": {"type": "string", "maxLength": 120},
        "blur_scope": {
            "type": "string",
            "enum": ["none", "localized", "global", "unknown"],
        },
        "operation_visibility": {
            "type": "string",
            "enum": ["clear", "partial", "obscured", "not_applicable", "unknown"],
        },
        "hand_action_visible": {"type": "boolean"},
        "monitoring_subject_affected": {"type": "boolean"},
        "outdoor_view": {
            "type": "string",
            "enum": ["proper", "facing_sky", "facing_wall", "unknown"],
        },
        "angle_quality": {
            "type": "string",
            "enum": ["good", "slightly_off", "bad", "unknown"],
        },
        "angle_evidence": {"type": "string", "maxLength": 120},
        "visible_elements": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {
                        "type": "string",
                        "enum": list(CANONICAL_ELEMENTS),
                    },
                    "visible": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string", "maxLength": 60},
                },
                "required": ["id", "visible", "confidence", "evidence"],
            },
        },
        "business_scene_summary": {"type": "string", "maxLength": 180},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "camera_type", "is_store_outdoor", "is_entrance_flow", "target_area_visible",
        "view_valid", "view_issue", "primary_subject_central", "subject_coverage",
        "view_evidence", "occlusion", "occlusion_evidence", "occlusion_area_ratio",
        "occlusion_affects_subject", "occlusion_scope", "blur", "blur_evidence", "blur_scope",
        "operation_visibility", "hand_action_visible",
        "monitoring_subject_affected",
        "outdoor_view", "angle_quality", "angle_evidence",
        "visible_elements", "business_scene_summary", "confidence",
    ],
}

OPERATION_VERIFY_PROMPT = """只判断这张监控画面中的员工操作是否清楚，不需要判断场景、面部或身份。
重点关注员工的手臂、手部、身体姿态，以及收银、打包、炸制、取物等工作动作。
强光、反光、面部遮挡、背景遮挡本身不影响判断；只有它们覆盖手部、手臂或工作操作区域时才算影响。
如果至少一名员工的主要手部和具体工作动作能够辨认，action_clear=true。
如果看不到员工，或员工手部和操作动作无法辨认，action_clear=false。
只输出JSON对象。"""

OPERATION_VERIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action_clear": {"type": "boolean"},
        "human_present": {"type": "boolean"},
        "hands_arms_visible": {"type": "boolean"},
        "work_area_visible": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    },
    "required": [
        "action_clear", "human_present", "hands_arms_visible",
        "work_area_visible", "confidence", "evidence",
    ],
}

OBSTRUCTION_VERIFY_PROMPT = """只做镜头空间可用性复核。
第一步确定主业务工作区，并估计目标区外接框占整幅画面的比例 target_area_ratio。第二步识别位于镜头与工作区之间、视觉上明显更近的中间层区域，并按并集估计它占整幅画面的比例 foreground_area_ratio，以及覆盖目标工作区的比例 target_overlap_ratio。比例字段只作证据，不能单独决定 scene_usable。
场景自身的边界、墙体、背景以及与目标同平面的柜台、墙面固定菜单屏/广告屏、背板、金属、玻璃和设备表面不属于中间层。墙面或柜台后方的固定屏不能仅因面积大而判作遮挡。窄长线状或透明边缘如果只跨过画面而没有连续遮住核心工作区，不单独构成 major obstruction。
只有中间层实际遮住核心设备、人员手部动作或完整工位，导致业务活动无法继续监控时，才判 scene_usable=false。面积较大但核心活动仍可辨认时，scene_usable=true。
专用摄像头可以只监控一个工位，不要求目标位于中央或占画面三分之一。只要完整工位清楚可监控，occupies_primary_view=true、narrow_side_view=false，即使目标位于左侧或右侧。
墙面、卷帘门、柜体、设备外壳和固定场景边界不属于中间层。目标位于左侧、右侧、侧中部或占比不足三分之一，都不能单独构成 narrow_side_view 或 scene_usable=false；只有独立物体实际挡住完整工位或核心操作时才可失败。
但固定结构只有在业务工位仍可正常监控时才算正常场景。墙面、招牌板、柜体侧面或近距离表面占据画面大部分、业务区只剩边缘小角时，表示镜头朝向错误或视线被切断，应判 scene_usable=false。
天花板、吊顶、灯具和空调出风口不是业务锚点。它们占据画面上半部主体、业务区只在底部露出一部分时，应判 scene_usable=false。
镜头自身外壳、防护罩、支架、机器盖板或与业务无关的邻近设备外壳不属于正常场景边界；它们占据较大画幅并把业务区挤到一侧时，应判 scene_usable=false。
雨棚、遮阳伞、帆布、篷布等表面即使属于正常设施，只要紧贴镜头并占据画面主体，只在边缘露出少量目标区域，也应判 scene_usable=false。
密集网格、笼架、护栏或筛网形成大面积连续前景屏障时，即使能部分透视，只要把业务区挤到一侧或明显削弱核心区域可读性，也应判 scene_usable=false。单根细线或少量稀疏线条才可作为局部干扰。
如果近镜头物体从上方和下方同时侵入画面，只剩中间一条窄窗口能看到业务区，即使设备局部可见，也属于多面前景夹击，应判 scene_usable=false。
近镜头立柱、门框或招牌形成连续纵向屏障，把业务区切成窄条时，也应判 scene_usable=false。
后厨、收银等作业场景必须包含可用工作平面、设备正面/操作界面、交易区或实际作业区域。只看到设备顶部、背面、外壳、布线、墙面、天花板或空地面时，按无业务锚点处理，scene_usable=false。
设备碎片、局部柜台、二维码/宣传贴纸、菜单板、墙角和空地面不能作为后厨/收银业务锚点。至少要有一个完整可用的操作面、完整工作台面、完整交易面或人员实际作业区域，否则 scene_usable=false。
设备内部机械、电机、接线、维护舱、电气控制腔和设备顶盖不属于业务界面，不能命中设备元素，也不能据此判 scene_usable=true。
人员必须与明确业务界面或作业区域一起出现。只有人员身体/脚步、地面和设备侧边外壳时，scene_usable=false；employee 本身不能作为有效覆盖的唯一证据。
仓库/储物区例外：纸箱、货架、货物和储物设备是正常内容。只要能大致看清仓库内部全貌和设施分布，scene_usable=true；局部遮挡导致全貌不确定时按不确定转人工；只有近镜头物体大面积盖住画面、无法看清整体布局时 scene_usable=false。
门口和店外场景不要求正对门洞或门头，也不要求当前有人流。能看到入口交界、人行道、店外桌椅、街道、停车区或其他目标外部区域，即可判 scene_usable=true；outdoor_view=unknown 不单独触发失败。
员工和顾客通常是监控业务主体，不是中层遮挡物；但如果人员形成横跨画面的大面积模糊前景带，遮住操作台、核心动作或主要业务区域，应按严重遮挡处理，scene_usable=false。
吊顶附近的线缆、管道、吊扇、灯具和支架横穿画面上方时，如果操作台、人员手部和核心作业面仍然清楚，不计入中层遮挡，也不能按线缆外接范围估计遮挡比例。
如果画面中至少一名员工的手部、手臂和具体工作动作清楚可见，且业务工作区仍能识别，前景网罩、边框、立柱或局部结构不能单独构成 narrow_side_view 或 scene_usable=false；只有核心业务活动确实无法观察时才判不可用。
物体名称、颜色和材质不影响判断。只有独立中间层实际盖住核心操作、把完整工位切成碎片，或业务目标本身不可识别时，scene_usable=false；否则 scene_usable=true。目标居中程度、画面占比、目标偏侧和固定结构面积均不能单独决定失败。
输出中文 evidence，并严格按结构输出。"""

OBSTRUCTION_VERIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target_area_description": {"type": "string"},
        "middle_layer_intrusion": {"type": "boolean"},
        "target_area_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "occupies_primary_view": {"type": "boolean"},
        "target_position": {
            "type": "string",
            "enum": ["center", "lower_center", "left_side", "right_side", "upper", "other"],
        },
        "narrow_side_view": {"type": "boolean"},
        "foreground_area_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "target_overlap_ratio": {"type": "number", "minimum": 0, "maximum": 1},
        "core_area_split": {"type": "boolean"},
        "scene_usable": {"type": "boolean"},
        "obstruction_type": {
            "type": "string",
            "enum": [
                "none",
                "plastic_bag",
                "packaging",
                "box",
                "pillar",
                "ceiling_fixture",
                "equipment",
                "other",
            ],
        },
        "evidence": {"type": "string"},
    },
    "required": [
        "target_area_description",
        "middle_layer_intrusion",
        "target_area_ratio",
        "occupies_primary_view",
        "target_position",
        "narrow_side_view",
        "foreground_area_ratio",
        "target_overlap_ratio",
        "core_area_split",
        "scene_usable",
        "obstruction_type",
        "evidence",
    ],
}


class VLMError(RuntimeError):
    pass


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _choice(value: Any, choices: Iterable[str], default: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text if text in set(choices) else default


def normalize_semantic(raw: Dict[str, Any]) -> Dict[str, Any]:
    """把模型自由格式输出收敛到稳定的业务枚举。"""
    camera_types = [
        "store_outdoor", "store_entrance", "kitchen", "cashier", "kitchen_cashier",
        "dining", "storage", "other_indoor", "unknown",
    ]
    camera_type = _choice(raw.get("camera_type"), camera_types, "unknown")
    is_outdoor = _as_bool(raw.get("is_store_outdoor"), camera_type == "store_outdoor")
    is_entrance_flow = _as_bool(
        raw.get("is_entrance_flow"), camera_type == "store_entrance"
    )
    if is_entrance_flow:
        camera_type = "store_entrance"
    elif camera_type == "store_outdoor":
        is_outdoor = True
    elif is_outdoor and camera_type == "unknown":
        camera_type = "store_outdoor"

    elements: List[Dict[str, Any]] = []
    seen = set()
    values = raw.get("visible_elements") or []
    if not isinstance(values, list):
        values = []
    for item in values:
        if isinstance(item, str):
            item = {"id": item, "visible": True, "confidence": 0.7, "evidence": ""}
        if not isinstance(item, dict):
            continue
        element_id = str(item.get("id") or "").strip().lower()
        if element_id not in CANONICAL_ELEMENTS or element_id in seen:
            continue
        seen.add(element_id)
        elements.append({
            "id": element_id,
            "label": CANONICAL_ELEMENTS[element_id]["label"],
            # 提示词要求只返回命中元素，部分模型会省略冗余的 visible=true。
            # 已出现在 visible_elements 中时默认视为命中；显式 false 仍保留。
            "visible": _as_bool(item.get("visible"), default=True),
            "confidence": _as_float(item.get("confidence")),
            "evidence": str(item.get("evidence") or "")[:120],
        })
    # 所有合法元素都进入结果，前端和统计可以看到“未命中”而不只是数组缺失。
    for element_id, meta in CANONICAL_ELEMENTS.items():
        if element_id not in seen:
            elements.append({
                "id": element_id,
                "label": meta["label"],
                "visible": False,
                "confidence": 0.0,
                "evidence": "",
            })
    elements.sort(key=lambda item: (-item["visible"], -item["confidence"], item["id"]))

    return {
        "camera_type": camera_type,
        "is_store_outdoor": is_outdoor,
        "is_entrance_flow": is_entrance_flow,
        "target_area_visible": _as_bool(raw.get("target_area_visible")),
        "occlusion": _choice(
            raw.get("occlusion"), ["none", "partial", "severe", "unknown"], "unknown"
        ),
        "occlusion_evidence": str(raw.get("occlusion_evidence") or "")[:240],
        "occlusion_area_ratio": _as_float(raw.get("occlusion_area_ratio")),
        "occlusion_affects_subject": _as_bool(raw.get("occlusion_affects_subject")),
        "occlusion_scope": _choice(
            raw.get("occlusion_scope"),
            ["none", "peripheral", "critical", "unknown"],
            "unknown",
        ),
        "blur": _choice(
            raw.get("blur"), ["none", "slight", "severe", "unknown"], "unknown"
        ),
        "blur_evidence": str(raw.get("blur_evidence") or "")[:240],
        "blur_scope": _choice(
            raw.get("blur_scope"),
            ["none", "localized", "global", "unknown"],
            "unknown",
        ),
        "operation_visibility": _choice(
            raw.get("operation_visibility"),
            ["clear", "partial", "obscured", "not_applicable", "unknown"],
            "unknown",
        ),
        "hand_action_visible": _as_bool(raw.get("hand_action_visible")),
        "monitoring_subject_affected": _as_bool(
            raw.get("monitoring_subject_affected")
        ),
        "obstruction_verified": _as_bool(raw.get("obstruction_verified")),
        "view_valid": _as_bool(raw.get("view_valid")),
        "view_issue": _choice(
            raw.get("view_issue"),
            [
                "proper",
                "wrong_area",
                "too_close",
                "wall_ceiling",
                "obstructed",
                "blurred",
                "unrecognized",
                "unknown",
            ],
            "unknown",
        ),
        "primary_subject_central": _as_bool(
            raw.get("primary_subject_central")
        ),
        "subject_coverage": _choice(
            raw.get("subject_coverage"),
            ["none", "small", "medium", "large"],
            "none",
        ),
        "view_evidence": str(raw.get("view_evidence") or "")[:240],
        "outdoor_view": _choice(
            raw.get("outdoor_view"),
            ["proper", "facing_sky", "facing_wall", "unknown"],
            "unknown",
        ),
        "angle_quality": _choice(
            raw.get("angle_quality"),
            ["good", "slightly_off", "bad", "unknown"],
            "unknown",
        ),
        "angle_evidence": str(raw.get("angle_evidence") or "")[:240],
        "visible_elements": elements,
        "business_scene_summary": str(raw.get("business_scene_summary") or "")[:240],
        "confidence": _as_float(raw.get("confidence")),
    }


def score_elements(semantic: Dict[str, Any], min_confidence: float = 0.55) -> Dict[str, Any]:
    elements = semantic.get("visible_elements") or []
    hit_ids = {
        item["id"]
        for item in elements
        if item.get("visible") and float(item.get("confidence", 0.0)) >= min_confidence
    }
    camera_type = semantic.get("camera_type", "unknown")
    expected = EXPECTED_ELEMENTS.get(camera_type, list(CANONICAL_ELEMENTS))
    expected = [element_id for element_id in expected if element_id in CANONICAL_ELEMENTS]
    expected_weight = sum(CANONICAL_ELEMENTS[element_id]["weight"] for element_id in expected)
    hit_weight = sum(
        CANONICAL_ELEMENTS[element_id]["weight"]
        for element_id in expected
        if element_id in hit_ids
    )
    score = hit_weight / expected_weight if expected_weight else 0.0
    return {
        "score": round(score, 4),
        "hit_ids": sorted(hit_ids),
        "hit_labels": [CANONICAL_ELEMENTS[element_id]["label"] for element_id in sorted(hit_ids)],
        "hit_count": len(hit_ids),
        "expected_ids": expected,
        "expected_hit_count": sum(1 for element_id in expected if element_id in hit_ids),
        "expected_count": len(expected),
    }


def _strip_fence(content: str) -> str:
    text = content.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if match:
        text = match.group(1).strip()
    return text


def parse_json_content(content: str) -> Dict[str, Any]:
    text = _strip_fence(content)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise VLMError("Qwen 未返回 JSON 对象")
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise VLMError(f"Qwen JSON 解析失败: {exc}") from exc
    if not isinstance(value, dict):
        raise VLMError("Qwen 返回结果不是 JSON 对象")
    return value


def repair_truncated_json(content: str) -> Dict[str, Any]:
    """Close a JSON object truncated by the model's output-token limit."""
    text = _strip_fence(content)
    start = text.find("{")
    if start < 0:
        raise VLMError("Qwen 返回内容中没有 JSON 对象")
    text = text[start:]
    stack: List[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if stack:
                stack.pop()
    repaired = text.rstrip()
    if in_string:
        repaired += '"'
    repaired = repaired.rstrip()
    if repaired.endswith(":"):
        repaired += "0"
    while repaired.endswith(","):
        repaired = repaired[:-1].rstrip()
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    try:
        value = json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise VLMError(f"Qwen JSON 无法修复: {exc}") from exc
    if not isinstance(value, dict):
        raise VLMError("Qwen 返回结果不是 JSON 对象")
    return value


def prepare_image(path: Path, max_side: int, jpeg_quality: int) -> Tuple[bytes, Dict[str, Any]]:
    data = path.read_bytes()
    image = cv2.imdecode(__import__("numpy").frombuffer(data, dtype="uint8"), cv2.IMREAD_COLOR)
    if image is None:
        raise VLMError(f"无法解码图片: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    quality = max(60, min(95, int(jpeg_quality)))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise VLMError("Qwen 输入图片编码失败")
    return encoded.tobytes(), {
        "source_width": width,
        "source_height": height,
        "input_width": int(image.shape[1]),
        "input_height": int(image.shape[0]),
        "jpeg_quality": quality,
        "encoded_bytes": int(encoded.nbytes),
    }

def _foreground_focus_suspicion(image_bytes: bytes) -> Dict[str, float]:
    """Detect a large low-focus border region relative to the sharp scene core."""
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"border_low_focus_ratio": 0.0, "reference_sharpness": 0.0}
    height, width = image.shape[:2]
    scale = min(1.0, 640.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    height, width = image.shape[:2]
    grid = np.zeros((4, 4), dtype=np.float64)
    for row in range(4):
        for col in range(4):
            tile = image[
                row * height // 4:(row + 1) * height // 4,
                col * width // 4:(col + 1) * width // 4,
            ]
            grid[row, col] = float(cv2.Laplacian(tile, cv2.CV_64F).var())
    reference = float(np.percentile(grid, 85))
    low_focus = grid < max(50.0, reference * 0.20)
    border = np.zeros((4, 4), dtype=bool)
    border[0, :] = border[3, :] = border[:, 0] = border[:, 3] = True
    return {
        "border_low_focus_ratio": round(float(np.sum(low_focus & border)) / 16.0, 4),
        "reference_sharpness": round(reference, 2),
    }


class QwenSemanticClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._obstruction_verify_lock: Optional[asyncio.Lock] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.vlm_enabled)

    @staticmethod
    def _store_name(camera: Dict[str, Any]) -> str:
        meta = camera.get("meta")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (TypeError, ValueError):
                meta = {}
        return str((meta or {}).get("store_name", ""))

    async def _verify_operation(
        self,
        image_bytes: bytes,
        camera: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_body = {
            "model": self.config.vlm_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": OPERATION_VERIFY_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(image_bytes).decode("ascii")
                        },
                    },
                ],
            }],
            "max_tokens": 180,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20261123,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "camangle_operation_verification",
                    "strict": True,
                    "schema": OPERATION_VERIFY_SCHEMA,
                },
            },
        }
        endpoint = f"{self.config.vlm_base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vlm_api_key}"
        async with httpx.AsyncClient(timeout=self.config.vlm_timeout, trust_env=False) as client:
            response = await client.post(endpoint, json=request_body, headers=headers)
            response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        raw = parse_json_content(choice["message"]["content"])
        return {
            "action_clear": _as_bool(raw.get("action_clear")),
            "human_present": _as_bool(raw.get("human_present")),
            "hands_arms_visible": _as_bool(raw.get("hands_arms_visible")),
            "work_area_visible": _as_bool(raw.get("work_area_visible")),
            "confidence": _as_float(raw.get("confidence")),
            "evidence": str(raw.get("evidence") or "")[:240],
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage") or {},
        }

    async def _verify_obstruction(
        self,
        image_bytes: bytes,
        camera: Dict[str, Any],
        seed: int = 20261123,
    ) -> Dict[str, Any]:
        request_body = {
            "model": self.config.vlm_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": OBSTRUCTION_VERIFY_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(image_bytes).decode("ascii")
                        },
                    },
                ],
            }],
            "max_tokens": 300,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "camangle_obstruction_verification",
                    "strict": True,
                    "schema": OBSTRUCTION_VERIFY_SCHEMA,
                },
            },
        }
        endpoint = f"{self.config.vlm_base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.vlm_api_key:
            headers["Authorization"] = f"Bearer {self.config.vlm_api_key}"
        if self._obstruction_verify_lock is None:
            self._obstruction_verify_lock = asyncio.Lock()
        async with self._obstruction_verify_lock:
            async with httpx.AsyncClient(
                timeout=self.config.vlm_timeout, trust_env=False
            ) as client:
                response = await client.post(endpoint, json=request_body, headers=headers)
                response.raise_for_status()
        payload = response.json()
        choice = payload["choices"][0]
        raw = parse_json_content(choice["message"]["content"])
        foreground_ratio = _as_float(raw.get("foreground_area_ratio"))
        target_overlap_ratio = _as_float(raw.get("target_overlap_ratio"))
        target_area_ratio = _as_float(raw.get("target_area_ratio"))
        core_area_split = _as_bool(raw.get("core_area_split"))
        scene_usable = _as_bool(raw.get("scene_usable"), default=True)
        middle_layer_intrusion = _as_bool(raw.get("middle_layer_intrusion"))
        occupies_primary_view = _as_bool(
            raw.get("occupies_primary_view"), default=True
        )
        target_position = _choice(
            raw.get("target_position"),
            ["center", "lower_center", "left_side", "right_side", "upper", "other"],
            "other",
        )
        narrow_side_view = _as_bool(raw.get("narrow_side_view"))
        return {
            "target_area_description": str(raw.get("target_area_description") or "")[:240],
            "middle_layer_intrusion": middle_layer_intrusion,
            "target_area_ratio": target_area_ratio,
            "occupies_primary_view": occupies_primary_view,
            "target_position": target_position,
            "narrow_side_view": narrow_side_view,
            "foreground_area_ratio": foreground_ratio,
            "target_overlap_ratio": target_overlap_ratio,
            "obstruction_area_ratio": target_overlap_ratio,
            "core_area_split": core_area_split,
            "scene_usable": scene_usable,
            "lens_obstructed": bool(
                middle_layer_intrusion
                and (
                    narrow_side_view
                    or target_position in {"left_side", "right_side"}
                    or not occupies_primary_view
                    or target_area_ratio < 0.25
                    or target_overlap_ratio >= 0.20
                    or core_area_split
                )
            ),
            "affects_core_area": bool(
                narrow_side_view or target_overlap_ratio >= 0.20 or core_area_split
            ),
            "obstruction_type": _choice(
                raw.get("obstruction_type"),
                [
                    "none",
                    "plastic_bag",
                    "packaging",
                    "box",
                    "pillar",
                    "ceiling_fixture",
                    "equipment",
                    "other",
                ],
                "none",
            ),
            "evidence": str(raw.get("evidence") or "")[:400],
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage") or {},
        }

    async def analyze(
        self,
        image_path: Path,
        camera: Dict[str, Any],
        prompts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "status": "skipped",
                "confidence": 0.0,
                "reason": "VLM 未启用",
                "data": {},
                "metrics": {},
            }

        started = time.perf_counter()
        content = ""
        usage: Dict[str, Any] = {}
        finish_reason = ""
        json_repaired = False
        try:
            image_bytes, image_meta = await asyncio.to_thread(
                prepare_image,
                image_path,
                self.config.vlm_max_image_side,
                self.config.vlm_jpeg_quality,
            )
            focus_metrics = await asyncio.to_thread(
                _foreground_focus_suspicion, image_bytes
            )
            active_prompts = prompts or self.config.vlm_prompts
            system_prompt = (
                active_prompts.get("system")
                if isinstance(active_prompts.get("system"), str) and active_prompts.get("system").strip()
                else DEFAULT_SYSTEM_PROMPT
            )
            user_prompt = (
                active_prompts.get("user")
                if isinstance(active_prompts.get("user"), str) and active_prompts.get("user").strip()
                else DEFAULT_USER_PROMPT
            )
            prompt = (
                f"摄像头通道：{camera.get('channel_name') or '未知'}\n"
                f"所属门店：{self._store_name(camera)}\n"
                f"{user_prompt}"
            )
            request_body = {
                "model": self.config.vlm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/jpeg;base64,"
                                    + base64.b64encode(image_bytes).decode("ascii")
                                },
                            },
                        ],
                    },
                ],
                # 输出包含场景、遮挡、模糊、操作可见性和元素证据。
                # 多元素画面需要更充足的空间，避免 JSON 在对象中途截断。
                "max_tokens": 1200,
                "temperature": 0.0,
                "top_p": 1.0,
                "frequency_penalty": 0.3,
                "seed": 20261123,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "camangle_semantic_result",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    },
                },
            }
            endpoint = f"{self.config.vlm_base_url}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.config.vlm_api_key:
                headers["Authorization"] = f"Bearer {self.config.vlm_api_key}"
            async with httpx.AsyncClient(timeout=self.config.vlm_timeout, trust_env=False) as client:
                response = await client.post(endpoint, json=request_body, headers=headers)
                if response.status_code == 400 and "response_format" in response.text:
                    # 兼容不支持 JSON Schema 的旧版 vLLM；提示词仍要求纯 JSON。
                    request_body["response_format"] = {"type": "json_object"}
                    response = await client.post(endpoint, json=request_body, headers=headers)
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                content = choice["message"]["content"]
                finish_reason = str(choice.get("finish_reason") or "")
                usage = payload.get("usage") or {}
                try:
                    raw = parse_json_content(content)
                except VLMError:
                    retry_prompt = (
                        prompt
                        + "\n\n【紧凑重试】只输出完整 JSON。"
                        "evidence 每条最多 30 个汉字，禁止重复词语、重复元素或列举清单；"
                        "visible_elements 最多 4 个。不要输出任何 JSON 之外的文字。"
                    )
                    retry_body = dict(request_body)
                    retry_body["messages"] = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": retry_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/jpeg;base64,"
                                        + base64.b64encode(image_bytes).decode("ascii")
                                    },
                                },
                            ],
                        },
                    ]
                    retry_body["max_tokens"] = 1200
                    retry_body["frequency_penalty"] = 0.7
                    retry_body["seed"] = 20261124
                    response = await client.post(
                        endpoint, json=retry_body, headers=headers
                    )
                    response.raise_for_status()
                    payload = response.json()
                    choice = payload["choices"][0]
                    content = choice["message"]["content"]
                    finish_reason = str(choice.get("finish_reason") or "")
                    usage = payload.get("usage") or {}
                    try:
                        raw = parse_json_content(content)
                    except VLMError:
                        if finish_reason != "length":
                            raise
                        raw = repair_truncated_json(content)
                        json_repaired = True
            semantic = normalize_semantic(raw)
            obstruction_verification = None
            main_occlusion_evidence = semantic["occlusion_evidence"]
            semantic["main_occlusion_evidence"] = main_occlusion_evidence
            semantic["main_target_area_visible"] = semantic["target_area_visible"]
            semantic["main_view_valid"] = semantic["view_valid"]
            semantic["main_subject_coverage"] = semantic["subject_coverage"]
            semantic["main_occlusion_area_ratio"] = semantic["occlusion_area_ratio"]
            fixed_plane_evidence = bool(re.search(
                r"广告屏|菜单屏|背板|屏幕|反光|玻璃|门框|墙体|墙面|固定隔断",
                main_occlusion_evidence,
            ))
            visible_anchor = any(
                item.get("visible")
                and float(item.get("confidence", 0.0)) >= 0.55
                and item.get("id") in {
                    "counter",
                    "cash_register",
                    "oil_fryer",
                    "range_hood",
                    "sink",
                    "prep_area",
                    "breading_area",
                    "refrigerator",
                    "display_case",
                }
                for item in semantic["visible_elements"]
            )
            if fixed_plane_evidence and visible_anchor:
                semantic["occlusion"] = "none"
                semantic["occlusion_area_ratio"] = 0.0
                semantic["occlusion_scope"] = "none"
                semantic["occlusion_affects_subject"] = False
                semantic["monitoring_subject_affected"] = False
                if semantic["blur"] == "severe":
                    semantic["blur"] = "slight"
                    semantic["blur_scope"] = "localized"
            line_like_low_impact = (
                bool(re.search(
                    r"线缆|电线|线状|线束|吊线|灯杆|细杆|护网|支架|长条|横梁|轨道|管道|通风",
                    main_occlusion_evidence,
                ))
                and (
                    not semantic["occlusion_affects_subject"]
                    or bool(re.search(
                        r"未切断|未遮挡|不影响主体|仍清晰|仍可辨",
                        main_occlusion_evidence,
                    ))
                )
            )
            fixture_like_low_impact = (
                fixed_plane_evidence
            )
            boundary_like_low_impact = fixed_plane_evidence
            if (
                (
                    (
                        semantic["occlusion"] == "partial"
                        and (
                            semantic["occlusion_affects_subject"]
                            or semantic["occlusion_scope"] == "critical"
                            or semantic["occlusion_area_ratio"] >= 0.30
                        )
                    )
                    or (
                        semantic["occlusion"] == "none"
                        and semantic["occlusion_area_ratio"] >= 0.10
                        and bool(main_occlusion_evidence)
                        and semantic["target_area_visible"]
                    )
                    or (
                        focus_metrics["border_low_focus_ratio"] >= 0.20
                        and semantic["target_area_visible"]
                        and semantic["camera_type"] in {
                            "kitchen",
                            "cashier",
                            "kitchen_cashier",
                            "storage",
                            "other_indoor",
                        }
                    )
                )
                and not line_like_low_impact
                and not fixture_like_low_impact
                and not boundary_like_low_impact
            ):
                try:
                    obstruction_verification = await self._verify_obstruction(
                        image_bytes, camera
                    )
                    verified_ratio = max(
                        obstruction_verification["foreground_area_ratio"],
                        obstruction_verification["target_overlap_ratio"],
                    )
                    if (
                        obstruction_verification["lens_obstructed"]
                        and not obstruction_verification["scene_usable"]
                    ):
                        semantic["obstruction_verified"] = True
                        semantic["occlusion"] = "severe"
                        semantic["occlusion_area_ratio"] = max(
                            semantic["occlusion_area_ratio"], verified_ratio
                        )
                        semantic["occlusion_scope"] = "critical"
                        semantic["occlusion_affects_subject"] = True
                        semantic["monitoring_subject_affected"] = True
                        semantic["view_valid"] = False
                        semantic["view_issue"] = "obstructed"
                        semantic["primary_subject_central"] = False
                        semantic["subject_coverage"] = "small"
                        semantic["target_area_visible"] = False
                        semantic["occlusion_evidence"] = obstruction_verification["evidence"]
                        semantic["view_evidence"] = (
                            "遮挡专项复核：" + obstruction_verification["evidence"]
                        )
                        semantic["business_scene_summary"] = (
                            "遮挡专项复核判定画面不可用："
                            + obstruction_verification["evidence"]
                        )[:240]
                except Exception as verify_exc:
                    obstruction_verification = {
                        "lens_obstructed": False,
                        "obstruction_area_ratio": 0.0,
                        "affects_core_area": False,
                        "scene_usable": True,
                        "obstruction_type": "none",
                        "evidence": "",
                        "error": str(verify_exc),
                    }
            operation_verification = None
            if (
                semantic["operation_visibility"] in {"obscured", "unknown"}
                or semantic["monitoring_subject_affected"]
            ):
                try:
                    operation_verification = await self._verify_operation(
                        image_bytes, camera
                    )
                    if (
                        operation_verification["action_clear"]
                        and operation_verification["confidence"] >= 0.60
                    ):
                        semantic["hand_action_visible"] = True
                        semantic["operation_visibility"] = "clear"
                        semantic["monitoring_subject_affected"] = False
                        semantic["occlusion_affects_subject"] = False
                except Exception as verify_exc:
                    operation_verification = {
                        "action_clear": False,
                        "confidence": 0.0,
                        "error": str(verify_exc),
                    }
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
            return {
                # JSON 成功解析即代表模型可用；unknown 是有效的业务结论，
                # 不能与网络/解析失败混为一谈。
                "status": "pass",
                "confidence": semantic["confidence"],
                "reason": semantic["business_scene_summary"],
                "data": semantic,
                "metrics": {
                    "vlm_latency_ms": elapsed_ms,
                    "vlm_prompt_tokens": usage.get("prompt_tokens"),
                    "vlm_completion_tokens": usage.get("completion_tokens"),
                    "vlm_finish_reason": finish_reason,
                    "vlm_json_repaired": json_repaired,
                    "vlm_input": image_meta,
                    "vlm_foreground_focus_suspicion": focus_metrics,
                },
                "request": {
                    "endpoint": endpoint,
                    "model": self.config.vlm_model,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "image": {
                        "source": str(image_path),
                        **image_meta,
                    },
                },
                "raw_response": content,
                "obstruction_verification": obstruction_verification,
                "operation_verification": operation_verification,
                "_prepared_image": image_bytes,
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
            return {
                "status": "unknown",
                "confidence": 0.0,
                "reason": f"Qwen 语义分析不可用: {exc}",
                "data": {},
                "metrics": {
                    "vlm_latency_ms": elapsed_ms,
                    "vlm_prompt_tokens": usage.get("prompt_tokens"),
                    "vlm_completion_tokens": usage.get("completion_tokens"),
                    "vlm_finish_reason": finish_reason,
                },
                "request": {
                    "endpoint": f"{self.config.vlm_base_url}/chat/completions",
                    "model": self.config.vlm_model,
                    "image": {"source": str(image_path)},
                },
                "raw_response": content,
                "error": str(exc),
            }
