from fastapi import FastAPI, UploadFile, Form, Depends, File, APIRouter
from fastapi.exceptions import HTTPException
import onnxruntime
from src.face_judgement_align import IDphotos_create
from src.layoutCreate import generate_layout_photo, generate_layout_image
from hivisionai.hycv.vision import add_background
import base64
import numpy as np
import cv2
import ast

#######
from meutils.pipe import logger, Optional
from meutils.serving.fastapi import App
from meutils.io.files_utils import to_bytes

from meutils.oss.minio_oss import Minio
from meutils.schemas.idphoto_types import SIZES, COLORS
from meutils.llm.openai_utils import ppu_flow
from meutils.serving.fastapi.dependencies.auth import get_bearer_token, HTTPAuthorizationCredentials

router = APIRouter()

# 加载权重文件
HY_HUMAN_MATTING_WEIGHTS_PATH = "./hivision_modnet.onnx"
sess = onnxruntime.InferenceSession(HY_HUMAN_MATTING_WEIGHTS_PATH)


# 将图像转换为 Base64 编码


def numpy_2_base64(img: np.ndarray):
    retval, buffer = cv2.imencode(".png", img)
    _ = base64.b64encode(buffer).decode("utf-8")
    return _


async def numpy_to_url(img: np.ndarray):
    retval, buffer = cv2.imencode(".png", img)

    file = buffer.tobytes()
    content_type = "image/png"

    file_object = await Minio().put_object_for_openai(file=file, bucket_name='caches', content_type=content_type)
    return file_object.filename


# 证件照智能制作接口
@router.post("/generations")
async def idphoto_inference(
        file: Optional[UploadFile] = None,
        url: Optional[str] = Form(None),

        size: str = Form("国家公务员考试"),
        background_color: str = Form("白色"),  # "蓝色", "白色", "红色"
        render_mode: str = Form('pure_color'),  # updown_gradient center_gradient

        head_measure_ratio: float = 0.2,
        head_height_ratio: float = 0.45,
        top_distance_max: float = 0.12,
        top_distance_min: float = 0.10,

        auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),

):
    api_key = auth and auth.credentials or None

    image_bytes = await to_bytes(file) or await to_bytes(url)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 将字符串转为元组
    size = SIZES.get(size, size)
    size = ast.literal_eval(size)

    # COLORS
    background_color = COLORS.get(background_color, background_color)
    background_color = ast.literal_eval(background_color)[::-1]

    logger.debug(size)
    logger.debug(background_color)

    async with ppu_flow(api_key=api_key, post='api-idphotos'):

        (
            result_image_hd,
            result_image_standard,
            typography_arr,
            typography_rotate,
            _,
            _,
            _,
            _,
            status,
        ) = IDphotos_create(
            img,
            size=size,
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
            align=False,
            beauty=False,
            fd68=None,
            human_sess=sess,
            IS_DEBUG=False,
            top_distance_max=top_distance_max,
            top_distance_min=top_distance_min,
        )

        # 如果检测到人脸数量不等于 1（照片无人脸 or 多人脸）
        if status == 0:
            raise HTTPException(status_code=401, detail="请检查照片：照片无人脸 or 多人脸")

        # 如果检测到人脸数量等于 1, 则返回标准证和高清照结果（png 4 通道图像）
        else:
            # if background_color != (255, 255, 255):  # 默认白色

            result_image_standard = np.uint8(
                add_background(
                    result_image_standard,
                    bgr=background_color,
                    mode=render_mode,
                )
            )
            result_image_hd = np.uint8(
                add_background(
                    result_image_hd,
                    bgr=background_color,
                    mode=render_mode,
                )
            )

            result_messgae = {
                "status": True,
                "img_output_standard": await numpy_to_url(result_image_standard),
                "img_output_standard_hd": await numpy_to_url(result_image_hd),
            }

        return result_messgae


# 透明图像添加纯色背景接口
@router.post("/add_background")
async def photo_add_background(input_image: UploadFile, color: str = Form(...)):
    # 读取图像
    image_bytes = await input_image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # 将字符串转为元组
    color = ast.literal_eval(color)
    # 将元祖的 0 和 2 号数字交换
    color = (color[2], color[1], color[0])

    # try:
    result_messgae = {
        "status": True,
        "image": await numpy_to_url(add_background(img, bgr=color)),
    }

    # except Exception as e:
    #     print(e)
    #     result_messgae = {
    #         "status": False,
    #         "error": e
    #     }

    return result_messgae


# 六寸排版照生成接口
@router.post("/generate_layout_photos")
async def generate_layout_photos(input_image: UploadFile, size: str = Form("(413, 295)")):
    try:
        image_bytes = await input_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        size = SIZES.get(size, size)
        size = ast.literal_eval(size)

        typography_arr, typography_rotate = generate_layout_photo(
            input_height=size[0], input_width=size[1]
        )

        result_layout_image = generate_layout_image(
            img, typography_arr, typography_rotate, height=size[0], width=size[1]
        )

        result_messgae = {
            "status": True,
            "image": await numpy_to_url(result_layout_image),
        }

    except Exception as e:
        result_messgae = {
            "status": False,
        }

    return result_messgae


from routers import videos

app = App()
app.include_router(router, "/v1/idphotos", tags=['证件照'])
app.include_router(videos.router, "/v1/videos", tags=videos.TAGS)

if __name__ == '__main__':
    app.run()
