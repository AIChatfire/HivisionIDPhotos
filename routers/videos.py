#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : AI.  @by PyCharm
# @File         : video
# @Time         : 2024/9/6 10:07
# @Author       : betterme
# @WeChat       : meutils
# @Software     : PyCharm
# @Description  :
from meutils.pipe import *
from meutils.ai_video.video import video2audio
from meutils.io.files_utils import to_bytes
from meutils.oss.minio_oss import Minio
from meutils.llm.openai_utils import ppu_flow

from meutils.serving.fastapi.dependencies.auth import get_bearer_token, HTTPAuthorizationCredentials

from fastapi import APIRouter, File, UploadFile, Query, Form, Depends, Request, HTTPException, status, BackgroundTasks

router = APIRouter()
TAGS = ["视频"]


@router.post("/video2audio")
async def create_video2audio(
        file: Optional[UploadFile] = None,
        url: Optional[str] = Form(None),

        auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),

):
    api_key = auth and auth.credentials or None

    async with ppu_flow(api_key, post='api-video2audio'):
        file_bytes = await to_bytes(file) or await to_bytes(url)
        with tempfile.NamedTemporaryFile(mode='wb+') as file, \
                tempfile.NamedTemporaryFile(mode='wb+', suffix=".mp3") as audio:
            file.write(file_bytes)
            # file.seek(0)

            video2audio(file.name, audio.name)

            file_object = await Minio().put_object_for_openai(
                audio.read(), bucket_name="caches",
                content_type='audio/mpeg'
            )
            return file_object


if __name__ == '__main__':
    from meutils.serving.fastapi import App

    app = App()

    app.include_router(router, '/v1')

    app.run(port=8888)
