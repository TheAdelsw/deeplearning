import asyncio, math, pyvts

async def main():
    vts = pyvts.vts(plugin_info={"plugin_name": "liangshi", "developer": "TheAdelsw", "authentication_token_path": r"D:\Project\deeplearning\VtuberPart\vts_token.txt"})                    # 建对象，给插件起名 liangshi
    await vts.connect()                     #拨电话（跑之前 VTS 必须开着、API 已启用）
    await vts.request_authenticate_token()  #要通行证（首次会弹窗！盯着 VTS 窗口点允许）
    await vts.request_authenticate()        #出示通行证

    # ④ 30fps 循环约 3 秒（~90 帧）：
    #    t = 帧号 / 30
    #    value = (math.sin(t * 若干圈数) + 1) / 2   # 挤进 0~1 区间
    #    msg = vts.vts_request.requestSetParameterValue(parameter="MouthOpenY", value=value)
    #    await vts.request(msg)
    #    await asyncio.sleep(0.033)              # 1/30 秒，asyncio 版的 time.sleep（想想为什么必须用这个）

    for i in range(180):
        t = i / 30
        value = (math.sin(t * 12) + 1) / 2
        #value = 1.0 if math.sin(t * 3) > 0 else 0.0   # 张1秒、闭1秒，干脆利落
        msg = vts.vts_request.requestSetParameterValue(parameter="MouthOpen", value=value)
        # msg = vts.vts_request.requestSetParameterValue(parameter="EyeOpenRight", value=0.0)
        await vts.request(msg)
        await asyncio.sleep(0.033) 


    await vts.close()                       #挂电话

if __name__ == '__main__':

    asyncio.run(main())
