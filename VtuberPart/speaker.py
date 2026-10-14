"""
将文字转为语音输出

"""



import asyncio
import edge_tts
import pygame



from config import MP3_PATH, TTS_VOICE, TTS_RATE, TTS_PITCH


def Text2Speak_Save(text, voice, rate, pitch, path):
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    asyncio.run(communicate.save(path))

def speak(text):
    Text2Speak_Save(text, voice=TTS_VOICE, rate=TTS_RATE, pitch=TTS_PITCH, path=MP3_PATH)
    pygame.mixer.init()
    pygame.mixer.music.load(MP3_PATH)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():     #没播完 每隔10ms问一次
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()              #播完释放文件句柄, 否则下一轮edge-tts写不进



if __name__ == '__main__':
    text = "你好可爱，哈哈哈哈哈"
    speak(text)

    

