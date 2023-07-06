import datetime
import queue
import subprocess
import threading

from apscheduler.schedulers.asyncio import AsyncIOScheduler


QuestionList = queue.Queue(10)  # 定义问题 用户名 回复 播放列表 四个先进先出队列
QuestionName = queue.Queue(10)
AnswerList = queue.Queue()
MpvList = queue.Queue()
LogsList = queue.Queue()
is_ai_ready = True  # 定义chatglm是否转换完成标志
is_tts_ready = True  # 定义语音是否生成完成标志
is_mpv_ready = True  # 定义是否播放完成标志
AudioCount = 0


def check_tts():
    """
    如果语音已经放完且队列中还有回复 则创建一个生成并播放TTS的线程
    :return:
    """
    global is_tts_ready
    if not AnswerList.empty() and is_tts_ready:
        is_tts_ready = False
        tts_thread = threading.Thread(target=tts_generate())
        tts_thread.start()


def tts_generate():
    """
    从回复队列中提取一条，通过edge-tts生成语音对应AudioCount编号语音
    :return:
    """
    global is_tts_ready
    global AnswerList
    global MpvList
    global AudioCount
    response = AnswerList.get()
    print(f"\033[31m[ChatGLM]\033[0m{response}")  # 打印AI回复信息
    with open("./output/output.txt", "w", encoding="utf-8") as f:
        f.write(f"{response}")  # 将要读的回复写入临时文件
    subprocess.run(f'edge-tts --voice zh-CN-XiaoyiNeural --f .\output\output.txt --write-media .\output\output{AudioCount}.mp3', shell=True)  # 执行命令行指令
    begin_name = response.find('回复')
    end_name = response.find("：")
    name = response[begin_name+2:end_name]
    print(f'\033[32mSystem>>\033[0m对[{name}]的回复已成功转换为语音并缓存为output{AudioCount}.mp3')
    MpvList.put(AudioCount)
    AudioCount += 1
    is_tts_ready = True  # 指示TTS已经准备好回复下一个问题


def check_mpv():
    """
    若mpv已经播放完毕且播放列表中有数据 则创建一个播放音频的线程
    :return:
    """
    global is_mpv_ready
    global MpvList
    if not MpvList.empty() and is_mpv_ready:
        is_mpv_ready = False
        tts_thread = threading.Thread(target=mpv_read())
        tts_thread.start()


def mpv_read():
    """
    按照MpvList内的名单播放音频直到播放完毕
    :return:
    """
    global MpvList
    global is_mpv_ready
    while not MpvList.empty():
        temp1 = MpvList.get()
        current_mpvlist_count = MpvList.qsize()
        print(f'\033[32mSystem>>\033[0m开始播放output{temp1}.mp3，当前待播语音数：{current_mpvlist_count}')
        subprocess.run(f'mpv.exe -vo null .\output\output{temp1}.mp3 1>nul', shell=True)  # 执行命令行指令
        subprocess.run(f'del /f .\output\output{temp1}.mp3 1>nul', shell=True)
    is_mpv_ready = True
