from waifu.Waifu import Waifu
from waifu.StreamCallback import WaifuCallback
from waifu.llm.GPT import GPT
from waifu.llm.Claude import Claude
from waifu.llm.GLM import GLM
from tts.TTS import TTS
from tts.edge.edge import speak
from qqbot.qqbot import make_qq_bot
from waifu.Tools import load_prompt, load_emoticon, load_memory, str2bool
import configparser
from time import sleep

config = configparser.ConfigParser()

# 读取配置文件
config_files = config.read('config.ini', 'utf-8')
if len(config_files) == 0:
    raise FileNotFoundError('配置文件 config.ini 未找到，请检查是否配置正确！')

# CyberWaifu 配置
name 		 = config['CyberWaifu']['name']
username     = config['CyberWaifu']['username']
charactor 	 = config['CyberWaifu']['charactor']
send_text    = str2bool(config['CyberWaifu']['send_text'])
send_voice   = str2bool(config['CyberWaifu']['send_voice'])
use_emoji 	 = str2bool(config['Thoughts']['use_emoji'])
use_qqface   = str2bool(config['Thoughts']['use_qqface'])
use_emoticon = str2bool(config['Thoughts']['use_emoticon'])
use_search 	 = str2bool(config['Thoughts']['use_search'])
use_emotion  = str2bool(config['Thoughts']['use_emotion'])
search_api	 = config['Thoughts_GoogleSerperAPI']['api']
voice 		 = config['TTS']['voice']

prompt = load_prompt(charactor)

# 语音配置
tts_model = config['TTS']['model']
if tts_model == 'Edge':
	tts = TTS(speak, voice)
	api = config['TTS_Edge']['azure_speech_key']
	if api == '':
		use_emotion = False

# Thoughts 思考链配置
emoticons = config.items('Thoughts_Emoticon')
load_emoticon(emoticons)

# LLM 模型配置
model = config['LLM']['model']
if model == 'OpenAI':
    openai_api = config['LLM_OpenAI']['openai_key']
    callback = WaifuCallback(tts, send_text, send_voice)
    brain = GPT(openai_api, name, stream=True, callback=callback)
elif model == 'Claude':
	callback = None
	user_oauth_token = config['LLM_Claude']['user_oauth_token']
	bot_id = config['LLM_Claude']['bot_id']
	brain = Claude(bot_id, user_oauth_token, name)
elif model == 'GLM':
	callback = None
	brain = GLM('x', 'x', False, model="THUDM/chatglm2-6b", is_cuda=True)

waifu = Waifu(brain=brain,
				prompt=prompt,
				name=name,
                username=username,
				use_search=use_search,
				search_api=search_api,
				use_emoji=use_emoji,
				use_qqface=use_qqface,
                use_emotion=use_emotion,
				use_emoticon=use_emoticon)

# 记忆导入
filename = config['CyberWaifu']['memory']
if filename != '':
	memory = load_memory(filename, waifu.name)
	waifu.import_memory_dataset(memory)


from langchainex.stream_whisper import StreamWhisper
# from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import threading
i_whisper = StreamWhisper(non_english=True, is_cuda=True)
AudioCount = 0

def check_answer():
    """
    如果AI没有在生成回复且队列中还有问题 则创建一个生成的线程
    :return:
    """
    global i_whisper
    global waifu
    global AudioCount
    while True:
        if(i_whisper.QueryLen() <= 0):
            sleep(0.25)
            continue
        text = i_whisper.getQuery()
        print(text)
        response = waifu.ask(text)
        print(response)
        with open("./output/output.txt", "w", encoding="utf-8") as f:
            f.write(f"{response}")  # 将要读的回复写入临时文件
        subprocess.run(f'edge-tts --voice zh-CN-XiaoyiNeural --f .\output\output.txt --write-media .\output\output{AudioCount}.mp3', shell=True)  # 执行命令行指令
        subprocess.run(f'mpv.exe -vo null .\output\output{AudioCount}.mp3 1>nul', shell=True)  # 执行命令行指令
        AudioCount += 1




if __name__ == "__main__":
    # sched1 = BackgroundScheduler(timezone="Asia/Shanghai")

    # sched1.add_job(check_answer, 'interval', seconds=1, id=f'answer', max_instances=1)
    # sched1.start()
    thread = threading.Thread(target=check_answer, name="check_answers")
    thread.start()
    i_whisper.listen_mic()

# if model == 'OpenAI':
# 	callback.register(waifu)
# print(waifu.ask("How do you do?"))
# make_qq_bot(callback, waifu, send_text, send_voice, tts)