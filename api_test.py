from multiprocessing import Process
import requests
import json
import os


def make_request(num):

    text = "测试测试，这里是测试"
    print(num)

    if num % 2 == 0:

        response = requests.get(f"http://127.0.0.1:9880/?text={text}&text_lang=zh&ref_audio_path=./参考音频/[jok老师]说得好像您带我以来我考好过几次一样.wav&prompt_lang=zh&prompt_text=说得好像您带我以来我考好过几次一样&text_split_method=cut5&batch_size=20&media_type=wav&speed_factor=1.0")
        audio_data = response.content

    else:

        response = requests.get(f"http://127.0.0.1:9880/?text={text}&text_lang=zh&ref_audio_path=./参考音频/[团长_愤怒]哪些死去士兵的意义将由我们来赋予.wav&prompt_lang=zh&prompt_text=哪些死去士兵的意义将由我们来赋予&text_split_method=cut5&batch_size=20&media_type=wav&speed_factor=1.0")
        audio_data = response.content

    with open(f"{text}_{num}.wav","wb") as f:
        f.write(audio_data)


if __name__ == '__main__':

    
    p_list = [Process(target=make_request,args=(x,)) for x in range(2)]

    for p in p_list:
        p.start()
    for p in p_list:
        p.join()

    print('main process end')
