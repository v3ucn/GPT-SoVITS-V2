from multiprocessing import Process
import requests
import json
import os


def make_request(text,num):

    text = text[0]

    print(text)
    print(num)

    if num % 2 == 0:

        requests.get("http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_weights/Keira-e15.ckpt")
        requests.get("http://127.0.0.1:9880/set_sovits_weights?weights_path=SoVITS_weights/Keira_e8_s96.pth")

        response = requests.get(f"http://127.0.0.1:9880/?text={text}&text_lang=zh&ref_audio_path=./Keira.wav&prompt_lang=zh&prompt_text=光动嘴不如亲自做给你看,等我一下呀&text_split_method=cut5&batch_size=10&media_type=wav&speed_factor=1.0")
        audio_data = response.content

    else:

        requests.get("http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_weights/gakki-e15.ckpt")
        requests.get("http://127.0.0.1:9880/set_sovits_weights?weights_path=SoVITS_weights/gakki_e8_s104.pth")

        response = requests.get(f"http://127.0.0.1:9880/?text={text}&text_lang=zh&ref_audio_path=./gakki.wav&prompt_lang=ja&prompt_text=恥を捨てて、頑張れたというか&text_split_method=cut5&batch_size=20&media_type=wav&speed_factor=1.0")
        audio_data = response.content

    with open(f"{text}_{num}.wav","wb") as f:
        f.write(audio_data)



text = "测试测试，这里是测试"
   

if __name__ == '__main__':

    # anae = os.path.basename("./Keira.wav")

    # print(anae)

    res = requests.get("https://u393077-8021-91e43fcf.cqa1.seetacloud.com:8443/set_gpt_weights?weights_path=/root/autodl-tmp/GPT_weights/老嫂子.ckpt")

    print(res.content)

    res = requests.get("https://u393077-8021-91e43fcf.cqa1.seetacloud.com:8443/set_sovits_weights?weights_path=/root/autodl-tmp/SoVITS_weights/老嫂子.pth")

    print(res.content)

    headers = {'Content-Type': 'application/json'}

    gpt = {"text":text,"text_lang":"zh","ref_audio_path":"./audio/老嫂子#zh#这不仅是对他国主权的侵犯，也可能引发不必要的冲突。.wav","prompt_lang":"zh","prompt_text":"这不仅是对他国主权的侵犯，也可能引发不必要的冲突。","text_split_method":"cut5","batch_size":5,"speed_factor":1.0}

    response = requests.post("https://u393077-8021-91e43fcf.cqa1.seetacloud.com:8443/",data=json.dumps(gpt),headers=headers)

    # response = requests.get(f"https://u393077-8021-cce6d924.cqa1.seetacloud.com:8443/?text={text}&text_lang=zh&ref_audio_path=./audio/嫂子#zh#咱们不管是送人，还是自用，都特别划算，而且给大家用的，都是这种专用的泡沫箱.wav&prompt_lang=zh&prompt_text=咱们不管是送人，还是自用，都特别划算，而且给大家用的，都是这种专用的泡沫箱&text_split_method=cut5&batch_size=1&media_type=wav&speed_factor=1.0")

    print(response.content)


    audio_data = response.content

    with open(f"new.wav","wb") as f:
        f.write(audio_data)




    # p_list = [Process(target=make_request,args=(text,x)) for x in range(2)]

    # for p in p_list:
    #     p.start()
    # for p in p_list:
    #     p.join()

    # print('main process end')
