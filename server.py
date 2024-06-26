import shutil

import gradio as gr
from gradio.components.chatbot import FileMessage
from gradio.data_classes import FileData

from build_rag import retrieve, get_retriever_engine, retrieve_image_to_image
from PIL import Image
import numpy as np
import sys
from model_LLM import OurLLM, MiniCPM, MobileVLM


model_name = sys.argv[1]
if model_name == "MobileVLM":
    llm = MobileVLM()
elif model_name == "MiniCPM":
    llm = MiniCPM()

def complex_analysis(retriever_engine_dict,query_str, image, image_file_name):
    retriever_engine = retriever_engine_dict["retriever_engine"]
    img, txt = retrieve(retriever_engine, query_str)

    if image is not None and img == []:
        image = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
        # 保存Image对象为文件
        image.save("storage/cache/%s" % (image_file_name))
        img1 = retrieve_image_to_image(retriever_engine, "storage/cache/%s"%(image_file_name))
        set1 = set(img1)
        set2 = set(img)
        union_set = set1.union(set2)
        img = list(union_set)

    if txt != []:
        context_str = ";".join(txt)
    else:
        context_str = "没有上下文，直接回答"

    if img == []:
        return  context_str, None, None
    return context_str, img, img[0]

def call_LLM(prompt, image_path,image=None):
    if image is not None:
        image = Image.open(io.BytesIO(base64.b64decode(image))).convert('RGB')
        image.save(image_path)

    response = llm.chat({0: prompt, 1:image_path}).replace("<|endoftext|>","")
    return response


import base64
import io
import os.path
import rarfile
import gradio as gr

from PIL import Image

import zipfile
def upload_img(file, _chatbot, _app_session):
    _app_session['sts'] = None
    _app_session['ctx'] = []

    with open(file, 'rb') as f1:
        base64_str = base64.b64encode(f1.read()) # str类型
    _app_session['file'] = base64_str
    _app_session['file_name'] = os.path.basename(file)

    return _chatbot, _app_session

def unzip(zip_file, output=None):
    """解压zip文件"""
    if not os.path.exists(output):
        os.mkdir(output)
    #zip = zipfile.ZipFile(zip_file)
    output = output or os.path.dirname(zip_file)  # 默认解压到当前目录同名文件夹中
    # 返回所有文件夹和文件
    #zip_list = zip.namelist()
    #zip_list_new = []
    # 确保输出目录存在

    # 打开 ZIP 文件
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for zip_info in zip_ref.infolist():
            # 解决中文文件名乱码问题
            zip_info.filename = zip_info.filename.encode('cp437').decode('gbk')
            zip_ref.extract(zip_info, output)

def unrar(zip_file, output=None):
    rf = rarfile.RarFile(zip_file)
    rf.extractall(output)

def respond(_app_cfg, _chat_bot):
    if _app_cfg['file'] is None:
        _app_cfg['ret'] = "您还没上传任何文件"
        return '', _app_cfg, _chat_bot

    debase = base64.b64decode(_app_cfg['file'])
    with open("storage/"+_app_cfg["file_name"], 'wb') as f1:
        f1.write(debase)  # str类型
    if _app_cfg["file_name"].endswith(".zip"):
        unzip("storage/"+_app_cfg["file_name"],"storage/decompress")
    if _app_cfg["file_name"].endswith(".rar"):
        unrar("storage/"+_app_cfg["file_name"],"storage/decompress")

    if _app_cfg['ret'] is not None and not isinstance(_app_cfg['ret'], str):
        _app_cfg['ret']["client"].close()
    code, message = get_retriever_engine("storage/decompress")
    if code == 200:
        _app_cfg['ret'] = message
        _app_cfg['message'] = "success"
        _chat_bot.append(('', '知识库更新成功，现在你可以和我聊天啦！'))
        return "success", _app_cfg, _chat_bot
    else:
        _app_cfg['ret'] = None
        _app_cfg['message'] = message
        _chat_bot.append(('', '知识库更新失败，原因：%s'%message))
        return message, _app_cfg, _chat_bot

def respond1(message, _chat_bot, _app_cfg, prompt):
    if _app_cfg['ctx'] is None:
        _app_cfg['ctx'] = []

    base64_str = None
    image_file_name = None
    if message['files'] is not None:
        for item in message['files']:
            with open(item, 'rb') as f1:
                base64_str = base64.b64encode(f1.read()) # str类型
                image_file_name = os.path.basename(item)

    _question = message["text"]
    _context = _app_cfg['ctx'].copy()
    if _context:
        _context.append({"role": "user", "content": _question})
    else:
        _context = [{"role": "user", "content": _question}]

    print('<User>:', _question)

    if _app_cfg['ret'] is None:
        if image_file_name is None:
            _answer = call_LLM(_question,None)
            _app_cfg['ctx'] = _context
            _app_cfg['sts'] = 200
            _context.append({"role": "assistant", "content": _answer})
            _chat_bot.append((_question, _answer))

        else:
            _answer = call_LLM(_question,image_path=os.path.join("storage","cache", image_file_name),image=base64_str)
            _app_cfg['ctx'] = _context
            _app_cfg['sts'] = 200
            _app_cfg['img']= os.path.join("storage","cache",image_file_name)
            _context.append({"role": "assistant", "content": _answer, "img": _app_cfg['img']})
            _chat_bot.append(([_app_cfg['img'].__str__(), _question], _answer))
        print('<Assistant>:', _answer)

        #_chat_bot.append(("", [_app_cfg['img'].__str__(), _answer]))
        message["text"] = ""
        message['files'] = None
        return gr.MultimodalTextbox(interactive=True, file_types=["image"],
                                      placeholder="输入消息或者上传图片", show_label=False), _chat_bot, _app_cfg

    _knowledge, img, img_path = complex_analysis(_app_cfg['ret'], _question, base64_str, image_file_name)
    try:
        _answer = call_LLM("上下文是:"+_knowledge +"。问题是：" +_question + prompt, img_path)
    except Exception as e:
        _answer = "调用大模型出现错误，错误原因: %s"%(e.__str__())

    _context.append({"role": "assistant", "content": _answer,"img": os.path.relpath(img_path) if img_path is not None else None})
    if img_path is not  None:
        _chat_bot.append((_question, [os.path.relpath(img_path).__str__(),""]))
        _chat_bot.append((None, _answer))
    else:
        _chat_bot.append((_question, _answer))

    _app_cfg['ctx'] = _context
    _app_cfg['sts'] = 200
    _app_cfg['img'] = os.path.relpath(img_path) if img_path is not None else None
    print('<Assistant>:', _answer)
    message["text"] = ""
    message['files'] = None
    return gr.MultimodalTextbox(interactive=True, file_types=["image"],
                                      placeholder="输入消息或者上传图片", show_label=False), _chat_bot, _app_cfg

def regenerate_button_clicked( _chat_bot, _app_cfg):
    message, app_session,_chat_bot = respond(_app_cfg, _chat_bot)
    return _chat_bot, app_session

def delete_know_base(_app_cfg, _chat_bot):
    _app_cfg['ret']["client"].close()
    shutil.rmtree("storage/decompress")
    shutil.rmtree("qdrant_mm_db/collection")

    _chat_bot.append(('', '知识库删除成功。'))
    return _app_cfg, _chat_bot

def create_component(params):
    return gr.Button(
        value=params['value'],
        interactive=True

    )

with gr.Blocks() as funclip_service:
    with gr.Column():
        with gr.Row():
            bt_file = gr.File(label="上传知识库 .zip 格式文件")
            konwbase = create_component({'value': '向量化'})
            rmkonwbase = create_component({'value': '删除知识库'})

        with gr.Column():
            prompt = gr.Textbox(label="prompt",
                                value="根据上下和图片给出简短的回答。文本中分号后面括号里为图片坐标，一个二维坐标(x,y)，表示图中主要物体相对中心点的水平坐标，供参考。",
                                interactive=True)
            chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"],
                                      placeholder="输入消息或者上传图片", show_label=False)

    with gr.Column():
        with gr.Row():
            app_session = gr.State({'sts': None, 'ctx': None, 'img': None,'ret':None,'file':None})
            chat_bot = gr.Chatbot(label=f"Chat with {model_name}")

    konwbase.click(
        regenerate_button_clicked,
       [chat_bot, app_session],
        [chat_bot, app_session]
    )
    rmkonwbase.click(
        delete_know_base,
        [app_session, chat_bot],
        [app_session, chat_bot]
    )
    chat_input.submit(
        respond1,
        [chat_input, chat_bot, app_session, prompt],
        [chat_input, chat_bot, app_session]
    )
    bt_file.upload(lambda: None, None, chat_bot, queue=False).then(upload_img,
                                                                  inputs=[bt_file, chat_bot, app_session],
                                                                  outputs=[chat_bot, app_session])

if __name__ == '__main__':
    funclip_service.launch(share=False)
