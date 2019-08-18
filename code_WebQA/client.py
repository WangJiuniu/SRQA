import requests
import json

input_json = {
    "action": "mrc",
    "data": [
        {
            "question_id": "1",
            "question": "中国商代最后一个君王是谁？",
            "passages": [
                {
                    "passage_id": "1",
                    "text": "商朝最后一个皇帝是商纣王，因为宠幸苏妲己而亡国。"
                },
                {
                    "passage_id": "2",
                    "text": "我觉得是商纣王"
                }
            ]
        },
        {
            "question_id": "2",
            "question": "国务委员兼外交部长王毅就什么问题作出回应？",
            "passages": [
                {
                    "passage_id": "1",
                    "text": "据外交部网站8月2日消息，当地时间2019年8月2日，国务委员兼外交部长王毅在泰国曼谷出席东亚合作系列外长会期间，就个别国家在南海问题上无端指责中国作出回应。"
                },
                {
                    "passage_id": "2",
                    "text": "王毅就个别国家在南海问题上无端指责中国作出回应"
                }
            ]
        }
    ]
}

output_json = requests.post("http://127.0.0.1:5001/mrc", data=json.dumps(input_json))
output_json = json.loads(output_json.text)

print(output_json)
