import http.client
import json  # 用于确保 JSON 格式化正确

def deepseek_r1(p):
    conn = http.client.HTTPSConnection("cloud.infini-ai.com")

    # 构造 payload，将 p 替换到 content 中
    payload_dict = {
        "model": "deepseek-r1",
        "messages": [
            {
                "role": "user",
                "content": p
            }
        ]
    }
    # 使用 json.dumps 确保 payload 格式正确
    payload = json.dumps(payload_dict)

    headers = {
        'Content-Type': "application/json",
        'Authorization': "Bearer sk-dasb57xneg7br2fz"
    }

    conn.request("POST", "/maas/v1/chat/completions", payload, headers)

    res = conn.getresponse()
    data = res.read()

    print(data.decode("utf-8"))

def get_message(good_templates_str, bad_templates_str):
    p = f"""
Hi ChatGPT, I have two lists of templates: one with good templates and the other with bad templates. There are characteristics that make a template good or bad. Based on these characteristics, give me better templates. 
Here is the list of good templates:
{good_templates_str}

Here is the list of bad templates: 
{bad_templates_str}

Here are my requirements:
- Please only reply the template.
- The template should be less than 15 words.
- The template should have similar structure to the above template.
- Only the template should start with '- ' in a separate line.    
    """
    return p

if __name__ == '__main__':
    good_prompts = [
        "A cat storms through grass, frenzy fading to gentle, swaying stillness.",
        "A cat erupts in wild bounds, settling to subtle grass tremors."
    ]

    bad_prompts = [
        "A cat blurs through chaos, energy ebbing into calm, faint rustles.",
        "A cat tears across the field, momentum melting into delicate steps."
    ]

    good_templates_str = '\n'.join(good_prompts)
    bad_templates_str = '\n'.join(bad_prompts)

    # 调用 get_message 生成内容
    p = get_message(good_templates_str, bad_templates_str)
    print(p)
    # 调用 deepseek_r1 将 p 作为 content 发出请求
    #deepseek_r1(p)
