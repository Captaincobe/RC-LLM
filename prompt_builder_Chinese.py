def build_prompt(features):
    encryption = "使用加密协议 TLS {}".format(features["c_TLSvers"]) if features["c_iscrypto"] == 1 else "未加密通信"
    
    prompt = f"""以下是一条网络流量的统计信息：
- 客户端浏览器：{features['browser']}
- 访问网站：{features['website']}
- 行为类型：{features['behaviour']}
- 客户端发送数据包数：{features['c_pkts_all']}，ACK 包数：{features['c_ack_cnt']}
- 传输数据字节：{features['c_bytes_uniq']}
- HTTP 请求数：{features['http_req_cnt']}，响应数：{features['http_res_cnt']}
- 平均 RTT：{features['c_rtt_avg']} 毫秒
- 是否加密：{encryption}

请用一句自然语言总结该流量的类型、通信模式与潜在风险。"""
    return prompt
