import time
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.utils import encode_features, retrieve_similar_flows
from datetime import datetime

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0, -1] == stop_id for stop_id in self.stop_ids)

def agent_1(dataset, features):
    prompt = f"""
        As a deep flow analyzer, evaluate the behavior of the current network session using the following metrics:\n
            - Source IP and Port: {features['src_ip']}:{features['src_port']}
            - Destination IP and Port: {features['dst_ip']}:{features['dst_port']}
            - Duration: {features['Duration']} seconds
            - Bytes Sent: {features['FlowBytesSent']}, Received: {features['FlowBytesReceived']}
            - Flow Rates - Sent: {features['FlowSentRate']} Bps, Received: {features['FlowReceivedRate']} Bps
            - Avg. Packet Length: {features['PacketLengthMean']}, Std Dev: {features['PacketLengthStandardDeviation']}
            - Packet Timing - Mean Interval: {features['PacketTimeMean']} s, Std Dev: {features['PacketTimeStandardDeviation']}
            - Response Time Mean: {features['ResponseTimeTimeMean']} s

            Your analysis should include:\n
            - The type of network traffic and the nature of the protocol used.
            - The directionality and intensity of communication.
            - Any available TLS-related characteristics (if applicable), such as version, cipher suite, or session resumption.
            - Any potential signs of abnormal, suspicious, or malicious behavior.
            - How this traffic aligns with or differs from patterns commonly seen in the dataset.

        Please provide a **clear, concise analysis** explaining:
        - Your interpretation of the session's behavior.
        - And any subtle elements regarding whether the session may be benign or malicious.
    """
    return prompt
def agent_2(dataset, features):
    # 加载特征嵌入和描述
    out_path = f"datasets/{dataset}/outputs"
    feature_emb_path = f"{out_path}/feature_embeddings.npy"
    desc_path = f"{out_path}/descriptions-concise-pro.csv"
    
    # 生成当前特征的嵌入
    current_embedding = encode_features(features)
    
    if os.path.exists(feature_emb_path) and os.path.exists(desc_path):
        feature_embeddings = np.load(feature_emb_path)
        descriptions = pd.read_csv(desc_path)['description'].tolist()
        
        # 检索相似的流量
        similar_descriptions, similar_scores = retrieve_similar_flows(current_embedding, feature_embeddings, descriptions)
        
        # 构建相似流量信息
        similar_flows_info = "\nSimilar traffic patterns found:\n"
        for i, (desc, score) in enumerate(zip(similar_descriptions, similar_scores), 1):
            similar_flows_info += f"{i}. Similarity: {score:.2f} - {desc}\n"
            print(f"similar_flows_info: {similar_flows_info}")
    else:
        similar_flows_info = "\nNo similar traffic patterns available yet."
        prompt = "[Insufficient contextual flows available for analysis]"
        return prompt
    prompt = f"""
        As a contextual traffic analyst, compare the current session against the following similar sessions from the dataset:
            {similar_flows_info}\n
            Focus especially on subtle but meaningful deviations rather than superficial similarities. Consider:\n
            - Is this session fully consistent with the retrieved examples?\n
            - Does it deviate in any notable way (e.g., timing regularity, packet structure, flow rates)?\n
            - Could its similarity be intentional to evade detection (e.g., protocol mimicry)?\n
            - Could the deviations suggest automation, beaconing, covert tunneling, or other stealthy malicious behavior?

        Please provide a **clear, concise analysis** explaining:
        - Your comparison with the context.
        - And any subtle elements regarding whether the session may be benign or malicious.\n
    """
    return prompt

def build_prompt(dataset, features):
    # 加载特征嵌入和描述
    out_path = f"datasets/{dataset}/outputs"
    feature_emb_path = f"{out_path}/feature_embeddings.npy"
    desc_path = f"{out_path}/descriptions-concise-pro.csv"
    
    # 生成当前特征的嵌入
    current_embedding = encode_features(features)
    
    if os.path.exists(feature_emb_path) and os.path.exists(desc_path):
        feature_embeddings = np.load(feature_emb_path)
        descriptions = pd.read_csv(desc_path)['description'].tolist()
        
        # 检索相似的流量
        similar_descriptions, similar_scores = retrieve_similar_flows(current_embedding, feature_embeddings, descriptions)
        
        # 构建相似流量信息
        similar_flows_info = "\nSimilar traffic patterns found:\n"
        for i, (desc, score) in enumerate(zip(similar_descriptions, similar_scores), 1):
            similar_flows_info += f"{i}. Similarity: {score:.2f} - {desc}\n"
    else:
        similar_flows_info = "\nNo similar traffic patterns available yet."

    if dataset == 'CICIDS':
        prompt = f"""
        Follow these steps:\n
        1. As a deep flow analyzer, evaluate the behavior of the current network session using the following metrics:\n
            - Source: {features['src_ip']}:{features['src_port']}
            - Destination: {features['dst_ip']}:{features['dst_port']}
            - Protocol: {features['protocol']}
            - Flow Duration: {features["flow_duration"]} μs
            - Total Packets — Forward: {features["total_fwd_pkts"]}, Backward: {features["total_bwd_pkts"]}
            - Avg. Packet Length — Fwd: {features["fwd_pkt_len_mean"]}, Bwd: {features["bwd_pkt_len_mean"]}
            - Direction Ratio (Fwd/Bwd Packet Length Mean): {features["direction_ratio"]}
            - Flow Rate — Packets/s: {features["flow_pkts_per_sec"]}, Bytes/s: {features["flow_bytes_per_sec"]}
            - Active Period Mean: {features["active_mean"]} μs, Idle Period Mean: {features["idle_mean"]} μs

            Your analysis should include:\n
            - The type of network traffic and the nature of the protocol used.
            - The directionality and intensity of communication.
            - Any available TLS-related characteristics (if applicable), such as version, cipher suite, or session resumption.
            - Any potential signs of abnormal, suspicious, or malicious behavior.
            - How this traffic aligns with or differs from patterns commonly seen in the dataset.

        2. As a contextual traffic analyst, compare the current session against the following similar sessions from the dataset:
            {similar_flows_info}\n
            Focus especially on subtle but meaningful deviations rather than superficial similarities. Consider:\n
            - Is this session fully consistent with the retrieved examples?\n
            - Does it deviate in any notable way (e.g., timing regularity, packet structure, flow rates)?\n
            - Could its similarity be intentional to evade detection (e.g., protocol mimicry)?\n
            - Could the deviations suggest automation, beaconing, covert tunneling, or other stealthy malicious behavior?

        Please provide a **clear, concise analysis** explaining:
        - Your interpretation of the session's behavior.
        - Your comparison with the context.
        - And any subtle elements regarding whether the session may be benign or malicious.
        """
        # Please summarize this session in **one concise and informative sentence**, addressing the following:
        # 1. The type of network traffic and the nature of the protocol used.
        # 2. The directionality and intensity of communication.
        # 3. Any potential signs of abnormal or malicious behavior based on the observed metrics.

    elif dataset == 'TONIoT':
        prompt = f"""The following is a summary of an encrypted network session:
        - Timestamp: {features['Timestamp']}
        - Source IP and Port: {features['src_ip']}:{features['src_port']}
        - Destination IP and Port: {features['dst_ip']}:{features['dst_port']}
        - Protocol: {features['protocol']}
        - Connection State: {features['conn_state']}
        - Duration: {features['duration']} seconds
        - Bytes Sent: {features['src_bytes']}, Received: {features['dst_bytes']}
        - Packets Sent: {features['src_pkts']}, Received: {features['dst_pkts']}
        - SSL Version: {features['ssl_version']}
        - SSL Cipher: {features['ssl_cipher']}
        - SSL Established: {features['ssl_established']}
        - SSL Resumed: {features['ssl_resumed']}
        - SSL Subject: {features['ssl_subject']}
        - SSL Issuer: {features['ssl_issuer']}
        - Any TLS anomalies (e.g., weird events): {features['weird_name']}
        {similar_flows_info}

        Please summarize this session in **one concise and informative sentence**, addressing the following:
        1. The type of network traffic and the nature of the protocol used.
        2. The directionality and intensity of communication.
        3. Characteristics of the TLS handshake (e.g., version, cipher, session resumption).
        4. Any potential signs of abnormal or malicious behavior based on the observed metrics.
        5. How this traffic compares to similar patterns found in the dataset.
        """
    elif dataset == 'DoHBrw':
        prompt = f"""The following is a summary of a DoH (DNS over HTTPS) network session:
        - Source IP and Port: {features['src_ip']}:{features['src_port']}
        - Destination IP and Port: {features['dst_ip']}:{features['dst_port']}
        - Duration: {features['Duration']} seconds
        - Bytes Sent: {features['FlowBytesSent']}, Received: {features['FlowBytesReceived']}
        - Flow Rates - Sent: {features['FlowSentRate']} Bps, Received: {features['FlowReceivedRate']} Bps
        - Avg. Packet Length: {features['PacketLengthMean']}, Std Dev: {features['PacketLengthStandardDeviation']}
        - Packet Timing - Mean Interval: {features['PacketTimeMean']} s, Std Dev: {features['PacketTimeStandardDeviation']}
        - Response Time Mean: {features['ResponseTimeTimeMean']} s
        {similar_flows_info}

        Based on this information and the similar traffic patterns observed, generate a **concise one-sentence description** that captures:
        1. The nature of the traffic flow (e.g., interactive, bursty, idle).
        2. Communication intensity and direction.
        3. Any observable patterns that may imply typical or anomalous behavior.
        4. How this traffic compares to similar patterns found in the dataset."""
    else:
        encryption = "Encrypted with TLS {}".format(features["c_TLSvers"]) if features["c_iscrypto"] == 1 else "Unencrypted communication"
        prompt = f"""Below is a summary of a network traffic session:
        - Browser: {features['browser']}
        - Website accessed: {features['website']}
        - Behavior type: {features['behaviour']}
        - Client packets sent: {features['c_pkts_all']}, ACK packets: {features['c_ack_cnt']}
        - Unique bytes transferred: {features['c_bytes_uniq']}
        - HTTP requests: {features['http_req_cnt']}, responses: {features['http_res_cnt']}
        - Avg RTT: {features['c_rtt_avg']} ms
        - Encryption: {encryption}
        {similar_flows_info}

        Summarize this traffic in one sentence, including type, communication pattern, any potential risks, and how it compares to similar patterns found in the dataset."""

    return prompt
        # - Missed bytes: {features['missed_bytes']}
        # - Service: {features['service']}

def build_prompt_protocol(dataset, features):
    # 加载特征嵌入和描述
    out_path = f"datasets/{dataset}/outputs"
    feature_emb_path = f"{out_path}/feature_embeddings.npy"
    desc_path = f"{out_path}/descriptions-concise-pro.csv"
    
    # 生成当前特征的嵌入
    current_embedding = encode_features(features)
    
    if os.path.exists(feature_emb_path) and os.path.exists(desc_path):
        feature_embeddings = np.load(feature_emb_path)
        descriptions = pd.read_csv(desc_path)['description'].tolist()
        
        # 检索相似的流量
        similar_descriptions, similar_scores = retrieve_similar_flows(current_embedding, feature_embeddings, descriptions)
        
        # 构建相似流量信息
        similar_flows_info = "\nSimilar traffic patterns found:\n"
        for i, (desc, score) in enumerate(zip(similar_descriptions, similar_scores), 1):
            similar_flows_info += f"{i}. Similarity: {score:.2f} - {desc}\n"
    else:
        similar_flows_info = "\nNo similar traffic patterns available yet."

    if dataset == 'TONIoT':
        prompt = f"""The following is a summary of an encrypted network session:
        - Timestamp: {features['Timestamp']}
        - Source IP and Port: {features['src_ip']}:{features['src_port']}
        - Destination IP and Port: {features['dst_ip']}:{features['dst_port']}
        - Protocol: {features['protocol']}
        - Connection State: {features['conn_state']}
        - Duration: {features['duration']} seconds
        - Bytes Sent: {features['src_bytes']}, Received: {features['dst_bytes']}
        - Packets Sent: {features['src_pkts']}, Received: {features['dst_pkts']}
        - SSL Version: {features['ssl_version']}
        - SSL Cipher: {features['ssl_cipher']}
        - SSL Established: {features['ssl_established']}
        - SSL Resumed: {features['ssl_resumed']}
        - SSL Subject: {features['ssl_subject']}
        - SSL Issuer: {features['ssl_issuer']}
        - Any TLS anomalies (e.g., weird events): {features['weird_name']}
        {similar_flows_info}

        Based on this information and the similar traffic patterns observed, please provide a short judgment on whether this session appears consistent with legitimate DoH usage, or shows signs of suspicious/malicious behavior:
        1. Determine whether the packet timing and flow rates are typical of interactive DNS queries.
        2. Assess whether the payload sizes and intervals suggest periodic or automated traffic patterns.
        3. Comment on whether the behavior indicates misuse or obfuscation of DoH traffic.
        4. Compare this traffic with the similar patterns found and explain any significant differences or similarities.

        """
    elif dataset == 'DoHBrw':
        prompt = f"""The following is a summary of a DoH (DNS over HTTPS) network session:
        - Source IP and Port: {features['src_ip']}:{features['src_port']}
        - Destination IP and Port: {features['dst_ip']}:{features['dst_port']}
        - Duration: {features['Duration']} seconds
        - Bytes Sent: {features['FlowBytesSent']}, Received: {features['FlowBytesReceived']}
        - Flow Rates - Sent: {features['FlowSentRate']} Bps, Received: {features['FlowReceivedRate']} Bps
        - Avg. Packet Length: {features['PacketLengthMean']}, Std Dev: {features['PacketLengthStandardDeviation']}
        - Packet Timing - Mean Interval: {features['PacketTimeMean']} s, Std Dev: {features['PacketTimeStandardDeviation']}
        - Response Time Mean: {features['ResponseTimeTimeMean']} s
        {similar_flows_info}

        Based on this information and the similar traffic patterns observed, generate a **concise one-sentence description** that captures:
        1. The nature of the traffic flow (e.g., interactive, bursty, idle).
        2. Communication intensity and direction.
        3. Any observable patterns that may imply typical or anomalous behavior.
        4. How this traffic compares to similar patterns found in the dataset."""
    else:
        encryption = "Encrypted with TLS {}".format(features["c_TLSvers"]) if features["c_iscrypto"] == 1 else "Unencrypted communication"
        prompt = f"""Below is a summary of a network traffic session:
        - Browser: {features['browser']}
        - Website accessed: {features['website']}
        - Behavior type: {features['behaviour']}
        - Client packets sent: {features['c_pkts_all']}, ACK packets: {features['c_ack_cnt']}
        - Unique bytes transferred: {features['c_bytes_uniq']}
        - HTTP requests: {features['http_req_cnt']}, responses: {features['http_res_cnt']}
        - Avg RTT: {features['c_rtt_avg']} ms
        - Encryption: {encryption}
        {similar_flows_info}

        Summarize this traffic in one sentence, including type, communication pattern, and any potential risks. Compare this traffic with similar patterns found in the dataset and highlight any significant differences or similarities."""

    return prompt


def chat(messages, model, tokenizer, max_tokens=256):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []

    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_token_ids)]),
        do_sample=True,     # 开启随机采样
        top_p=0.95,         # 模型会从所有可能的下一个token中，选出累计概率前95%的那部分词汇作为候选，再在其中采样。
        temperature=0.7     # 值越高 → 越随机、多样性强但可能语义不稳
    )

    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response


def generate_description(model, tokenizer, prompt):
    # torch.cuda.empty_cache()
    try:
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            start = time.time() 
            response = chat([
                {"role": "system", "content": "You are an expert in encrypted traffic analysis. You detect anomalies and identify potentially malicious behavior in network sessions, even when the data appears superficially benign."},
                {"role": "user", "content": f"{prompt}"}
            ], model, tokenizer)

        end = time.time()
        print(f"Generation time: {end - start:.2f} seconds")
        return response

    except Exception as e:
        print(f"LLM call failed: {e}")
        return "Description generation failed"


# Please generate a concise description based on the following features: {prompt}


def summarize_connection(row):
    return {
        "src_ip": row["src_ip"],
        "dst_ip": row["dst_ip"],
        "dst_port": int(row["dst_port"]),
        "duration": float(row["duration"]),
        "src_bytes": int(row["src_bytes"]),
        "dst_bytes": int(row["dst_bytes"]),
        "conn_state": row["conn_state"]
    }

def summarize_context(df, current_row, time_window=300):
    ts = float(current_row["timestamp"])
    src_ip = current_row["src_ip"]

    # 过滤时间窗口内同源 IP 的连接
    context_df = df[(df["src_ip"] == src_ip) &
                    (df["timestamp"] >= ts - time_window) &
                    (df["timestamp"] < ts)]

    if context_df.empty:
        return "No previous flows found for this IP in the time window."

    port_list = context_df["dst_port"].astype(int).tolist()
    short_conns = context_df[context_df["duration"] < 0.1]
    conn_states = context_df["conn_state"].value_counts().to_dict()

    return (
        f"- Src IP made {len(context_df)} connections in past {time_window // 60} minutes.\n"
        f"- Target ports: {sorted(set(port_list))}\n"
        f"- {len(short_conns)} connections lasted <0.1s\n"
        f"- Conn_state distribution: {conn_states}\n"
        f"- DNS query fields: {'Yes' if context_df['dns_query'].astype(str).str.len().sum() > 0 else 'No'}\n"
        f"- HTTP data present: {'Yes' if context_df['http_uri'].astype(str).str.len().sum() > 0 else 'No'}"
    )


