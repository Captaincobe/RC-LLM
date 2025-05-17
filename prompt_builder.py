import time
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.utils import encode_features, retrieve_similar_flows
print("C")
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0, -1] == stop_id for stop_id in self.stop_ids)

def agent_1(dataset, features):
    if dataset == 'CICIDS':
        prompt = f"""
            As a deep flow analyzer, evaluate the behavior of the current network session using the following metrics:\n
                - Source IP and Port: {features['src_ip']}:{features['src_port']}
                - Destination IP and Port: {features['dst_ip']}:{features['dst_port']}
                - Flow Duration: {features['duration']} microseconds seconds
                - Bytes Sent: {features['src_bytes']}, Received: {features['dst_bytes']}
                - Flow Rate — Bytes/s: {features['flow_bytes_per_sec']}, Packets/s: {features['flow_pkts_per_sec']}
                - Direction Ratio (Fwd/Bwd Packet Length Mean): {features['direction_ratio']}
                - Total Packets — Forward: {features['total_fwd_pkts']}, Backward: {features['total_bwd_pkts']}
                - Total Bytes — Forward: {features['src_bytes']}, Backward: {features['dst_bytes']}
                - Avg. Packet Length — Fwd: {features['fwd_pkt_len_mean']}, Bwd: {features['bwd_pkt_len_mean']}
                - Activity Timing — Active Mean: {features['active_mean']} microsec, Idle Mean: {features['idle_mean']} microsec


                Your analysis should include:\n
                - The type of network traffic and the nature of the protocol used.
                - The directionality and intensity of communication.
                - Any available TLS-related characteristics (if applicable), such as version, cipher suite, or session resumption.
                - Any potential signs of abnormal, suspicious, or malicious behavior.
                - How this traffic aligns with or differs from patterns commonly seen in the dataset.

            Please provide a **clear, one-sentence, and concise analysis** explaining:
            - Your interpretation of the session's behavior.
            - And any subtle elements regarding whether the session may be benign or malicious.
        """
    elif dataset == 'TONIoT':
        prompt = f"""
            As a deep flow analyzer, please be aware of the following prior knowledge:
                    - The dataset includes nine major categories (Scanning, Reconnaissance, DoS, DDoS, Ransomware, Backdoor, Injection, XSS, Password Cracking, MITM) of attacks targeting IoT/IIoT devices and sensors.
                    - Scanning attacks originate from IPs like 192.168.1.30–33 and 192.168.1.38 using tools such as Nmap and Nessus.
                    - DoS/DDoS attacks are launched from 192.168.1.30–31 and 192.168.1.34–39 using Scapy scripts and UFONet.
                    - Ransomware and Backdoor attacks use Metasploitable3, launched from 192.168.1.33 and 192.168.1.37.
                    - Injection and XSS attacks involve Bash scripts and tools like XSSer, targeting DVWA and IoT web interfaces.
                    - Password cracking uses CeWL and Hydra from various IPs such as 192.168.1.30–36, 38.
                    - MITM attacks use Ettercap and originate from 192.168.1.31–34, involving ARP poisoning and port sniffing.\n
            Now, evaluate the behavior of the current network session using the following metrics:\n
                - Source IP and Port: {features['src_ip']}:{features['src_port']}
                - Destination IP and Port: {features['dst_ip']}:{features['dst_port']}
                - Duration: {features['duration']} seconds
                - Bytes Sent: {features['src_bytes']}, Received: {features['dst_bytes']}
                - Packets Sent: {features['src_pkts']}, Received: {features['dst_pkts']}
                - SSL Version: {features['ssl_version']}
                - SSL Cipher: {features['ssl_cipher']}
                - SSL Established: {features['ssl_established']}
                - SSL Resumed: {features['ssl_resumed']}
                - Weird Name: {features['weird_name']}
                - DNS Query: {features['dns_query']}
                - HTTP URI: {features['http_uri']}
                - HTTP User-Agent: {features['http_user_agent']}
                - HTTP Referer: {features['http_referer']}
                
            Your analysis should include:\n
                - The type of network traffic and the nature of the protocol used.
                - The directionality and intensity of communication.
                - Any available TLS-related characteristics (if applicable), such as version, cipher suite, or session resumption.
                - Any potential signs of abnormal, suspicious, or malicious behavior.
                - How this traffic aligns with or differs from patterns commonly seen in the dataset.
                
            Please provide a **clear, one-sentence, and concise analysis** explaining:
            - Your interpretation of the session's behavior.
            - And any subtle elements regarding whether the session may be benign or malicious.
        """ 
    elif dataset == 'DoHBrw':
        prompt = f"""
            As a a deep flow analyzer, please be aware of the following prior knowledge:
                - Benign DoH traffic was generated using browsers such as Firefox and Chrome, accessing top Alexa sites via public DoH servers (e.g., Cloudflare, Google DNS, AdGuard).
                - Malicious DoH traffic was crafted using tunneling tools such as dns2tcp, DNSCat2, and Iodine, which encapsulate other protocols within DNS-over-HTTPS queries.
                - Only tunneled malicious DoH behavior is present in this dataset; C2-only behaviors are excluded.\n
            Now, please evaluate the behavior of the current network session using the following metrics:\n
                - Source IP and Port: {features['src_ip']}:{features['src_port']}
                - Destination IP and Port: {features['dst_ip']}:{features['dst_port']}
                - Duration: {features['duration']} seconds
                - Bytes Sent: {features['src_bytes']}, Received: {features['dst_bytes']}
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

            Please provide a **clear, one-sentence, and concise analysis** explaining:
            - Your interpretation of the session's behavior.
            - And any subtle elements regarding whether the session may be benign or malicious.
        """
    return prompt
def agent_2(dataset, features):
    # 加载特征嵌入和描述
    # out_path = f"datasets/{dataset}/outputs"
    # feature_emb_path = f"{out_path}/feature_embeddings.npy"
    context = summarize_context(features, dataset)
    # context = summarize_context(features, dataset, time_window=300)
    session_desc = f"Session: {features['src_ip']}:{features['src_port']} → {features['dst_ip']}:{features['dst_port']}, " \
                f"duration={features['duration']}s, bytes={features['src_bytes']}/{features['dst_bytes']}, " \
                f"protocol={'TLS' if features.get('ssl_version') else 'Unknown'}, dns={str(features.get('dns_query', ''))[:30]}"

    if context is None:
        print("No similar traffic patterns available yet.")
        prompt = """The contextual information is missing or insufficient, please respond with:
                Observation: Insufficient context for behavior analysis.
                Interpretation: No peer behavior patterns available for comparative reasoning."""
    else:
        if dataset == 'TONIoT':
            prompt = f"""
                As a contextual traffic analyst, please be aware of the following prior knowledge:
                    - The dataset includes nine major categories (Scanning, Reconnaissance, DoS, DDoS, Ransomware, Backdoor, Injection, XSS, Password Cracking, MITM) of attacks targeting IoT/IIoT devices and sensors.
                    - Scanning attacks originate from IPs like 192.168.1.30–33 and 192.168.1.38 using tools such as Nmap and Nessus.
                    - DoS/DDoS attacks are launched from 192.168.1.30–31 and 192.168.1.34–39 using Scapy scripts and UFONet.
                    - Ransomware and Backdoor attacks use Metasploitable3, launched from 192.168.1.33 and 192.168.1.37.
                    - Injection and XSS attacks involve Bash scripts and tools like XSSer, targeting DVWA and IoT web interfaces.
                    - Password cracking uses CeWL and Hydra from various IPs such as 192.168.1.30–36, 38.
                    - MITM attacks use Ettercap and originate from 192.168.1.31–34, involving ARP poisoning and port sniffing.\n
                    Current Session: {session_desc}\n
                Now, compare the current session against the following similar sessions from the dataset: {context}\n
                    Focus especially on subtle but meaningful deviations rather than superficial similarities. Consider:\n
                    - Is this session fully consistent with the retrieved examples?\n
                    - Does it deviate in any notable way (e.g., timing regularity, packet structure, flow rates)?\n
                    - Could its similarity be intentional to evade detection (e.g., protocol mimicry)?\n
                    - Could the deviations suggest automation, beaconing, covert tunneling, or other stealthy malicious behavior?

                Please provide a **clear, one-sentence, and concise analysis** explaining:
                - Your comparison with the context.
                - And any subtle elements regarding whether the session may be benign or malicious.\n
            """
        elif dataset == 'DoHBrw':
            prompt = f"""
                As a contextual traffic analyst, please be aware of the following prior knowledge:
                - Benign DoH traffic was generated using browsers such as Firefox and Chrome, accessing top Alexa sites via public DoH servers (e.g., Cloudflare, Google DNS, AdGuard).
                - Malicious DoH traffic was crafted using tunneling tools such as dns2tcp, DNSCat2, and Iodine, which encapsulate other protocols within DNS-over-HTTPS queries.
                - Only tunneled malicious DoH behavior is present in this dataset; C2-only behaviors are excluded.\n
                Current Session: {session_desc}\n
                Now, compare the current session against the following similar sessions from the dataset: {context}\n
                    Focus especially on subtle but meaningful deviations rather than superficial similarities. Consider:\n
                    - Is this session fully consistent with the retrieved examples?\n
                    - Does it deviate in any notable way (e.g., timing regularity, packet structure, flow rates)?\n
                    - Could its similarity be intentional to evade detection (e.g., protocol mimicry)?\n
                    - Could the deviations suggest automation, beaconing, covert tunneling, or other stealthy malicious behavior?

                Please provide a **clear, one-sentence, and concise analysis** explaining:
                - Your comparison with the context.
                - And any subtle elements regarding whether the session may be benign or malicious.\n
            """
        else:
            prompt = f"""
                Current Session:b{session_desc}\n
                As a contextual traffic analyst, compare the current session against the following similar sessions from the dataset:
                    {context}\n
                    Focus especially on subtle but meaningful deviations rather than superficial similarities. Consider:\n
                    - Is this session fully consistent with the retrieved examples?\n
                    - Does it deviate in any notable way (e.g., timing regularity, packet structure, flow rates)?\n
                    - Could its similarity be intentional to evade detection (e.g., protocol mimicry)?\n
                    - Could the deviations suggest automation, beaconing, covert tunneling, or other stealthy malicious behavior?

                Please provide a **clear, one-sentence, and concise analysis** explaining:
                - Your comparison with the context.
                - And any subtle elements regarding whether the session may be benign or malicious.\n
            """
    return prompt

def chat(messages, model, tokenizer, max_tokens=512):
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
        start = time.time()
        
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            response = chat([
                {"role": "system", "content": "You are an expert in encrypted traffic analysis. You detect anomalies and identify potentially malicious behavior in network sessions, even when the data appears superficially benign."},
                {"role": "user", "content": f"{prompt}"}
            ], model, tokenizer)
        else:
            # 为不支持聊天模板的模型添加一个替代方案
            # 比如直接使用 model.generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, max_new_tokens=256)
            response = tokenizer.decode(output[0], skip_special_tokens=True)

        end = time.time()
        print(f"Generation time: {end - start:.2f} seconds")
        return response

    except Exception as e:
        print(f"LLM call failed: {e}")
        return "Description generation failed"


# Please generate a concise description based on the following features: {prompt}





# 全局变量缓存数据集
_dataset_cache = {}

def summarize_context(feature, dataset_name, max_samples=10):
    """基于特征嵌入检索相似连接，生成上下文描述（字段缺失自动忽略）"""
    global _dataset_cache

    out_path = f"datasets/{dataset_name}"
    data_path = f"{out_path}/text_data.csv"
    emb_path = f"{out_path}/outputs/feature_embeddings.npy"

    try:
        if dataset_name not in _dataset_cache:
            if not os.path.exists(data_path) or not os.path.exists(emb_path):
                print(f"缺失历史数据或嵌入文件: {data_path} / {emb_path}")
                return "- No historical data available for analysis"

            print(f"首次加载上下文缓存: {dataset_name}")
            df = pd.read_csv(data_path)
            df.fillna('', inplace=True)
            embedding_matrix = np.load(emb_path)
            _dataset_cache[dataset_name] = (df, embedding_matrix)
        else:
            df, embedding_matrix = _dataset_cache[dataset_name]

        # 当前流量的嵌入
        current_embedding = encode_features(feature)
        sims = cosine_similarity([current_embedding], embedding_matrix)[0]
        top_k_indices = np.argsort(sims)[-max_samples:][::-1]
        context_rows = df.iloc[top_k_indices]

        result_lines = []
        for idx, row in context_rows.iterrows():
            parts = []

            if 'src_ip' in row and 'src_port' in row and 'dst_ip' in row and 'dst_port' in row:
                parts.append(f"Session {idx}: src={row['src_ip']}:{row['src_port']} → dst={row['dst_ip']}:{row['dst_port']}")

            if 'duration' in row:
                parts.append(f"duration={row['duration']}s")
            if 'src_bytes' in row and 'dst_bytes' in row:
                parts.append(f"bytes={row['src_bytes']}/{row['dst_bytes']}")
            if 'ssl_version' in row and pd.notna(row['ssl_version']):
                parts.append("protocol=TLS")
            elif 'PacketLengthMean' in row and pd.notna(row['PacketLengthMean']):
                parts.append(f"avg_pkt_len={row['PacketLengthMean']}")
            if 'dns_query' in row and pd.notna(row['dns_query']):
                parts.append(f"dns={str(row['dns_query'])[:30]}")

            result_lines.append(", ".join(parts))

        return "\n".join(result_lines)

    except Exception as e:
        print(f"Error in summarize_context: {str(e)}")
        return "- Error analyzing traffic context\n- Proceeding with limited historical data"

# def summarize_context(feature, dataset_name, max_samples=10):
#     """基于特征嵌入检索相似连接，生成上下文描述"""
#     global _dataset_cache

#     out_path = f"datasets/{dataset_name}"
#     data_path = f"{out_path}/text_data.csv"
#     emb_path = f"{out_path}/outputs/feature_embeddings.npy"

#     try:
#         if dataset_name not in _dataset_cache:
#             if not os.path.exists(data_path) or not os.path.exists(emb_path):
#                 print(f"缺失历史数据或嵌入文件: {data_path} / {emb_path}")
#                 return "- No historical data available for analysis"

#             print(f"首次加载上下文缓存: {dataset_name}")
#             df = pd.read_csv(data_path)
#             df.fillna('', inplace=True)
#             embedding_matrix = np.load(emb_path)
#             _dataset_cache[dataset_name] = (df, embedding_matrix)
#         else:
#             df, embedding_matrix = _dataset_cache[dataset_name]

#         # 当前流量的嵌入
#         current_embedding = encode_features(feature)
#         sims = cosine_similarity([current_embedding], embedding_matrix)[0]
#         top_k_indices = np.argsort(sims)[-max_samples:][::-1]
#         context_rows = df.iloc[top_k_indices]

#         result_lines = []
#         for idx, row in context_rows.iterrows():
#             line = f"Session {idx}: src={row['src_ip']}:{row['src_port']} to dst={row['dst_ip']}:{row['dst_port']}, " \
#                    f"duration={row['duration']}s, bytes={row['src_bytes']}/{row['dst_bytes']}, " \
#                    f"protocol={'TLS' if row.get('ssl_version') else 'Unknown'}, dns={str(row.get('dns_query', ''))[:30]}"
#             result_lines.append(line)

#         return "\n".join(result_lines)

#     except Exception as e:
#         print(f"Error in summarize_context: {str(e)}")
#         return "- Error analyzing traffic context\n- Proceeding with limited historical data"


