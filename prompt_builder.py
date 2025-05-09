import time
import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.utils import encode_features, retrieve_similar_flows
print("C")
from datetime import datetime



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
            As a deep flow analyzer, evaluate the behavior of the current network session using the following metrics:\n
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
            As a deep flow analyzer, evaluate the behavior of the current network session using the following metrics:\n
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

    # 生成当前特征的嵌入
    current_embedding = encode_features(features)
    
    if context is None:
        print("No similar traffic patterns available yet.")
        prompt = """The contextual information is missing or insufficient, please respond with:
                Observation: Insufficient context for behavior analysis.
                Interpretation: No peer behavior patterns available for comparative reasoning."""
    else:
        prompt = f"""
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
    """作用: 分析同一源IP和目标IP之间的历史网络流量模式"""
    global _dataset_cache
    
    # 数据集路径
    out_path = f"datasets/{dataset_name}"
    data_path = f"{out_path}/text_data.csv"
    
    try:
        # 1. 使用缓存避免重复加载数据集
        if dataset_name not in _dataset_cache:
            if not os.path.exists(data_path):
                print(f"数据文件不存在: {data_path}")
                return f"- No historical data available for analysis"
                
            print(f"首次加载数据集: {dataset_name}")
            df = pd.read_csv(data_path)
            df.fillna('', inplace=True)
            _dataset_cache[dataset_name] = df
        else:
            df = _dataset_cache[dataset_name]
        
        # 2. 直接提取关键信息，避免复杂处理
        src_ip = feature["src_ip"]
        dst_ip = feature["dst_ip"]
        
        # 3. 快速筛选 - 只获取需要的列
        needed_columns = ["src_ip", "dst_ip", "dst_port", "duration", 
                         "src_bytes", "dst_bytes", "ssl_version", 
                         "dns_query", "http_uri"]
        
        existing_columns = [col for col in needed_columns if col in df.columns]
        filtered_df = df[existing_columns]
        
        # 4. 简化匹配逻辑
        # 先尝试快速查找完全匹配的连接对
        same_pair = (filtered_df["src_ip"] == src_ip) & (filtered_df["dst_ip"] == dst_ip)
        context_df = filtered_df[same_pair].head(max_samples)
        
        # 如果不足3个结果，再加入源IP匹配的记录
        if len(context_df) < 3:
            context_df = filtered_df[filtered_df["src_ip"] == src_ip].head(max_samples)
        
        # 5. 如果仍然没找到，返回固定消息避免额外处理
        if context_df.empty:
            return f"- No similar connection patterns found in dataset\n- First observed communication for this IP pair"
        
        # 6. 快速提取统计信息，避免复杂计算
        result_lines = []
        result_lines.append(f"- Found {len(context_df)} similar connections")
        
        # 7. 只处理确定存在的列，避免错误
        if "dst_port" in context_df.columns:
            ports = context_df["dst_port"].astype(str).value_counts().head(3)
            port_str = ", ".join([f"{p}({c})" for p, c in ports.items()])
            result_lines.append(f"- Popular destination ports: {port_str}")
        
        if "duration" in context_df.columns:
            try:
                short_count = (pd.to_numeric(context_df["duration"], errors='coerce') < 1.0).sum()
                result_lines.append(f"- Short connections (<1s): {short_count}")
            except:
                pass
        
        # 8. 简化协议检测
        for protocol, col in [("DNS", "dns_query"), ("HTTP", "http_uri"), ("SSL/TLS", "ssl_version")]:
            if col in context_df.columns:
                has_proto = "Yes" if context_df[col].astype(str).str.len().sum() > 0 else "No"
                result_lines.append(f"- {protocol} traffic observed: {has_proto}")
        
        return "\n".join(result_lines)
            
    except Exception as e:
        print(f"Error in summarize_context: {str(e)}")
        # 出错时返回固定值，而不是None，避免agent_2中的条件判断
        return "- Error analyzing traffic context\n- Proceeding with limited historical data"