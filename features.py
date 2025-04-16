import pandas as pd



def extract_key_features(dataset_name, row):
    if dataset_name == "CICIDS":
        return {
            "src_ip": row["src_ip"],
            "src_port": row["src_port"],
            "dst_ip": row["dst_ip"],
            "dst_port": row["dst_port"],
            "protocol": row["Protocol"],  # 注意大小写！

            "direction_ratio": round(
                row["Fwd Packet Length Mean"] / (row["Bwd Packet Length Mean"] + 1e-5), 4
            ),
            "flow_duration": round(row["Flow Duration"], 4),
            "total_fwd_pkts": row["Total Fwd Packets"],
            "total_bwd_pkts": row["Total Backward Packets"],
            "fwd_pkt_len_mean": round(row["Fwd Packet Length Mean"], 4),
            "bwd_pkt_len_mean": round(row["Bwd Packet Length Mean"], 4),
            "flow_pkts_per_sec": round(row["Flow Packets/s"], 4),
            "flow_bytes_per_sec": round(row["Flow Bytes/s"], 4),
            "active_mean": round(row["Active Mean"], 4),
            "idle_mean": round(row["Idle Mean"], 4),
        }
    elif dataset_name == "TONIoT":
        return {
            "Timestamp": row["Timestamp"],
            "src_ip": row["src_ip"],
            "src_port": row["src_port"],
            "dst_ip": row["dst_ip"],
            "dst_port": row["dst_port"],
            "protocol": row["proto"], 
            "conn_state": row["conn_state"],
            "duration": round(row["duration"], 4),
            "src_bytes": round(row["src_bytes"], 4),
            "dst_bytes": round(row["dst_bytes"], 4),
            "src_pkts": round(row["src_pkts"], 4),
            "dst_pkts": round(row["dst_pkts"], 4),
            "ssl_version": row["ssl_version"],
            "ssl_cipher": row["ssl_cipher"],
            "ssl_established": row["ssl_established"],
            "ssl_resumed": row["ssl_resumed"],
            "ssl_subject": row["ssl_subject"],
            "ssl_issuer": row["ssl_issuer"],
            "weird_name": row["weird_name"],
        }
    elif dataset_name == "DoHBrw":
        return {
            "src_ip": row["src_ip"],
            "src_port": row["src_port"],
            "dst_ip": row["dst_ip"],
            "dst_port": row["dst_port"],
            "Duration": round(row["Duration"], 4),
            "FlowBytesSent": round(row["FlowBytesSent"], 4),
            "FlowBytesReceived": round(row["FlowBytesReceived"], 4),
            "FlowSentRate": round(row["FlowSentRate"], 4),
            "FlowReceivedRate": round(row["FlowReceivedRate"], 4),
            "PacketLengthMean": round(row["PacketLengthMean"], 4),
            "PacketLengthStandardDeviation": round(row["PacketLengthStandardDeviation"], 4),
            "PacketTimeMean": round(row["PacketTimeMean"], 4),
            "PacketTimeStandardDeviation": round(row["PacketTimeStandardDeviation"], 4),
            "ResponseTimeTimeMean": round(row["ResponseTimeTimeMean"], 4),
        }

