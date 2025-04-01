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
                row["Fwd Packet Length Mean"] / (row["Bwd Packet Length Mean"] + 1e-5), 2
            ),
            "flow_duration": round(row["Flow Duration"], 2),
            "total_fwd_pkts": row["Total Fwd Packets"],
            "total_bwd_pkts": row["Total Backward Packets"],
            "fwd_pkt_len_mean": round(row["Fwd Packet Length Mean"], 2),
            "bwd_pkt_len_mean": round(row["Bwd Packet Length Mean"], 2),
            "flow_pkts_per_sec": round(row["Flow Packets/s"], 2),
            "flow_bytes_per_sec": round(row["Flow Bytes/s"], 2),
            "active_mean": round(row["Active Mean"], 2),
            "idle_mean": round(row["Idle Mean"], 2),
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
            "duration": row["duration"],
            "src_bytes": row["src_bytes"],
            "dst_bytes": row["dst_bytes"],
            "src_pkts": row["src_pkts"],
            "dst_pkts": row["dst_pkts"],
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
            "Duration": row["Duration"],
            "FlowBytesSent": row["FlowBytesSent"],
            "FlowBytesReceived": row["FlowBytesReceived"],
            "FlowSentRate": row["FlowSentRate"],
            "FlowReceivedRate": row["FlowReceivedRate"],
            "PacketLengthMean": row["PacketLengthMean"],
            "PacketLengthStandardDeviation": row["PacketLengthStandardDeviation"],
            "PacketTimeMean": row["PacketTimeMean"],
            "PacketTimeStandardDeviation": row["PacketTimeStandardDeviation"],
            "ResponseTimeTimeMean": row["ResponseTimeTimeMean"],
        }
    else:
        return {
            "browser": row.get("browser", "unkonwn"),
            "website": row.get("website", "unkonwn"),
            "behaviour": row.get("behaviour", "unkonwn"),
            "c_pkts_all": row["c_pkts_all:3"],
            "c_ack_cnt": row["c_ack_cnt:5"],
            "c_bytes_uniq": row["c_bytes_uniq:7"],
            "http_req_cnt": row.get("http_req_cnt:111", 0),
            "http_res_cnt": row.get("http_res_cnt:112", 0),
            "c_TLSvers": row.get("c_TLSvers:132", "unkonwn"),
            "c_iscrypto": row["c_iscrypto:40"],
            "c_rtt_avg": row.get("c_rtt_avg:45", 0)
        }
