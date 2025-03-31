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
    # "label": row.get("Label", "Benign"),
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
