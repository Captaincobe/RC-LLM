

        Agent1. As a deep flow analyzer, evaluate the behavior of the current network session using the following metrics:\n
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

        Please provide a **clear, concise analysis** explaining:
        - Your interpretation of the session's behavior.
        - And any subtle elements regarding whether the session may be benign or malicious.

        Agent2. As a contextual traffic analyst, compare the current session against the following similar sessions from the dataset:
            {similar_flows_info}\n
            Focus especially on subtle but meaningful deviations rather than superficial similarities. Consider:\n
            - Is this session fully consistent with the retrieved examples?\n
            - Does it deviate in any notable way (e.g., timing regularity, packet structure, flow rates)?\n
            - Could its similarity be intentional to evade detection (e.g., protocol mimicry)?\n
            - Could the deviations suggest automation, beaconing, covert tunneling, or other stealthy malicious behavior?

        Please provide a **clear, concise analysis** explaining:
        - Your comparison with the context.
        - And any subtle elements regarding whether the session may be benign or malicious.
        
## TONIoT
### prompt 1
    Based on this information, please evaluate:
    1. Whether the TLS handshake parameters appear consistent with legitimate client/server behavior.
    2. If the cipher suite and TLS version are commonly seen in benign web traffic.
    3. Whether there is anything suspicious about the certificate (subject/issuer mismatch, self-signed, etc.)
    4. Whether any anomalies suggest potential misuse or obfuscation.

    Please provide a short assessment explaining whether this session likely conforms to expected TLS behavior or not.

### prompt 2
    Your task:
    1. Determine whether the packet timing and flow rates are typical of interactive DNS queries.
    2. Assess whether the payload sizes and intervals suggest periodic or automated traffic patterns.
    3. Comment on whether the behavior indicates misuse or obfuscation of DoH traffic.

    Please provide a short judgment on whether this session appears consistent with legitimate DoH usage, or shows signs of suspicious/malicious behavior.

    Here are several previous network sessions that are most similar to the current one based on statistical features.

    Please analyze how the current session compares to them:
    - Is it fully consistent with their patterns?
    - Does it deviate in some meaningful way (timing, flow rate, packet structure)?
    - Could it indicate malicious intent despite superficial similarity?

    Explain your reasoning.

### prompt 3
    Given the following summary of an encrypted communication session, your task is to evaluate whether this session exhibits expected communication patterns under its reported protocol (e.g., TLS, HTTPS, etc.).
    Focus on:
    1. Whether the encryption parameters (version, cipher) are common and valid.
    2. Whether the certificate data is consistent with expected communication partners.
    3. Whether the session shows signs of reuse (resumed sessions) or anomalies.
    4. Whether any protocol events suggest manipulation or non-standard behavior.

    Summarize your judgment: is the session consistent with expected protocol behavior, or are there signs of protocol misuse or obfuscation?


### prompt 4
    Please examine the following session's metadata and determine whether it shows any unexpected or suspicious behavior based on:
    - Communication patterns (bytes, packets, duration)
    - Protocol fields and values (e.g., connection state, resumption, anomalies)
    - Certificate or identity metadata (if any)

    Please explain whether the session seems typical for encrypted communication or whether there are reasons to suspect it deviates from normal patterns.

### prompt 5
    Analyze the following encrypted session and compare it against several similar past flows.

    Current Session:
    {current_feature_summary}

    Similar Sessions:
    {context_topk}

    Please:
    1. Describe the current session's behavior in detail (directionality, timing, structure).
    2. Identify any subtle deviations from the examples — not just obvious differences.
    3. Determine whether these deviations could indicate automation, tunneling, or malicious behavior.


## DoHBrw
### prompt 1
    You are a DNS-over-HTTPS (DoH) usage pattern analyst. Analyze the following encrypted DNS session and assess whether the session's behavior is consistent with typical DoH activity, as used by browsers and standard DNS clients.

    Please consider:
    1. Whether the packet timing and payload size are consistent with interactive DNS queries.
    2. Whether the flow rates suggest automated, periodic, or beaconing patterns.
    3. Whether the response time is normal or shows signs of manipulation or obfuscation.
    4. Whether the data patterns suggest unusual or suspicious use of the DoH protocol.

    Summarize your judgment in a short paragraph explaining whether this session likely reflects legitimate DNS behavior or abnormal DoH activity.


### prompt 2
    You are a protocol consistency analyst. Please examine the following session's metadata and determine whether it shows any unexpected or suspicious behavior based on:

    - Communication patterns (bytes, packets, duration)
    - Protocol fields and values (e.g., connection state, resumption, anomalies)
    - Certificate or identity metadata (if any)

    Please explain whether the session seems typical for encrypted communication or whether there are reasons to suspect it deviates from normal patterns.