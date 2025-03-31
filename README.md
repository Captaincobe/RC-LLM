llm_flow_description/
├── main.py                      # 主程序
├── config.py                    # 参数配置
├── features.py                  # 特征预处理器
├── prompt_builder.py            # Prompt 构造器
├── llm_generator.py             # LLM 调用接口
├── embedder.py                  # 编码器接口（sentence-BERT）
├── data/
│   └── flow_data.csv            # 你的原始流量统计数据
├── outputs/
│   ├── descriptions.csv         # 每条流量对应的描述
│   └── embeddings.npy           # 每条描述的向量表示


preprocess.py --> dataset.csv
preload.py --> text_data.csv + test_data.csv
llm_generator.py --> text --> embedding
main.py
