import streamlit as st
import google.generativeai as genai
import json

# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="汉语二语写作多维分析 (CSL-Metrix)",
    page_icon="🇨🇳",
    layout="wide"
)

# --- 2. 核心 Prompt (这是工具的大脑) ---
# 这里集成了 Coh-Metrix 的 11 个维度逻辑
SYSTEM_PROMPT = """
你是一个计算语言学和二语习得(CSL)专家。你的任务是模拟 "Coh-Metrix 3.0" 对汉语二语文本进行分析。
请分析用户输入的文本，并严格输出合法的 JSON 格式。

分析维度说明：
1. 描述性：段落/句子/字数/平均句长。
2. 易读性(0-100)：叙述性(是否讲故事)、句法简单性、词的具体性。
3. 参照性衔接：名词重叠率(0-1)、论元重叠。
4. LSA语义：相邻句子语义相似度(0-1)。
5. 词汇多样性：TTR(类符/形符比)。
6. 连接词密度：每1000词中出现的连接词数量(因果/逻辑/转折/时间)。
7. 情景模式：时间衔接性、因果动词密度。
8. 句法复杂性：平均小句长度、主语前修饰语长度。
9. 句法模式：把字句/被字句/疑问句的使用情况。
10. 词汇信息：词性分布、平均HSK等级(难度代理)。
11. 综合可读性：预估HSK难度等级(如 HSK4, HSK6)。

重要：请直接返回 JSON 数据，不要包含 markdown 格式标记（如 ```json）。
JSON 结构模板：
{
  "summary": "一句话的综合简评",
  "basic_stats": {"words": 0, "sentences": 0, "avg_sent_len": 0},
  "scores": {
    "narrativity": 0, "syntactic_simplicity": 0, "referential_cohesion": 0, "semantic_similarity": 0
  },
  "readability": {"hsk_level": "HSK X", "score": 0},
  "details": "这里生成一段详细的 Markdown 文本，包含11个维度的详细表格分析，供用户阅读。"
}
"""

# --- 3. 侧边栏：API Key 输入 ---
with st.sidebar:
    st.header("🔧 设置")
    api_key = st.text_input("请输入 Google API Key", type="password")
    st.markdown("[👉 点击获取免费 API Key](https://aistudio.google.com/app/apikey)")
    st.info("提示：API Key 仅在内存中使用，不会被存储。")
    st.divider()
    st.caption("Designed for CSL Research")

# --- 4. 主界面 ---
st.title("🇨🇳 汉语二语写作多维分析工具")
st.markdown("基于 **Google Gemini** 构建，模拟 **Coh-Metrix** 指标体系。")

text_input = st.text_area("在此粘贴汉语文本：", height=250, placeholder="例如：去年夏天，我和朋友一起去了北京...")

if st.button("开始深度分析", type="primary"):
    if not api_key:
        st.error("请先在左侧输入 API Key 才能使用 AI 能力。")
    elif not text_input:
        st.warning("请输入需要分析的文本。")
    else:
        try:
            with st.spinner('AI 正在进行 11 个维度的计算（耗时约 10-20秒）...'):
                # 配置模型
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name="gemini-pro", # 使用 Pro 版本以获得更好的逻辑推理
                    system_instruction=SYSTEM_PROMPT
                )
                
                # 发送请求
                response = model.generate_content(f"请分析这段文本：\n{text_input}")
                
                # --- 5. 结果处理与展示 ---
                # 清洗数据（防止 AI 偶尔加 Markdown 标记）
                raw_text = response.text.replace("```json", "").replace("```", "").strip()
                
                try:
                    data = json.loads(raw_text)
                    
                    # 5.1 顶部关键指标卡片
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("预估 HSK 难度", data['readability']['hsk_level'])
                    col2.metric("叙述性 (Narrativity)", data['scores']['narrativity'])
                    col3.metric("语义连贯性 (LSA)", data['scores']['semantic_similarity'])
                    col4.metric("词汇多样性 (TTR)", data['basic_stats'].get('words', 'N/A')) # 这里仅作示例，实际可取TTR
                    
                    st.success(f"分析完成！综合评价：{data['summary']}")
                    st.divider()
                    
                    # 5.2 详细分析报告 (Markdown)
                    st.subheader("📊 详细分析报告")
                    st.markdown(data['details'])
                    
                    # 5.3 原始 JSON 数据 (供研究用)
                    with st.expander("查看原始 JSON 数据"):
                        st.json(data)

                except json.JSONDecodeError:
                    st.error("数据解析失败，展示原始 AI 回复：")
                    st.markdown(response.text)

        except Exception as e:
            st.error(f"连接出错: {e}")
# --- 在 app.py 的最后添加这段代码 ---

with st.sidebar:
    st.divider()
    st.header("🕵️‍♂️ 调试工具")
    if st.button("检查可用模型列表"):
        if not api_key:
            st.error("请先输入 API Key")
        else:
            try:
                genai.configure(api_key=api_key)
                st.write("正在查询 Google 服务器...")
                available_models = []
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
                
                if available_models:
                    st.success("查询成功！你的 API Key 支持以下模型：")
                    st.code("\n".join(available_models))
                    st.info("请复制上面列表中的任意一个名字（例如 models/gemini-pro），填入代码的 model_name 中。")
                else:
                    st.error("没有找到支持 generateContent 的模型。可能 API Key 无效。")
            except Exception as e:
                st.error(f"查询失败: {e}")
