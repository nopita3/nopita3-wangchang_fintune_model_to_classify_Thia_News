import json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Thai News Classification",
    page_icon="📰",
    layout="centered"
)

MODEL_DIR = "wcberta-prachathai67k-best"  # โฟลเดอร์ที่ save_pretrained ไว้
MAX_LEN   = 512
THRESH    = 0.5   

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)


with open(f"{MODEL_DIR}/label_names.json", encoding="utf-8") as f:
    LABELS = json.load(f)


def predict_with_probs(texts: list[str], threshold :float = THRESH , top_k:int = None):
    """
    คืน:
      - probs_sorted: รายชื่อคลาส + prob เรียงจากมากไปน้อย
      - chosen: รายการคลาสที่ “ผ่านเกณฑ์” (threshold หรือ top_k)
    """
    enc = tok(texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        logits = model(**enc).logits
        probs = sigmoid(logits).cpu().numpy()  # (B, C)

    results = []
    for i, text in enumerate(texts):
        p = probs[i]                                 # ความน่าจะเป็นต่อคลาส
        order = np.argsort(p)[::-1]                  # เรียงมาก→น้อย
        probs_sorted = [(LABELS[j], float(p[j])) for j in order]

        if top_k and top_k > 0:
            chosen = [LABELS[j] for j in order[:top_k]]
        else:
            chosen = []
            for name , prob in probs_sorted:
                
                if prob >= threshold:
                    chosen.append(name)
                else:
                    break
            

        results.append({
            "text": text,
            "probs_sorted": probs_sorted,  # ดูอันดับทั้งหมดได้
            "chosen": chosen               # คำตอบหลายคลาส
        })
    return results

st.title("📰 Thai News Classification")
st.markdown("แอปพลิเคชันสำหรับจำแนกประเภทข่าวภาษาไทย โดยใช้โมเดล `wcberta-prachathai67k-best`")

st.write("ป้อนข้อความที่ต้องการทำนาย (หนึ่งข้อความต่อหนึ่งบรรทัด)")
input_texts = st.text_area("ใส่ข้อความที่นี่ (ไม่เกิน 512 คำต่อบรรทัด):", height=200, placeholder="ตัวอย่าง: กรมอุตุฯ เตือนวันนี้ทั่วไทยเจอฝนถล่ม...")

if st.button("🧠 ทำนายผล"):
    if not input_texts.strip():
        st.warning("กรุณาป้อนข้อความเพื่อทำนายผล")
    else:
        texts = [text.strip() for text in input_texts.split('\n') if text.strip()]
        
        valid_texts = []
        for i, text in enumerate(texts):
            if len(text.split()) > 512:
                st.error(f"ข้อความที่ {i+1} มีความยาวเกิน 512 คำ: '{text[:50]}...'")
            else:
                valid_texts.append(text)

        if valid_texts:
            with st.spinner("🤖 กำลังวิเคราะห์ข้อความ..."):
                results = predict_with_probs(valid_texts, threshold=None, top_k=3)
                
                st.success("วิเคราะห์สำเร็จ!")
                
                st.subheader("ผลการทำนาย:")
                for result in results:
                    with st.container(border=True):
                        st.markdown(f"**ข้อความ:** {result['text']}")
                        
                        tags = ' '.join([f"`{cat}`" for cat in result['chosen']])
                        st.markdown(f"**หมวดหมู่ที่ทำนาย:** {tags}")

                        with st.expander("ดูความน่าจะเป็นทั้งหมด"):
                            df = pd.DataFrame(result['probs_sorted'], columns=['หมวดหมู่', 'ความน่าจะเป็น'])
                            st.dataframe(df, use_container_width=True)
                        st.markdown(f"**หมวดหมู่ที่ทำนาย:** {tags}")

                        with st.expander("ดูความน่าจะเป็นทั้งหมด"):
                            df = pd.DataFrame(result['probs_sorted'], columns=['หมวดหมู่', 'ความน่าจะเป็น'])
                            st.dataframe(df, use_container_width=True)


