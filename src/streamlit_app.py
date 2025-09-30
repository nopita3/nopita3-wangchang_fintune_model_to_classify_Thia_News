import json, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import pandas as pd

from huggingface_hub import hf_hub_download


st.set_page_config(
    page_title="Thai News Classification",
    page_icon="📰",
    layout="centered"
)

MODEL_DIR = "napadolP/wcberta-prachathai67k-best"
MAX_LEN   = 512
THRESH    = 0.5

# ใช้ cache_resource เพื่อโหลดโมเดลแค่ครั้งเดียว
@st.cache_resource
def load_model_and_tokenizer():
    """
    โหลด model, tokenizer, และ labels จาก Hugging Face Hub.
    ฟังก์ชันนี้จะทำงานแค่ครั้งเดียวและเก็บผลลัพธ์ไว้ใน cache
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. โหลด tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    # 2. โหลด model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        device_map=device, # ให้ transformers จัดการ device mapping
    )
    model.eval()
    print("Model loaded successfully.")

    # 3. โหลด label_names.json อย่างถูกวิธี
    try:
        label_path = hf_hub_download(repo_id=MODEL_DIR, filename="label_names.json")
        with open(label_path, encoding="utf-8") as f:
            labels = json.load(f)
        print("Labels loaded successfully.")
    except Exception as e:
        st.error(f"ไม่สามารถโหลด label_names.json ได้: {e}")
        labels = [f"class_{i}" for i in range(model.config.num_labels)] # Fallback

    return model, tok, labels, device

# --- โหลดโมเดล (จะถูก cache ไว้) ---
try:
    model, tok, LABELS, device = load_model_and_tokenizer()
    st.success("🟢 โมเดลพร้อมใช้งาน")
except Exception as e:
    st.error(f"🔴 เกิดข้อผิดพลาดร้ายแรงในการโหลดโมเดล: {e}")
    st.stop() # หยุดการทำงานของแอปถ้าโหลดโมเดลไม่ได้


def predict_with_probs(texts: list[str], model, tokenizer, labels, device, threshold: float = THRESH, top_k: int = None):
    """
    รับ model, tokenizer, labels ที่โหลดไว้แล้วมาใช้งาน
    """
    try:
        # Tokenize
        enc = tokenizer(texts, return_tensors="pt",
                  truncation=True,
                  max_length=MAX_LEN,
                  padding=True,
                  return_token_type_ids=False)
        enc = {k: v.to(device) for k, v in enc.items()}

        sigmoid = torch.nn.Sigmoid()
        with torch.no_grad():
            logits = model(**enc).logits
            probs = sigmoid(logits).cpu().numpy()

    except Exception as e:
        print(f"Error in tokenization or model prediction: {e}")
        return [{"text": text, "probs_sorted": [], "chosen": []} for text in texts]

    results = []
    for i, text in enumerate(texts):
        p = probs[i]
        order = np.argsort(p)[::-1]
        probs_sorted = [(labels[j], float(p[j])) for j in order]

        if top_k and top_k > 0:
            chosen = [labels[j] for j in order[:top_k]]
        else:
            chosen = []
            for name, prob in probs_sorted:
                if prob >= threshold:
                    chosen.append(name)
                else:
                    break

        results.append({
            "text": text,
            "probs_sorted": probs_sorted,
            "chosen": chosen
        })
    return results

st.title("📰 Thai News Classification")
st.markdown(f"แอปพลิเคชันสำหรับจำแนกประเภทข่าวภาษาไทย โดยใช้โมเดล `{MODEL_DIR}`")

st.write("ป้อนข้อความที่ต้องการทำนาย (หนึ่งข้อความต่อหนึ่งบรรทัด)")
input_texts = st.text_area("ใส่ข้อความที่นี่ (ไม่เกิน 512 คำต่อบรรทัด):", height=200, placeholder="ตัวอย่าง: กรมอุตุฯ เตือนวันนี้ทั่วไทยเจอฝนถล่ม...")

# --- เพิ่มส่วนควบคุมการทำนาย ---
st.write("---")
st.subheader("⚙️ ตัวเลือกการทำนาย")
prediction_mode = st.selectbox(
    "เลือกวิธีการทำนาย:",
    ("แสดง 3 อันดับแรก (Top-k)", "ตามเกณฑ์ความน่าจะเป็น (Threshold)")
)

# กำหนดค่าเริ่มต้น
top_k_value = None
threshold_value = None

if prediction_mode == "แสดง 3 อันดับแรก (Top-k)":
    top_k_value = 3
else:
    threshold_value = st.slider(
        "เลือกเกณฑ์ความน่าจะเป็น:",
        min_value=0.5,
        max_value=1.0,
        value=0.5, # ค่าเริ่มต้น
        step=0.05
    )
# ---------------------------------

if st.button("🧠 ทำนายผล"):
    if not input_texts.strip():
        st.warning("กรุณาป้อนข้อความเพื่อทำนายผล")
    else:
        # แบ่งตามบรรทัดและลบช่องว่าง
        valid_texts = [line.strip() for line in input_texts.split('\n') if line.strip()]

        # ตรวจสอบความยาว
        long_texts = [text for text in valid_texts if len(tok.tokenize(text)) > MAX_LEN]
        if long_texts:
            st.error(f"ข้อความบางส่วนมีความยาวเกิน {MAX_LEN} tokens และจะถูกตัดทอน")

        if valid_texts:
            with st.spinner("🤖 กำลังวิเคราะห์ข้อความ..."):

                try:
                    # ส่ง model, tok, LABELS, device และค่าที่เลือกเข้าไปในฟังก์ชัน
                    results = predict_with_probs(
                        valid_texts, 
                        model, 
                        tok, 
                        LABELS, 
                        device, 
                        threshold=threshold_value, 
                        top_k=top_k_value
                    )

                    if not results or not results[0]["probs_sorted"]:
                        st.error("ไม่พบผลการทำนาย")
                    else:
                        st.success("วิเคราะห์สำเร็จ!")
                        st.subheader("ผลการทำนาย:")
                        for result in results:
                            with st.container(border=True):
                                st.markdown(f"**ข้อความ:** {result['text']}")

                                tags = ' '.join([f"`{cat}`" for cat in result['chosen']])
                                st.markdown(f"**หมวดหมู่ที่ทำนาย:** {tags if tags else 'ไม่เข้าเกณฑ์'}")

                                with st.expander("ดูความน่าจะเป็นทั้งหมด"):
                                    df = pd.DataFrame(result['probs_sorted'], columns=['หมวดหมู่', 'ความน่าจะเป็น'])
                                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {str(e)}")
                    st.info("กรุณาตรวจสอบว่าโมเดลถูกโหลดอย่างถูกต้อง")