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

# กำหนด device ก่อน
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# โหลด tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

# โหลด model และจัดการ meta tensor
try:
    # วิธีที่ 1: โหลด model โดยตรงไปยัง device ที่ต้องการ
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, 
        torch_dtype=torch.float32,
        device_map=None,  # ปิด automatic device mapping
        low_cpu_mem_usage=False
    )
    model = model.to(device)
    model.eval()
    print("Model loaded successfully with direct device mapping")
    
except Exception as e:
    print(f"Direct loading failed: {e}")
    try:
        # วิธีที่ 2: ใช้ to_empty() สำหรับ meta tensor
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        if hasattr(model, 'to_empty'):
            model = model.to_empty(device=device)
        else:
            model = model.to(device)
        model.eval()
        print("Model loaded successfully with to_empty method")
        
    except Exception as e2:
        print(f"to_empty method failed: {e2}")
        # วิธีที่ 3: โหลดแบบ CPU แล้ว move ทีละชิ้น
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        model = model.to(device)
        model.eval()
        print("Model loaded successfully with CPU first method")


with open(f"{MODEL_DIR}/label_names.json", encoding="utf-8") as f:
    LABELS = json.load(f)


def predict_with_probs(texts: list[str], threshold: float = THRESH, top_k: int = None):
    """
    คืน:
      - probs_sorted: รายชื่อคลาส + prob เรียงจากมากไปน้อย
      - chosen: รายการคลาสที่ "ผ่านเกณฑ์" (threshold หรือ top_k)
    """
    try:
        # ทำความสะอาดข้อความก่อนส่งเข้า tokenizer
        cleaned_texts = []
        for text in texts:
            # ลบอักขระที่อาจทำให้เกิดปัญหา
            cleaned_text = text.strip()
            if not cleaned_text:
                cleaned_text = "ข้อความว่าง"  # fallback text
            cleaned_texts.append(cleaned_text)
        
        # Tokenize with safe parameters
        enc = tok(
            cleaned_texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=MAX_LEN, 
            padding=True,
            add_special_tokens=True,  # เพิ่ม special tokens
            return_attention_mask=True,  # ให้ return attention mask
            return_token_type_ids=False  # ไม่ต้องการ token type ids สำหรับ RoBERTa
        )
        
        # ตรวจสอบว่า input_ids ไม่มีค่าที่เกิน vocab_size
        vocab_size = model.config.vocab_size
        if torch.any(enc['input_ids'] >= vocab_size):
            print(f"Warning: Found token IDs >= vocab_size ({vocab_size})")
            # แทนที่ token ที่เกินด้วย UNK token
            enc['input_ids'] = torch.clamp(enc['input_ids'], 0, vocab_size - 1)
        
        enc = {k: v.to(device) for k, v in enc.items()}

        sigmoid = torch.nn.Sigmoid()
        with torch.no_grad():
            logits = model(**enc).logits
            probs = sigmoid(logits).cpu().numpy()  # (B, C)
            
    except Exception as e:
        print(f"Error in tokenization or model prediction: {e}")
        # Return empty results if error occurs
        return [{"text": text, "probs_sorted": [], "chosen": []} for text in texts]

    results = []
    for i, text in enumerate(texts):
        p = probs[i]                                 # ความน่าจะเป็นต่อคลาส
        order = np.argsort(p)[::-1]                  # เรียงมาก→น้อย
        probs_sorted = [(LABELS[j], float(p[j])) for j in order]

        if top_k and top_k > 0:
            chosen = [LABELS[j] for j in order[:top_k]]
        else:
            chosen = []
            for name, prob in probs_sorted:
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
                try:
                    results = predict_with_probs(valid_texts, threshold=None, top_k=3)
                    print(results)
                    if results :
                        st.success("วิเคราะห์สำเร็จ!")
                        
                        st.subheader("ผลการทำนาย:")
                        for result in results:
                            with st.container(border=True):
                                st.markdown(f"**ข้อความ:** {result['text']}")
                                
                                tags = ' '.join([f"`{cat}`" for cat in result['chosen']])
                                st.markdown(f"**หมวดหมู่ที่ทำนาย:** {tags}")

                                with st.expander("ดูความน่าจะเป็นทั้งหมด"):
                                    df = pd.DataFrame(result['probs_sorted'], columns=['หมวดหมู่', 'ความน่าจะเป็น'])
                                    st.dataframe(df, width='stretch')
                    else:
                        st.error("เกิดข้อผิดพลาดในการทำนาย กรุณาลองใหม่อีกครั้ง")
                        
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {str(e)}")
                    st.info("กรุณาตรวจสอบว่าโมเดลถูกโหลดอย่างถูกต้อง")